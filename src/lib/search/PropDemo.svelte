<script lang="ts">
	/**
	 * The demo's first part: the propagator, one layer that sums every walk of every length with one linear solve,
	 * trained on 49-town graphs and run here on the benchmark's questions at 49, 125 and 250 towns. The map lights as the
	 * walks come back from the goal one length at a time, then holds on the solve and draws the road the layer chooses.
	 * Beside it: the current town's roads and the walk weight that reached each, and the learned gates.
	 */
	import { onMount } from 'svelte';
	import Icon from '#lib/ui/Icon.svelte';
	import * as I from '#lib/ui/icons.ts';
	import MapView from './MapView.svelte';
	import RoadsChart from './RoadsChart.svelte';
	import GateStrip from './GateStrip.svelte';
	import TokenPanel from './TokenPanel.svelte';
	import ReadoutPanel from './ReadoutPanel.svelte';
	import { Walks, LOOKS, TOWNS, type Opening, type Towns } from './walks.svelte.ts';
	import { lightAt } from './walks.ts';
	import { loadMap } from './maps.ts';
	import { renameLayout, type MapLayout } from './map.ts';
	import { current, encode, vertices } from './tokens.ts';
	import { goodSteps } from './graph.ts';

	let { opening, layout250, eq }: { opening: Opening; layout250: MapLayout; eq: string } = $props();

	// the opening seeds the demo once; later runs replace it
	// svelte-ignore state_referenced_locally
	const s = new Walks(opening);
	/** before the page's script runs, the map is drawn for a fixed width and scales with its box */
	let mounted = $state(false);
	onMount(() => {
		mounted = true;
		// /demo?towns=<49|125|250>&question=<n> opens that test question instead of the recorded one
		const u = new URLSearchParams(location.search);
		const towns = Number(u.get('towns'));
		const q = Number(u.get('question'));
		s.start(
			(TOWNS as readonly number[]).includes(towns)
				? { towns: towns as Towns, index: Number.isInteger(q) && q >= 1 ? q - 1 : 0 }
				: undefined
		);
		if (import.meta.env.DEV) Object.assign(globalThis, { walks: s });
		return () => s.stop();
	});

	// the 250-town map arrives with the page; the others are fetched
	// svelte-ignore state_referenced_locally
	let layout = $state.raw<MapLayout | null>(layout250);
	let mapError = $state<string | null>(null);
	$effect(() => {
		const i = s.index;
		const V = s.towns;
		if (V === 250 && i === 0) {
			layout = layout250;
			return;
		}
		loadMap(i, V)
			.then((m) => {
				if (s.index === i && s.towns === V) {
					layout = m;
					mapError = null;
				}
			})
			.catch((e: Error) => (mapError = e.message));
	});

	const f = $derived(s.frames);
	const light = $derived(f && s.phase !== 'waiting' ? lightAt(f, s.M, s.solved) : null);
	const cur = $derived(current(s.problem));
	const answer = $derived(s.run?.answer ?? f?.answer ?? null);
	const chosen = $derived(
		s.chose > 0 && answer !== null ? { u: cur, v: answer, t: s.chose } : null
	);
	const good = $derived(goodSteps(s.problem));
	const walkLen = $derived(f ? Math.min(f.reach, Math.floor(s.M + 1e-4)) : 0);
	let hoverRoad = $state<{ u: number; v: number } | null>(null);
	const looks = $derived(LOOKS[s.towns]);
	const hasRandom = $derived(s.questions.some((q) => q.lookahead === null));
	/** the scrubber: walk length 0 … reach, then one more step for the solve */
	const scrub = $derived(f ? (s.solved > 0 ? f.reach + s.solved : s.M) : 0);
	const roads = $derived(
		s.run && s.run.current === cur
			? s.run.roads
			: s.problem.edges.filter(([u]) => u === cur).map(([, v]) => ({ v, x: null, p: null }))
	);
	const busy = $derived(s.phase === 'waiting');
	/** the map under the names on screen (renamed when the visitor renames the towns) */
	const shown = $derived(layout && s.names ? renameLayout(layout, s.names) : layout);
	/** the tokens of the run on screen; while the layer runs, the new input */
	const tokens = $derived(
		s.run && !busy ? s.run.tokens : (encode(s.problem, s.spec) ?? s.run?.tokens ?? [])
	);
	let query = $state(false);
	/** after a renaming: the road the layer chose, and whether it is the road it chose before, renamed */
	const same = $derived.by(() => {
		const r = s.renaming;
		if (!r || !s.run || busy || s.recorded) return null;
		const was = r.step.get(r.before);
		return { now: s.run.answer, before: r.before, ok: was === s.run.answer };
	});
</script>

<section class="part" aria-labelledby="prop-title" data-run={s.runs}>
	<header class="head">
		<div>
			<p class="kicker">The propagator</p>
			<h2 id="prop-title">One layer, one linear solve</h2>
			<p class="lede">
				Each question is a graph from the search benchmark of Saparov et al., drawn as a map: every
				road is one edge, one way, from a town to the next. From the current town (dark), the layer
				must name the road toward the goal (rose ring). It sums the walks of every length at once,
				<i>X</i> = <i>S</i> (<i>I</i> − <i>G</i>)<sup>−1</sup>: the source <i>S</i> sits on the
				goal, the learned gates <i>G</i> weigh each step back along a road, and the weight of the walks
				that reach a town lights it. Trained on graphs of 49 vertices, it answered every test question
				at every lookahead on graphs of 250 vertices, in each of its three runs.
			</p>
		</div>
		<div class="model" aria-live="polite">
			{#if s.ready === true}
				<p class="note num">
					The layer runs in this browser: {s.params?.toLocaleString('en-US')} parameters{s.ms !==
					null
						? `, ${Math.max(1, Math.round(s.ms))} ms a forward pass`
						: ''}.
				</p>
			{:else if s.ready === false}
				<p class="note err">{s.error}</p>
			{:else}
				<p class="note">Playing the recorded run while the layer loads (300 KB).</p>
			{/if}
		</div>
	</header>

	<div class="controls">
		<span class="opts" role="group" aria-label="Map size">
			<span class="hint">Map</span>
			{#each TOWNS as V (V)}
				<button
					class="plain num"
					aria-pressed={s.towns === V}
					disabled={!s.have[V]}
					onclick={() => s.setTowns(V)}>{V} towns</button
				>
			{/each}
		</span>
		<span class="opts" role="group" aria-label="Pick a question by lookahead">
			<span
				class="hint"
				title="How many roads ahead one must look to tell the right first road from the others"
				>Lookahead</span
			>
			{#each looks as L (L)}
				<button
					class="plain num"
					aria-pressed={s.lookahead === L && !s.edited}
					disabled={!s.questions.length}
					onclick={() => s.pick(L, s.lookahead === L)}>{L}</button
				>
			{/each}
			{#if hasRandom}
				<button
					class="plain"
					aria-pressed={s.lookahead === null && !s.edited}
					onclick={() => s.pick(null, s.lookahead === null)}
					title="The benchmark's random-graph test set, whose roads merge">random graph</button
				>
			{/if}
			{#if s.questions.filter((q) => q.lookahead === s.lookahead).length > 1}
				<button
					class="plain ic"
					onclick={() => s.pick(s.lookahead, true)}
					title="Another test question with this lookahead"
					><Icon icon={I.Shuffle} size={15} />another</button
				>
			{/if}
		</span>
		<span class="opts acts">
			<button
				class="plain ic"
				disabled={!s.ready}
				onclick={() => s.rename()}
				title="Give every town a new random name and write the roads in a new random order: the same question, other tokens"
				><Icon icon={I.Rename} size={15} />Rename the towns</button
			>
			<button class="plain ic strong" onclick={() => s.replay()}
				><Icon icon={I.Replay} size={15} />Replay</button
			>
			{#if s.edited}
				<button class="plain ic" onclick={() => s.reset()}
					><Icon icon={I.Undo} size={15} />The question’s goal</button
				>
			{/if}
		</span>
	</div>

	<div class="stage">
		<figure class="stage-map">
			<figcaption class="cap">
				Next road from <b>{cur}</b> toward the goal <b class="goal">{s.problem.goal}</b>?
			</figcaption>
			<div class="map-box">
				{#if shown}
					<MapView
						layout={shown}
						question={s.problem}
						problem={s.problem}
						{light}
						front={s.solved > 0 ? null : s.M}
						{chosen}
						trees
						fixed={mounted ? undefined : 860}
						numbers={s.towns === 250 ? 'key' : 'all'}
						bind:hoverRoad
						onroadhover={() => {}}
						marked={query ? [s.problem.start, s.problem.goal] : null}
						onvertex={(v) => s.setGoal(v)}
						label="A map of the question: {vertices(s.problem)
							.length} towns. The weight of the walks to the goal lights the roads; click a town to make it the goal, hover a road to read its gates"
					/>
				{:else if mapError}
					<p class="note err">{mapError}</p>
				{/if}
			</div>
			{#if f}
				<div class="scrub">
					<label class="hint" for="walk-len">Walks</label>
					<input
						id="walk-len"
						type="range"
						min="0"
						max={f.reach + 1}
						step="0.01"
						value={scrub}
						aria-valuetext={s.solved > 0
							? 'every walk: the solve'
							: `walks of up to ${walkLen} roads`}
						oninput={(e) => s.seek(Number(e.currentTarget.value))}
					/>
					<span class="hint num len"
						>{s.solved > 0
							? 'all lengths'
							: `≤ ${walkLen} ${walkLen === 1 ? 'road' : 'roads'}`}</span
					>
				</div>
			{/if}
			<div class="status">
				<p class="verdict">
					{#if busy}
						Running the layer on this question…
					{:else if s.chose > 0 && answer !== null}
						The layer answers <strong class={good.includes(answer) ? 'ok' : 'bad'}>{answer}</strong
						>{#if good.includes(answer)}: the road that leads to the goal.{:else if !roads.some((r) => r.v === answer)}:
							no road leads there from {cur}.{:else}: no route to the goal from there.{/if}
					{:else if s.solved > 0}
						Every walk at once, by one linear solve: <i>X</i> = <i>S</i> (<i>I</i> − <i>G</i>)<sup
							>−1</sup
						>.
					{:else}
						The walks of up to <span class="num">{walkLen}</span>
						{walkLen === 1 ? 'road' : 'roads'} from the goal:
						<span class="sum"
							>Σ<sub><i>m</i> ≤ <span class="num">{walkLen}</span></sub>
							<i>S G<sup>m</sup></i></span
						>
					{/if}
				</p>
				<p class="facts">
					{#if s.note}
						<span class="msg">{s.note}</span>
					{:else if same}
						<span class="msg"
							>Every town renamed and the roads reordered: the layer answers <b>{same.now}</b>, the
							town it called {same.before} before{same.ok
								? ', the same road under its new name.'
								: '; a different road.'}</span
						>
					{:else}
						<span
							>{s.edited
								? 'your goal, on'
								: s.lookahead === null
									? 'random graph,'
									: `lookahead ${s.lookahead},`}
							test question {s.index + 1} of {s.questions.length || 1}</span
						><span>{vertices(s.problem).length} towns, {s.problem.edges.length} roads</span><span
							>{s.recorded ? 'recorded run' : 'live'}</span
						>
					{/if}
				</p>
			</div>
		</figure>

		<div class="side">
			<figure>
				<figcaption class="cap">
					What the layer reads
					<span class="sub"
						>Only this list of tokens, as the benchmark writes it: the map is our drawing, never its
						input, and the town numbers are names drawn at random for each question. Hover a road’s
						tokens to find it on the map.</span
					>
				</figcaption>
				<div class="panel-box io">
					<TokenPanel
						{tokens}
						V={s.towns}
						bind:hoverRoad
						onquery={(on) => (query = on)}
						edited={s.edited}
					/>
				</div>
			</figure>
			<figure>
				<figcaption class="cap">
					What it writes
					<span class="sub">one score for every town name; nothing is drawn by the model</span>
				</figcaption>
				<div class="panel-box io">
					{#if s.run}
						<ReadoutPanel
							names={s.run.names}
							logits={s.run.logits}
							current={s.run.current}
							pending={busy}
						/>
					{/if}
				</div>
			</figure>
			<figure>
				<figcaption class="cap">
					The current town’s roads
					<span class="sub">the walk weight that has come back to the town each road leads to</span>
				</figcaption>
				<div class="panel-box">
					{#if f}
						<RoadsChart
							frames={f}
							{roads}
							M={s.M}
							solved={s.solved}
							chosen={s.chose}
							hover={hoverRoad && hoverRoad.u === cur ? hoverRoad.v : null}
						/>
					{/if}
				</div>
			</figure>
			<figure>
				<figcaption class="cap">
					Learned gates
					<span class="sub"
						>the weight of one step along each road of this map, both ways; hover or tap a road to
						read it</span
					>
				</figcaption>
				<div class="panel-box">
					{#if s.run}
						<GateStrip gates={s.run.gates} hover={hoverRoad} />
					{/if}
				</div>
			</figure>
		</div>
	</div>
	<p class="how hint">
		Click a town to make it the goal: the walks start from there, and the query token changes. Hover
		or tap a road to read its gates and find its tokens; drag the walks slider to add the walks one
		length at a time; rename the towns to give the layer the same question in other tokens.
	</p>

	<div class="explain">
		<!-- eslint-disable-next-line svelte/no-at-html-tags -- the paper's own equation, set by KaTeX at build time -->
		<div class="eq">{@html eq}</div>
		<p>
			In each of four channels the layer forms gates <i>G</i> and a source <i>S</i> from the
			normalised pair state, caps every row of <i>G</i> so that it sums to at most <i>t</i><sub
				>0</sub
			> = 0.99, and solves for the walk sum. The cap keeps the spectral radius below one on any graph,
			so the walk sum converges. The map shows the channel that carries the answer: its gates are near
			one on roads walked back and near zero on roads walked forward and between towns with no road. The
			layer has 28,472 parameters. This page runs the same forward pass in TypeScript, checked against
			the training code’s own outputs at 49, 125 and 250 towns to within 0.001.
		</p>
	</div>
</section>

<style>
	.part {
		--col-gap: 44px;
	}
	.head {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		gap: 12px 40px;
	}
	.kicker {
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--color-rose);
	}
	h2 {
		margin-top: 4px;
		font-size: var(--text-head);
		font-weight: 500;
		letter-spacing: -0.02em;
		line-height: 1.1;
	}
	.lede {
		margin-top: 10px;
		max-width: 100ch;
		font-size: var(--text-ui);
		line-height: 1.6;
		color: var(--color-ink-soft);
	}
	.lede i,
	.verdict i,
	.explain i {
		font-family: var(--font-serif);
		font-size: 1.06em;
	}
	sup,
	sub {
		font-size: 0.72em;
	}
	.model {
		max-width: 340px;
		min-height: 34px;
		text-align: right;
		text-wrap: balance;
	}
	.note {
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.err {
		color: var(--color-rose);
	}
	.controls {
		display: flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 10px 30px;
		margin-top: 22px;
	}
	.opts {
		display: inline-flex;
		align-items: center;
		flex-wrap: wrap;
		gap: 4px 13px;
	}
	.opts .plain {
		font-size: var(--text-sm);
	}
	.strong {
		color: var(--color-ink);
		font-weight: 500;
	}
	.acts {
		margin-left: auto;
	}
	.stage {
		display: grid;
		grid-template-columns: minmax(0, 860px) minmax(320px, 1fr);
		gap: 28px var(--col-gap);
		margin-top: 20px;
		align-items: start;
	}
	figure {
		margin: 0;
		min-width: 0;
	}
	.cap {
		font-size: var(--text-sm);
		font-weight: 500;
		color: var(--color-ink);
		line-height: 1.35;
	}
	.cap b {
		font-weight: 600;
	}
	.cap .goal {
		color: var(--color-rose);
	}
	.cap .sub {
		display: block;
		font-weight: 400;
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.map-box {
		margin-top: 10px;
		overflow: hidden;
		border: 1px solid var(--color-edge);
		border-radius: 6px;
	}
	.scrub {
		display: grid;
		grid-template-columns: auto minmax(0, 1fr) 84px;
		align-items: center;
		gap: 12px;
		margin-top: 12px;
	}
	.scrub input {
		width: 100%;
		accent-color: var(--color-rose);
	}
	.len {
		text-align: right;
	}
	.status {
		display: grid;
		gap: 2px;
		margin-top: 8px;
		min-height: 44px;
	}
	.verdict {
		font-size: var(--text-ui);
		line-height: 20px;
		color: var(--color-ink);
	}
	.verdict strong {
		font-weight: 600;
	}
	.verdict .ok {
		color: var(--color-rose);
	}
	.verdict .bad {
		color: var(--color-cobalt);
	}
	.sum {
		margin-left: 6px;
		color: var(--color-ink-soft);
	}
	.facts {
		font-size: var(--text-xs);
		line-height: 20px;
		color: var(--color-ink-soft);
	}
	.facts span + span::before {
		content: ' · ';
	}
	.facts .msg {
		color: var(--color-ink);
	}
	.side {
		display: grid;
		gap: 34px;
		min-width: 0;
		padding-top: 2px;
	}
	.panel-box {
		margin-top: 12px;
		min-height: 120px;
	}
	.panel-box.io {
		min-height: 0;
		margin-top: 10px;
	}
	.facts .msg b {
		color: var(--color-rose);
	}
	.how {
		margin-top: 14px;
	}
	.explain {
		display: grid;
		gap: 6px;
		max-width: 860px;
		margin-top: 40px;
	}
	.eq {
		overflow-x: auto;
		overflow-y: hidden;
		color: var(--color-ink);
	}
	.explain p {
		max-width: 96ch;
		font-size: var(--text-sm);
		line-height: 1.6;
		color: var(--color-ink-soft);
	}
	@media (max-width: 1099px) {
		.head,
		.stage {
			grid-template-columns: minmax(0, 1fr);
		}
		.model {
			text-align: left;
		}
		.side {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
	@media (max-width: 719px) {
		.part {
			--col-gap: 20px;
		}
		.controls {
			gap: 10px 20px;
		}
		.acts {
			margin-left: 0;
		}
		.side {
			grid-template-columns: minmax(0, 1fr);
		}
	}
</style>
