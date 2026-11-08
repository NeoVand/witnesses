<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import Icon from '#lib/ui/Icon.svelte';
	import * as I from '#lib/ui/icons.ts';
	import { Search, MOVE, SIZES, MAX_K, type Size } from '#lib/search/demo.svelte.ts';
	import MapView from '#lib/search/MapView.svelte';
	import PairMap from '#lib/search/PairMap.svelte';
	import LayerChart from '#lib/search/LayerChart.svelte';
	import DepthChart from '#lib/search/DepthChart.svelte';
	import ReachChart from '#lib/search/ReachChart.svelte';
	import PropDemo from '#lib/search/PropDemo.svelte';
	import Meta from '#lib/ui/Meta.svelte';
	import { place } from '#lib/search/layout.ts';
	import { loadMap } from '#lib/search/maps.ts';
	import { renameLayout, route, type MapLayout } from '#lib/search/map.ts';
	import { current, encode, vertices } from '#lib/search/tokens.ts';
	import TokenPanel from '#lib/search/TokenPanel.svelte';
	import ReadoutPanel from '#lib/search/ReadoutPanel.svelte';
	import { goodSteps } from '#lib/search/graph.ts';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const s = new Search();
	onMount(() => {
		// /demo?q=<n> opens test question n (1 to 300) instead of the recorded opening; &size=125 the 125-town questions,
		// &model=loop the repeated layer, &k=<1..10> its iterations
		const u = new URLSearchParams(location.search);
		const q = Number(u.get('q'));
		const size = (SIZES as readonly number[]).includes(Number(u.get('size')))
			? (Number(u.get('size')) as Size)
			: undefined;
		s.start(Number.isInteger(q) && q >= 1 ? q - 1 : undefined, {
			size,
			model: u.get('model') === 'loop' ? 'loop' : undefined,
			K: Number(u.get('k')) || undefined
		});
		if (import.meta.env.DEV) Object.assign(globalThis, { search: s });
		return () => s.stop();
	});

	/** the map of the question on screen (an edit keeps its question's map) */
	let layout = $state.raw<MapLayout | null>(null);
	let mapError = $state<string | null>(null);
	$effect(() => {
		const i = s.index;
		const V = s.size;
		loadMap(i, V)
			.then((m) => {
				if (s.index === i && s.size === V) {
					layout = m;
					mapError = null;
				}
			})
			.catch((e: Error) => (mapError = e.message));
	});

	const LOOKAHEADS: Record<Size, number[]> = {
		49: [1, 2, 3, 4, 6, 8, 12, 16, 24],
		125: [1, 2, 4, 8, 16, 24, 32, 48, 62]
	};
	const looks = $derived(LOOKAHEADS[s.size]);
	const hasRandom = $derived(!s.questions.length || s.questions.some((q) => q.lookahead === null));
	const loop = $derived(s.model === 'loop');
	const unit = $derived(loop ? 'iteration' : 'layer');
	/** the most a pair can know at the playhead: routes of 2^l roads after l joins (shown for the repeated layer) */
	const reach = $derived(loop && s.trace ? 2 ** Math.round(s.layer) : null);
	/** a route on this map has at most (towns - 1) roads */
	const sees = $derived(
		reach === null
			? null
			: reach >= vertices(s.problem).length - 1
				? 'every route on this map'
				: `routes of up to ${reach} ${reach === 1 ? 'road' : 'roads'}`
	);
	const order = $derived(place(vertices(s.question), s.question.edges, s.question.start).order);
	const cur = $derived(current(s.problem));
	/** the branches of the town the shown run asked about (the walk's last question once it lands on the goal) */
	const asked = $derived(s.trace?.current ?? cur);
	const branches = $derived(s.problem.edges.filter(([u]) => u === asked).map(([, v]) => v));
	const answered = $derived(
		s.phase !== 'sweep' && s.phase !== 'walk' && s.layer >= s.depth - 0.001
	);
	const good = $derived(goodSteps(s.problem));
	const total = $derived(s.questions.length || 300);
	let hovered = $state<{ u: number; v: number; p: number } | null>(null);
	let hoverTown = $state<number | null>(null);
	const hoverRoute = $derived(hovered ? route(s.problem.edges, hovered.u, hovered.v) : null);
	const layerNow = $derived(Math.round(s.layer));

	const fmt = (p: number) => (p >= 0.995 ? '1.00' : p.toFixed(2));
	/** the map under the names on screen */
	const shown = $derived(layout && s.names ? renameLayout(layout, s.names) : layout);
	/** the tokens of the forward pass on screen (each step of the walk is one more pass, its path one token longer) */
	const tokens = $derived(
		s.trace?.tokens.length ? s.trace.tokens : (encode(s.problem, s.spec) ?? [])
	);
	const walked = $derived(
		Math.max(0, (s.trace?.tokens.length ? s.problem.path.length : 0) - s.question.path.length)
	);
	/** the read-out the model wrote on this pass: its logits when live, the recorded probabilities otherwise */
	const out = $derived.by(() => {
		const t = s.trace;
		if (!t) return null;
		const m = t.names.length;
		const ci = t.names.indexOf(t.current);
		const last = t.maps[t.maps.length - 1];
		return {
			names: t.names,
			logits: t.logits ?? null,
			probs: ci >= 0 && last ? last.slice(ci * m, ci * m + m) : null,
			current: t.current
		};
	});
	let road2 = $state<{ u: number; v: number } | null>(null);
	let query2 = $state(false);
	/** what the current town's row names at the playhead */
	const leadNow = $derived.by(() => {
		const t = s.trace;
		if (!t) return null;
		const m = t.names.length;
		const ci = t.names.indexOf(t.current);
		if (ci < 0) return null;
		const lo = Math.max(0, Math.min(s.depth, Math.floor(s.layer)));
		const hi = Math.min(s.depth, lo + 1);
		const f = s.layer - lo;
		let best = 0;
		let bp = -1;
		for (let k = 0; k < m; k++) {
			const p = t.maps[lo][ci * m + k] + (t.maps[hi][ci * m + k] - t.maps[lo][ci * m + k]) * f;
			if (p > bp) {
				bp = p;
				best = k;
			}
		}
		return { v: t.names[best], p: bp };
	});
</script>

<Meta
	title="Demo · Attention Is Only as Good as Its Witnesses"
	description="One propagator layer, trained on 49-vertex graphs, searches the benchmark's maps of up to 250 towns in your browser; then the pair state and the OR join, layer by layer."
	path="/demo"
/>

<main class="demo">
	<PropDemo opening={data.opening} layout250={data.layout250} eq={data.eq.prop} />

	<section class="part2" aria-labelledby="depth-title" data-run={s.runs}>
		<header class="head">
			<div>
				<p class="kicker">Layer by layer</p>
				<h2 id="depth-title">The depth law</h2>
				<p class="lede">
					Before one solve, depth. A pair state keeps a state for every pair of towns, and each
					layer sums over the two-step walks u → k → v through every middle town k. By the depth law
					(Theorem 1 of the paper), <i>L</i> such layers reach exactly 2<sup><i>L</i></sup> − 1
					roads beyond the current town’s next town, lookahead 2<sup><i>L</i></sup>, for every
					<i>L</i>; trained pair states over names answer at least 95% at every lookahead up to 2<sup
						><i>L</i></sup
					> and fall just past it. The 8 layers here are the pair state over vertex names trained at 49
					vertices (39,504 parameters).
				</p>
				<p class="lede">
					The second model is the OR join: one pair layer, repeated, whose sum over the middle town
					is read through an OR, 1 − e<sup>−<i>Z</i></sup>, so it says only whether some middle town
					connects a pair. Trained at 49 vertices and tested at 125, it answered at least 99.6% at
					every lookahead.
				</p>
			</div>
			<div class="model" aria-live="polite">
				{#if s.live === true}
					<p class="note num">
						{loop ? 'The OR join' : 'The 8 layers'} run in this browser: {s.params?.toLocaleString(
							'en-US'
						)} parameters{s.ms !== null
							? `, ${Math.max(1, Math.round(s.ms))} ms a forward pass${loop ? ` of ${s.K} iterations` : ''}`
							: ''}.
					</p>
				{:else if s.live === false}
					<p class="note err">{s.loadError}</p>
				{:else if loop}
					<p class="note">Loading the OR join (260 KB)…</p>
				{:else}
					<p class="note">Playing a recorded run while the model loads (450 KB).</p>
				{/if}
			</div>
		</header>

		<div class="controls models">
			<span class="opts" role="group" aria-label="Model">
				<span class="hint">Model</span>
				<button
					class="plain"
					aria-pressed={s.model === 'stack'}
					disabled={s.size !== 49}
					title={s.size !== 49
						? 'The 8 layers read only 49-town maps: they were trained with positions for that input length'
						: 'Eight different layers, each joining every pair through every middle town'}
					onclick={() => s.setModel('stack')}>8 layers</button
				>
				<button
					class="plain"
					aria-pressed={s.model === 'loop'}
					title="The OR join: one layer whose join is a soft OR, run again and again"
					onclick={() => s.setModel('loop')}>one layer, repeated</button
				>
			</span>
			<span class="opts" role="group" aria-label="Map size">
				<span class="hint">Map</span>
				{#each SIZES as V (V)}
					<button
						class="plain num"
						aria-pressed={s.size === V}
						disabled={!s.have[V]}
						onclick={() => s.setSize(V)}>{V} towns</button
					>
				{/each}
			</span>
			{#if loop}
				<span class="opts" role="group" aria-label="How many times the layer runs">
					<span class="hint" title="How many times the one layer runs on the question"
						>Repeated</span
					>
					{#each Array.from({ length: MAX_K }, (_, i) => i + 1) as k (k)}
						<button class="plain num" aria-pressed={s.K === k} onclick={() => s.setK(k)}>{k}</button
						>
					{/each}
					<span class="hint">times</span>
				</span>
			{/if}
		</div>

		<div class="controls">
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
						disabled={!s.questions.length}
						onclick={() => s.pick(null, s.lookahead === null)}
						title="The benchmark's random-graph test set, whose roads merge and whose lookahead is not controlled"
						>random graph</button
					>
				{/if}
				<button
					class="plain ic"
					disabled={!s.questions.length}
					onclick={() => s.pick(s.lookahead, true)}
					title="Another test question with this lookahead"
					><Icon icon={I.Shuffle} size={15} />another</button
				>
			</span>
			<span class="opts acts">
				<button
					class="plain ic"
					disabled={!s.live}
					onclick={() => s.rename()}
					title="Give every town a new random name and write the roads in a new random order: the same question, other tokens"
					><Icon icon={I.Rename} size={15} />Rename the towns</button
				>
				<button class="plain ic strong" onclick={() => s.restart(s.question, s.edited)}
					><Icon icon={I.Replay} size={15} />Run again</button
				>
				{#if s.edited}
					<button class="plain ic" onclick={() => s.reset()}
						><Icon icon={I.Undo} size={15} />Undo edits</button
					>
				{/if}
			</span>
			<span class="opts pace" role="group" aria-label="Pace">
				<button class="plain ic" aria-pressed={s.pace === 'slow'} onclick={() => (s.pace = 'slow')}
					><Icon icon={I.Slow} size={15} />layer by layer</button
				>
				<button class="plain ic" aria-pressed={s.pace === 'fast'} onclick={() => (s.pace = 'fast')}
					><Icon icon={I.Fast} size={15} />fast</button
				>
			</span>
		</div>

		<div class={['stage', { wide: s.size !== 49 }]}>
			<figure class="stage-map">
				<figcaption class="cap">
					{#if cur === s.problem.goal}
						The walk has reached the goal, <b class="goal">{cur}</b>.
					{:else}
						Next road from <b>{cur}</b> toward the goal <b class="goal">{s.problem.goal}</b>?
					{/if}
				</figcaption>
				<div class="map-box">
					{#if shown}
						<MapView
							layout={shown}
							question={s.question}
							problem={s.problem}
							trace={s.trace}
							layer={s.layer}
							{reach}
							{answered}
							moveMs={MOVE[s.pace]}
							pair={hovered}
							bind:hoverTown
							bind:hoverRoad={road2}
							onroadhover={() => {}}
							marked={query2 ? [s.problem.start, s.problem.goal] : null}
							onvertex={(v) => s.setGoal(v)}
							onedge={(u, v) => s.removeEdge(u, v)}
							onadd={(u, v) => s.addEdge(u, v)}
						/>
					{:else if mapError}
						<p class="note err">{mapError}</p>
					{/if}
				</div>
				<div class="status">
					<p class="verdict">
						{#if !s.trace}
							Reading the model…
						{:else if s.phase === 'walk'}
							Driving to <strong>{cur}</strong>…
						{:else if !answered && leadNow}
							After {unit} <span class="num">{layerNow}</span> of {s.depth}{#if sees}, seeing {sees},{/if}
							it would answer
							<strong>{leadNow.v}</strong> <span class="soft num">(p = {fmt(leadNow.p)})</span>
						{:else if s.phase === 'done' && cur === s.problem.goal}
							Reached the goal in <span class="num">{s.taken.length}</span>
							{s.taken.length === 1 ? 'road' : 'roads'}, one forward pass each.
						{:else if s.trace}
							The model answers <strong class={good.includes(s.trace.answer) ? 'ok' : 'bad'}
								>{s.trace.answer}</strong
							>{#if good.includes(s.trace.answer)}: a road that leads to the goal.{:else if !branches.includes(s.trace.answer)}:
								no road leads there from {cur}.{:else}: no route to the goal from there.{/if}
						{/if}
					</p>
					<p class="facts">
						{#if s.note}
							<span class="msg">{s.note}</span>
						{:else if s.renaming?.after !== undefined}
							<span class="msg"
								>Every town renamed and the roads reordered: from the start the model answers <b
									>{s.renaming.after}</b
								>, the town it called {s.renaming.before} before{s.renaming.step.get(
									s.renaming.before
								) === s.renaming.after
									? ', the same road under its new name.'
									: '; a different road.'}</span
							>
						{:else}
							<span
								>{s.edited
									? 'your edit of'
									: s.lookahead === null
										? 'random graph,'
										: `lookahead ${s.lookahead},`}
								test question {s.index + 1} of {total}</span
							><span>{vertices(s.question).length} towns, {s.question.edges.length} roads</span
							><span class="tints"
								>towns tinted by the current town’s read-out: <i class="rose">rose</i> above its
								average, <i class="cobalt">cobalt</i> below{loop
									? '; darker roads: within reach of the current town'
									: ''}</span
							>
						{/if}
					</p>
				</div>
				<div class="io">
					<figure>
						<figcaption class="cap">
							What the model reads
							<span class="sub"
								>Only these tokens, never the map; the town numbers are names drawn at random for
								each question. Each road walked is one more forward pass, its town appended to the
								path{walked ? `: pass ${walked + 1} reads ${walked} more` : ''}.</span
							>
						</figcaption>
						<div class="io-box">
							<TokenPanel
								{tokens}
								V={s.spec.V}
								bind:hoverRoad={road2}
								onquery={(on) => (query2 = on)}
								{walked}
								edited={s.edited}
							/>
						</div>
					</figure>
					<figure>
						<figcaption class="cap">
							What it writes
							<span class="sub"
								>after the last {unit}, one score for every town name; nothing is drawn by the model</span
							>
						</figcaption>
						<div class="io-box">
							{#if out}
								<ReadoutPanel
									names={out.names}
									logits={out.logits}
									probs={out.probs}
									current={out.current}
									pending={s.phase === 'walk'}
								/>
							{/if}
						</div>
					</figure>
				</div>
			</figure>

			<div class="side">
				<figure class="pairs">
					<figcaption class="cap">
						What the pair state would answer
						<span class="sub"
							>from each town (rows) to each town (columns), after {unit} {layerNow}</span
						>
					</figcaption>
					<div class="table-box">
						<PairMap
							trace={s.trace}
							{order}
							layer={s.layer}
							goal={s.problem.goal}
							hoverRow={hoverTown}
							bind:hovered
						/>
					</div>
					<p class="facts">
						{#if hovered}
							<span
								>from <b>{hovered.u}</b>, it would drive to <b>{hovered.v}</b> with p = {fmt(
									hovered.p
								)}</span
							><span
								>{hoverRoute
									? `the route there, ${hoverRoute.length - 1} ${hoverRoute.length === 2 ? 'road' : 'roads'}, is lit on the map`
									: `no route leads from ${hovered.u} to ${hovered.v}`}</span
							>
						{:else}
							<span>deeper rose, more probable; hover a cell to see its route</span>
						{/if}
					</p>
				</figure>

				<figure class="layers">
					<figcaption class="cap">
						{loop ? 'Iteration by iteration' : 'Layer by layer'}
						<span class="sub"
							>its answer if it stopped after each {unit}; click {loop ? 'one' : 'a layer'} to stop there</span
						>
					</figcaption>
					<div class="chart-box">
						<LayerChart
							trace={s.trace}
							{branches}
							layer={s.layer}
							layers={s.depth}
							{unit}
							onseek={(l) => s.seek(l)}
						/>
					</div>
					{#if s.phase === 'paused'}
						<p class="facts">
							<span
								>Paused at {unit}
								{layerNow}.
								<button class="link" onclick={() => s.restart(s.question, s.edited)}
									>Run again</button
								></span
							>
						</p>
					{/if}
				</figure>
			</div>
		</div>
		<p class="how hint">
			Click a town to make it the goal; drag from one town to another to build a road; click a road
			to close it. The model searches the new map from its start.
		</p>

		<section class="results" aria-labelledby="recorded">
			<h2 id="recorded" class="label">Recorded results</h2>
			<div class="res">
				<div class="depth-panel">
					<h3>8 layers: which layer answers which depth</h3>
					<p class="sub">
						share of the test questions (12 per lookahead) the read-out gets right after each layer;
						this question outlined
					</p>
					<DepthChart lookahead={s.edited ? null : s.lookahead} layer={s.layer} height={196} />
					<p class="sub foot">
						After the start and the first three layers it answers only lookahead 1. Layer 5 answers
						most questions of lookahead 2 to 5, layer 6 most up to 7, layer 7 most up to 14, and
						layer 8 all of them: the depth it sees roughly doubles per layer, as the depth law says.
					</p>
				</div>
				<div class="depth-panel">
					<h3>One layer, repeated: how far it sees</h3>
					<p class="sub">
						the longest lookahead it answers after each number of iterations, 99% or more of the
						test questions at every lookahead up to it
					</p>
					<ReachChart size={s.size} K={loop ? s.K : null} />
				</div>
			</div>
		</section>

		<section class="more" aria-labelledby="how">
			<h2 id="how" class="label">How it works</h2>
			<div class="prose">
				<p>
					The map is only a drawing: no model sees it. Each reads the question as the benchmark
					writes it, a list of edges <span class="num">E u v</span>, then
					<span class="num">Q start goal</span>
					and the path so far, one token each. Tokens are pooled into the town names they write, each
					read with its two neighbours, and a state <i>z<sub>uv</sub></i> of 16 numbers is kept for
					every ordered pair of names. With <i>t</i> the state normalised over its channels, each of the
					8 layers is the summed pair layer
				</p>
				<!-- eslint-disable-next-line svelte/no-at-html-tags -- the paper's own equation, set by KaTeX at build time -->
				<div class="eq">{@html data.eq.pair}</div>
				<p>
					a route from u to k and a route from k to v make a route from u to v, so the length of
					route a pair can know about can double with each layer. The OR join replaces the sum, with
					no factor 1/√n, by
				</p>
				<!-- eslint-disable-next-line svelte/no-at-html-tags -- the paper's own equation, set by KaTeX at build time -->
				<div class="eq">{@html data.eq.or}</div>
				<p>
					one matrix product per channel. Every name <i>v</i> is scored by <i>w</i> · norm(<i
						>z<sub>cv</sub></i
					>) for the current town <i>c</i>; the table shows that read-out applied to every row, and
					the map tints each town by the current row. The 8 layers (39,504 parameters) answer all
					300 exported test questions; the OR join has 24,896. This page runs both forward passes in
					plain TypeScript, checked against the training code: to within 0.001 in every logit for
					the 8 layers, to within 0.0001 after every iteration from 1 to 10 for the OR join, on maps
					of 49 and 125 towns. Each question’s map was laid out once, offline, with its own land;
					every road on it is an edge of the question.
				</p>
				<p>
					The benchmark is Saparov et al., “Transformers Struggle to Learn to Search” (ICLR 2025):
					their generator, their test sets and their Transformer code. The <a
						href={resolve('/paper')}>paper</a
					> states the depth law and the size law, the training runs, and where each layer falls short.
				</p>
			</div>
		</section>
	</section>
</main>

<style>
	.demo {
		--col-gap: 44px;
		max-width: 1440px;
		margin: 0 auto;
		padding: 28px 32px 96px;
	}
	/* the second part opens after a rule of space, not a line */
	.part2 {
		margin-top: 96px;
	}
	.kicker {
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--color-rose);
	}
	.kicker + h2 {
		margin-top: 4px;
	}
	.lede i,
	.prose i {
		font-family: var(--font-serif);
		font-size: 1.06em;
	}
	.lede sup,
	.prose sub {
		font-size: 0.72em;
	}
	.eq {
		overflow-x: auto;
		overflow-y: hidden;
		color: var(--color-ink);
	}
	.head {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		align-items: end;
		gap: 12px 40px;
	}
	h2 {
		font-size: var(--text-head);
		font-weight: 500;
		letter-spacing: -0.02em;
		line-height: 1.1;
	}
	.lede {
		margin-top: 8px;
		max-width: 96ch;
		font-size: var(--text-ui);
		line-height: 1.55;
		color: var(--color-ink-soft);
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
	.pace {
		margin-left: auto;
	}
	.pace .plain {
		font-size: var(--text-xs);
	}
	.how {
		margin-top: 14px;
	}
	.models {
		margin-top: 18px;
	}
	.models + .controls {
		margin-top: 10px;
	}
	.lede + .lede {
		margin-top: 6px;
	}

	/* the stage: the map, and beside it the pair table over the layer chart */
	.stage {
		display: grid;
		grid-template-columns: minmax(0, 800px) minmax(340px, 1fr);
		gap: 28px var(--col-gap);
		margin-top: 20px;
		align-items: start;
	}
	.stage.wide {
		grid-template-columns: minmax(0, 900px) minmax(340px, 1fr);
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
		min-height: 120px;
		overflow: hidden;
		border: 1px solid var(--color-edge);
		border-radius: 6px;
	}
	.status {
		display: grid;
		gap: 2px;
		margin-top: 10px;
		min-height: 44px;
	}
	.status p,
	.facts {
		min-width: 0;
		line-height: 20px;
	}
	.verdict {
		font-size: var(--text-ui);
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
	.verdict .soft {
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.facts {
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.facts b {
		color: var(--color-ink);
		font-weight: 600;
	}
	.facts span + span::before {
		content: ' · ';
	}
	.facts .msg {
		color: var(--color-ink);
	}
	.facts .msg b {
		color: var(--color-rose);
	}
	/* what the model reads and writes, under the map */
	.io {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 24px 32px;
		margin-top: 22px;
	}
	.io-box {
		margin-top: 10px;
	}
	.tints i {
		font-style: normal;
		font-weight: 500;
	}
	.tints .rose {
		color: var(--color-rose);
	}
	.tints .cobalt {
		color: var(--color-cobalt);
	}
	.link {
		margin-left: 6px;
		padding: 0;
		border: 0;
		background: none;
		font: inherit;
		font-weight: 500;
		text-decoration: underline;
		text-underline-offset: 3px;
		cursor: pointer;
		color: var(--color-ink);
	}
	.side {
		display: grid;
		gap: 30px;
		min-width: 0;
	}
	.table-box {
		margin-top: 10px;
		width: 100%;
		max-width: 440px;
		aspect-ratio: 1.08;
	}
	.pairs .facts {
		margin-top: 6px;
	}
	.chart-box {
		margin-top: 8px;
		height: 200px;
		max-width: 460px;
	}

	.results,
	.more {
		display: grid;
		grid-template-columns: 160px minmax(0, 1fr);
		gap: 0 var(--col-gap);
		margin-top: 72px;
	}
	.label {
		padding-top: 2px;
		font-size: var(--text-xs);
		font-weight: 500;
		letter-spacing: 0;
		color: var(--color-ink-soft);
		text-align: right;
	}
	.res {
		display: grid;
		gap: 44px;
	}
	.depth-panel {
		max-width: 760px;
	}
	.depth-panel h3 {
		font-size: var(--text-sm);
		font-weight: 500;
	}
	.depth-panel .sub {
		margin-bottom: 12px;
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.depth-panel .foot {
		margin-top: 8px;
	}
	.prose {
		display: grid;
		gap: 12px;
		max-width: 76ch;
		color: var(--color-ink-soft);
		line-height: 1.6;
	}

	@media (max-width: 1099px) {
		.head {
			grid-template-columns: minmax(0, 1fr);
		}
		.model {
			text-align: left;
		}
		.stage,
		.stage.wide {
			grid-template-columns: minmax(0, 1fr);
		}
		.stage-map {
			max-width: 800px;
		}
		.stage.wide .stage-map {
			max-width: 900px;
		}
		.side {
			grid-template-columns: repeat(2, minmax(0, 1fr));
			align-items: start;
		}
	}
	@media (max-width: 719px) {
		.demo {
			padding: 18px 16px 72px;
			--col-gap: 20px;
		}
		.head {
			grid-template-columns: minmax(0, 1fr);
		}
		.model {
			text-align: left;
		}
		.controls {
			gap: 10px 20px;
		}
		.pace {
			margin-left: 0;
		}
		.side,
		.io {
			grid-template-columns: minmax(0, 1fr);
		}
		.results,
		.more {
			grid-template-columns: minmax(0, 1fr);
			gap: 8px;
			margin-top: 52px;
		}
		.label {
			text-align: left;
		}
	}
</style>
