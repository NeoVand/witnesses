<script lang="ts">
	/**
	 * How far the repeated layer sees after K iterations: the longest lookahead its read-out answers (at least 99% of the
	 * benchmark's test questions at every lookahead up to it, as the training run scored this checkpoint), on 49- and
	 * 125-town maps, against the most a pair can know after K joins, a route of 2^K roads. Log scale in lookahead. The
	 * page's own K is marked; hovering a column reads its numbers.
	 */
	import data from './reach.json';

	let {
		size,
		K,
		height = 190
	}: {
		/** the map size on the stage: its line is drawn on top */
		size: number;
		/** the stage's iterations */
		K: number | null;
		height?: number;
	} = $props();

	const Ks: number[] = data.K;
	const SERIES = [
		{ V: 49, reach: data.reach['49'], all: data.longest['49'], cls: 'cobalt' },
		{ V: 125, reach: data.reach['125'], all: data.longest['125'], cls: 'rose' }
	];
	const TOP = 64;

	let width = $state(0);
	let hover = $state<number | null>(null);
	const M = { l: 30, r: 70, t: 26, b: 30 };
	const iw = $derived(Math.max(10, width - M.l - M.r));
	const ih = $derived(Math.max(10, height - M.t - M.b));
	const x = (k: number) => M.l + ((k - 1) / (Ks.length - 1)) * iw;
	const y = (L: number) =>
		M.t + (1 - Math.log2(Math.max(1, Math.min(TOP, L))) / Math.log2(TOP)) * ih;
	const TICKS = [1, 2, 4, 8, 16, 32, 64];
	const path = (pts: [number, number][]) =>
		pts.map(([k, L], i) => `${i ? 'L' : 'M'}${x(k).toFixed(1)},${y(L).toFixed(1)}`).join('');
	const bound = $derived(path(Ks.filter((k) => 2 ** k <= TOP).map((k) => [k, 2 ** k])));
	/** the stage's size drawn last, on top */
	const order = $derived(
		[...SERIES].sort((a, b) => (a.V === size ? 1 : 0) - (b.V === size ? 1 : 0))
	);
	const shown = $derived(hover ?? K);
	const read = $derived(
		shown === null
			? null
			: {
					k: shown,
					bound: 2 ** shown,
					by: SERIES.map((s) => ({ V: s.V, L: s.reach[shown - 1], all: s.all }))
				}
	);
</script>

<div class="chart" bind:clientWidth={width} style:height="{height}px">
	{#if width > 0}
		<svg
			{width}
			{height}
			role="img"
			aria-label="The longest lookahead the repeated layer answers after each number of iterations, on 49- and 125-town maps, against 2 to the K"
		>
			<!-- legend -->
			<g class="legend" transform="translate({M.l} 10)">
				<circle class="dot rose" cx="4" cy="0" r="4" />
				<text x="12" y="4">125 towns</text>
				<circle class="dot cobalt" cx="86" cy="0" r="4" />
				<text x="94" y="4">49 towns</text>
				<line class="bound" x1="160" x2="178" y1="0" y2="0" />
				<text x="184" y="4">at most 2<tspan class="sup" dy="-4">K</tspan></text>
			</g>
			{#each TICKS as t (t)}
				<line class="grid" x1={M.l} x2={M.l + iw} y1={y(t)} y2={y(t)} />
				<text class="tick" x={M.l - 7} y={y(t) + 3.5} text-anchor="end">{t}</text>
			{/each}
			{#each Ks as k (k)}
				<text class={['tick', { on: k === shown }]} x={x(k)} y={M.t + ih + 15} text-anchor="middle"
					>{k}</text
				>
			{/each}
			<text class="axis" x={M.l + iw} y={M.t + ih + 29} text-anchor="end"
				>iterations K; lookahead on a log scale</text
			>
			{#if shown !== null}
				<line class="head" x1={x(shown)} x2={x(shown)} y1={M.t} y2={M.t + ih} />
			{/if}
			<path class="bound" d={bound} />
			{#each order as s (s.V)}
				<path
					class={['ln', s.cls, { quiet: s.V !== size }]}
					d={path(Ks.map((k, i) => [k, s.reach[i]]))}
				/>
				{#each Ks as k, i (k)}
					<circle
						class={['pt', s.cls, { quiet: s.V !== size }]}
						cx={x(k)}
						cy={y(s.reach[i])}
						r={s.V === size ? 4 : 3}
					/>
				{/each}
				<text class="end" x={M.l + iw + 9} y={y(s.reach[Ks.length - 1]) + 4}
					>{s.reach[Ks.length - 1] === s.all ? `all ${s.all}` : s.reach[Ks.length - 1]}</text
				>
			{/each}
			<!-- hover: a column per K -->
			{#each Ks as k (k)}
				<rect
					class="hit"
					x={x(k) - iw / (Ks.length - 1) / 2}
					y={M.t}
					width={iw / (Ks.length - 1)}
					height={ih + 18}
					role="presentation"
					onpointerenter={() => (hover = k)}
					onpointerleave={() => (hover = null)}
				/>
			{/each}
		</svg>
		{#if read}
			<p class="read num" aria-live="polite">
				K = {read.k}: {#each read.by as b, i (b.V)}{i ? ', ' : ''}{b.L === b.all
						? `all ${b.L}`
						: `lookahead ${b.L}`} at {b.V} towns{/each}; 2<sup>K</sup> = {read.bound}
			</p>
		{/if}
	{/if}
</div>

<style>
	.chart {
		width: 100%;
		user-select: none;
	}
	svg {
		display: block;
		overflow: visible;
	}
	.grid {
		stroke: var(--color-edge);
		stroke-width: 1;
	}
	.tick,
	.axis,
	.legend text,
	.end {
		font-size: var(--text-2xs);
		fill: var(--color-ink-soft);
		font-variant-numeric: tabular-nums;
	}
	.tick.on {
		fill: var(--color-ink);
		font-weight: 600;
	}
	.sup {
		font-size: 9px;
	}
	.head {
		stroke: var(--color-ink);
		stroke-width: 1;
		stroke-dasharray: 1 3;
		opacity: 0.45;
	}
	.bound {
		fill: none;
		stroke: var(--color-ink-soft);
		stroke-width: 1;
		stroke-dasharray: 3 3;
	}
	.ln {
		fill: none;
		stroke-width: 2;
		stroke-linejoin: round;
		stroke-linecap: round;
		transition: opacity 700ms var(--ease);
	}
	.ln.rose {
		stroke: var(--color-rose);
	}
	.ln.cobalt {
		stroke: var(--color-cobalt);
	}
	.pt.rose,
	.dot.rose {
		fill: var(--color-rose);
	}
	.pt.cobalt,
	.dot.cobalt {
		fill: var(--color-cobalt);
	}
	.pt {
		stroke: var(--color-paper);
		stroke-width: 2;
		transition: opacity 700ms var(--ease);
	}
	.quiet {
		opacity: 0.45;
	}
	.hit {
		fill: transparent;
	}
	.hit:hover {
		fill: rgb(31 42 68 / 0.03);
	}
	.read sup {
		font-size: 9px;
	}
	.read {
		margin-top: 4px;
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
		line-height: 18px;
		min-height: 18px;
	}
</style>
