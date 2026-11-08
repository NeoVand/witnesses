<script lang="ts">
	/**
	 * The learned gates of the walk channel on this question (the paper's Fig. 2c, live): one dot per road for the step
	 * that walks it back, one for the step that walks it forward, a sample of town pairs with no road, and the
	 * question's own pair of start and goal. A hovered road's two dots are ringed and their values given.
	 */
	import { sci } from './walks.ts';
	import type { Gates } from './walkrun.ts';

	let { gates, hover = null }: { gates: Gates; hover?: { u: number; v: number } | null } = $props();

	let width = $state(0);
	const LABEL = 122;
	const ROW = 22;
	/** a fixed jitter per dot, so the rows read as clouds */
	const jit = (k: number) => (((Math.sin(k * 12.9898) * 43758.5453) % 1) + 1) % 1;
	const rows = $derived([
		{ key: 'back', label: 'road, walked back', vals: gates.roads.map((r) => r.back), cls: 'rose' },
		{
			key: 'along',
			label: 'road, walked forward',
			vals: gates.roads.map((r) => r.along),
			cls: 'cobalt'
		},
		{ key: 'none', label: 'no road', vals: gates.none, cls: 'grey' },
		{
			key: 'pair',
			label: 'start → goal',
			vals: gates.pair === null ? [] : [gates.pair],
			cls: 'open'
		}
	]);
	const median = (xs: number[]) => {
		const s = [...xs].sort((a, b) => a - b);
		return s.length ? s[Math.floor(s.length / 2)] : 0;
	};
	const stat = $derived({
		back: median(rows[0].vals),
		along: Math.max(0, ...rows[1].vals),
		none: Math.max(0, ...rows[2].vals)
	});
	const sx = (x: number) =>
		LABEL + 8 + Math.max(0, Math.min(1, x)) * Math.max(10, width - LABEL - 16);
	const hv = $derived(
		hover ? gates.roads.find((r) => r.u === hover.u && r.v === hover.v) : undefined
	);
	const H = $derived(rows.length * ROW + 22);
</script>

<div class="strip" bind:clientWidth={width}>
	{#if width > 0}
		<svg
			{width}
			height={H}
			role="img"
			aria-label="The learned gates of the walk channel on this question"
		>
			{#each rows as r, i (r.key)}
				{@const y = 10 + i * ROW}
				<text class="lab" x={LABEL} y={y + 4} text-anchor="end">{r.label}</text>
				<line class="base" x1={sx(0)} x2={sx(1)} y1={y} y2={y} />
				{#each r.vals as v, k (k)}
					<circle
						class={['dot', r.cls]}
						cx={sx(v)}
						cy={y + (r.key === 'pair' ? 0 : (jit(k + i * 977) - 0.5) * 9)}
						r={r.key === 'pair' ? 3.6 : 2}
					/>
				{/each}
			{/each}
			{#if hv}
				<circle class="ring" cx={sx(hv.back)} cy={10} r="5.5" />
				<circle class="ring" cx={sx(hv.along)} cy={10 + ROW} r="5.5" />
			{/if}
			{#each [0, 0.25, 0.5, 0.75, 1] as t (t)}
				<line class="tick" x1={sx(t)} x2={sx(t)} y1={H - 16} y2={H - 12} />
				<text class="tl" x={sx(t)} y={H - 1} text-anchor="middle">{t}</text>
			{/each}
		</svg>
	{:else}
		<!-- the chart's own height before it is measured, so nothing under it moves -->
		<div style:height="{H}px"></div>
	{/if}
	<p class="read">
		{#if hv}
			<span>road <b class="num">{hv.u} → {hv.v}</b></span><span
				>walked back <b class="num rose">{sci(hv.back)}</b></span
			><span>walked forward <b class="num">{sci(hv.along)}</b></span>
		{:else}
			<span>median gate walking a road back <b class="num rose">{stat.back.toFixed(2)}</b></span
			><span>largest forward <b class="num">{sci(stat.along)}</b></span><span
				>largest with no road <b class="num">{sci(stat.none)}</b></span
			>
		{/if}
	</p>
</div>

<style>
	.strip {
		width: 100%;
		min-width: 0;
	}
	svg {
		display: block;
		overflow: visible;
	}
	.lab,
	.tl {
		font-size: 11px;
		fill: var(--color-ink-soft);
	}
	.base {
		stroke: var(--color-edge);
		stroke-width: 1;
	}
	.tick {
		stroke: var(--color-mist);
	}
	.dot {
		stroke: none;
	}
	.dot.rose {
		fill: var(--color-rose);
		opacity: 0.7;
	}
	.dot.cobalt {
		fill: var(--color-cobalt);
		opacity: 0.7;
	}
	.dot.grey {
		fill: var(--color-blue-grey);
		opacity: 0.45;
	}
	.dot.open {
		fill: var(--color-paper);
		stroke: var(--color-ink);
		stroke-width: 1.4;
	}
	.ring {
		fill: none;
		stroke: var(--color-ink);
		stroke-width: 1.4;
	}
	.read {
		margin-top: 6px;
		font-size: var(--text-xs);
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.read span + span::before {
		content: ' · ';
	}
	.read b {
		font-weight: 600;
		color: var(--color-ink);
	}
	.read .rose {
		color: var(--color-rose);
	}
</style>
