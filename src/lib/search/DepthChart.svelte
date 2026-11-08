<script lang="ts">
	/**
	 * Which layer answers which depth: for every lookahead (columns) and every layer (rows, the last on top), the share
	 * of the test questions the read-out already answers right after that layer. The question on screen is outlined, and
	 * the playhead's layer is marked.
	 */
	import depth from './depth.json';
	import { roseRamp, rgb } from './theme.ts';

	let {
		lookahead,
		layer,
		height = 210
	}: { lookahead: number | null; layer: number; height?: number } = $props();

	let width = $state(0);
	const M = { l: 26, r: 8, t: 8, b: 30 };
	const L = depth.layers;
	const cols = depth.lookaheads;
	const iw = $derived(Math.max(10, width - M.l - M.r));
	const ih = $derived(height - M.t - M.b);
	const cw = $derived(iw / cols.length);
	const rh = $derived(ih / (L + 1));
	const share = (c: number, l: number) => depth.right[c][l] / depth.questions[c];
	const col = $derived(lookahead === null ? -1 : cols.indexOf(lookahead));
	const TICKS = [1, 4, 8, 12, 16, 20, 24];
</script>

<div class="depth" bind:clientWidth={width} style:height="{height}px">
	{#if width > 0}
		<svg
			{width}
			{height}
			role="img"
			aria-label="Share of test questions answered right after each layer, by lookahead"
		>
			{#each cols as c, ci (c)}
				{#each Array.from({ length: L + 1 }, (_, l) => l) as l (l)}
					<rect
						x={M.l + ci * cw + 0.5}
						y={M.t + (L - l) * rh + 0.5}
						width={Math.max(0, cw - 1)}
						height={Math.max(0, rh - 1)}
						fill={rgb(roseRamp(share(ci, l)))}
					>
						<title
							>lookahead {c}, layer {l}: {depth.right[ci][l]} of {depth.questions[ci]} right</title
						>
					</rect>
				{/each}
			{/each}
			{#each Array.from({ length: L + 1 }, (_, l) => l) as l (l)}
				<text
					class={['tick', { on: Math.round(layer) === l }]}
					x={M.l - 7}
					y={M.t + (L - l + 0.5) * rh + 3.5}
					text-anchor="end">{l}</text
				>
			{/each}
			<!-- the playhead's layer -->
			<rect
				class="row"
				x={M.l - 1}
				y={M.t + (L - Math.min(layer, L)) * rh}
				width={iw + 2}
				height={rh}
			/>
			{#if col >= 0}
				<rect class="this" x={M.l + col * cw} y={M.t} width={cw} height={ih} />
			{/if}
			{#each TICKS as t (t)}
				{@const ci = cols.indexOf(t)}
				{#if ci >= 0}
					<text class="tick" x={M.l + (ci + 0.5) * cw} y={M.t + ih + 13} text-anchor="middle"
						>{t}</text
					>
				{/if}
			{/each}
			<text class="axis" x={M.l + iw} y={M.t + ih + 27} text-anchor="end"
				>lookahead of the question</text
			>
			<text class="axis" x={M.l - 7} y={M.t + ih + 27} text-anchor="end">layer</text>
		</svg>
	{/if}
</div>

<style>
	.depth {
		width: 100%;
	}
	svg {
		display: block;
		overflow: visible;
	}
	.tick,
	.axis {
		font-size: var(--text-2xs);
		fill: var(--color-ink-soft);
		font-variant-numeric: tabular-nums;
	}
	.tick.on {
		fill: var(--color-ink);
		font-weight: 600;
	}
	.row {
		fill: none;
		stroke: var(--color-ink);
		stroke-width: 1;
		stroke-dasharray: 2 3;
		opacity: 0.55;
	}
	.this {
		fill: none;
		stroke: var(--color-ink);
		stroke-width: 1.5;
	}
</style>
