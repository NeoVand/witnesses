<script lang="ts">
	/**
	 * The pair state, read: one cell for every ordered pair of towns, rows "from" and columns "to", shaded by what the
	 * state would answer from the row's town (the read-out applied to z[u][v], normalised over each row; deeper rose,
	 * more probable). Rows and columns follow the graph's order. The current town's row is the model's own answer, the
	 * goal's column is marked, and a town hovered on the map lights its row. The table follows the playhead, blending
	 * between layers as it moves. Hover a cell to light that pair's route on the map.
	 */
	import type { Trace } from './trace.ts';
	import { PALETTE, roseRamp } from './theme.ts';

	let {
		trace,
		order,
		layer,
		goal,
		labels = true,
		hoverRow = null,
		hovered = $bindable(null)
	}: {
		trace: Trace | null;
		/** town names in the order rows and columns are drawn */
		order: number[];
		layer: number;
		goal: number;
		labels?: boolean;
		/** a town hovered on the map: its row is lit */
		hoverRow?: number | null;
		hovered?: { u: number; v: number; p: number } | null;
	} = $props();

	let w = $state(0);
	let h = $state(0);
	let canvas: HTMLCanvasElement | undefined = $state();

	const LEFT = $derived(labels ? 46 : 0);
	const TOP = $derived(labels ? 20 : 0);
	const side = $derived(
		Math.max(0, Math.min(w - LEFT - (labels ? 4 : 0), h - TOP - (labels ? 2 : 0)))
	);
	/** the display order restricted to names the model saw, as model indices */
	const idx = $derived(trace ? order.map((v) => trace.names.indexOf(v)).filter((i) => i >= 0) : []);
	const shownNames = $derived(trace ? idx.map((i) => trace.names[i]) : []);
	const cell = $derived(idx.length ? side / idx.length : 0);
	const curRow = $derived(trace ? shownNames.indexOf(trace.current) : -1);
	const goalCol = $derived(shownNames.indexOf(goal));
	const litRow = $derived(hoverRow !== null ? shownNames.indexOf(hoverRow) : -1);
	const hRow = $derived(hovered ? shownNames.indexOf(hovered.u) : -1);
	const hCol = $derived(hovered ? shownNames.indexOf(hovered.v) : -1);
	/** two labels this close (on large maps the rows are a few pixels tall) would print over each other */
	const apart = (a: number, b: number) => b < 0 || Math.abs(a - b) * cell >= 13;

	$effect(() => {
		if (!canvas || !trace || !idx.length || side <= 0) return;
		const dpr = window.devicePixelRatio || 1;
		const px = Math.round(side * dpr);
		if (canvas.width !== px) {
			canvas.width = px;
			canvas.height = px;
		}
		const ctx = canvas.getContext('2d')!;
		const m = trace.names.length;
		const k = idx.length;
		const lo = Math.max(0, Math.min(trace.maps.length - 1, Math.floor(layer)));
		const hi = Math.min(trace.maps.length - 1, lo + 1);
		const f = layer - lo;
		const A = trace.maps[lo];
		const B = trace.maps[hi];
		const img = ctx.createImageData(px, px);
		const data = img.data;
		const colour = new Uint8Array(k * k * 3);
		for (let i = 0; i < k; i++)
			for (let j = 0; j < k; j++) {
				const q = idx[i] * m + idx[j];
				const [r, g, b] = roseRamp(A[q] + (B[q] - A[q]) * f);
				const c = (i * k + j) * 3;
				colour[c] = r;
				colour[c + 1] = g;
				colour[c + 2] = b;
			}
		// nearest-neighbour cells with a hairline of white between them when they are big enough to keep it
		const gap = cell * dpr > 5 ? 1 : 0;
		for (let y = 0; y < px; y++) {
			const i = Math.min(k - 1, Math.floor((y / px) * k));
			const yIn = y - Math.floor((i * px) / k);
			for (let x = 0; x < px; x++) {
				const j = Math.min(k - 1, Math.floor((x / px) * k));
				const xIn = x - Math.floor((j * px) / k);
				const o = (y * px + x) * 4;
				if (gap && (xIn === 0 || yIn === 0)) {
					data[o] = data[o + 1] = data[o + 2] = 255;
				} else {
					const c = (i * k + j) * 3;
					data[o] = colour[c];
					data[o + 1] = colour[c + 1];
					data[o + 2] = colour[c + 2];
				}
				data[o + 3] = 255;
			}
		}
		ctx.putImageData(img, 0, 0);
		const band = (row: number, style: string, lw: number) => {
			ctx.strokeStyle = style;
			ctx.lineWidth = lw * dpr;
			const y0 = (row * px) / k;
			ctx.strokeRect(lw * dpr * 0.5, y0 + lw * dpr * 0.5, px - lw * dpr, px / k - lw * dpr + gap);
		};
		ctx.strokeStyle = PALETTE.edge;
		ctx.lineWidth = dpr;
		ctx.strokeRect(0.5 * dpr, 0.5 * dpr, px - dpr, px - dpr);
		if (litRow >= 0 && litRow !== curRow) band(litRow, PALETTE.blueGrey, 1);
		if (curRow >= 0) band(curRow, PALETTE.ink, 1.5);
		if (hRow >= 0 && hCol >= 0) {
			ctx.strokeStyle = PALETTE.ink;
			ctx.lineWidth = 1.5 * dpr;
			const s = px / k;
			ctx.strokeRect(hCol * s, hRow * s, s + gap, s + gap);
		}
	});

	function move(e: PointerEvent) {
		if (!trace || !idx.length) return;
		const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const i = Math.floor(((e.clientY - r.top) / r.height) * idx.length);
		const j = Math.floor(((e.clientX - r.left) / r.width) * idx.length);
		if (i < 0 || j < 0 || i >= idx.length || j >= idx.length) return (hovered = null);
		const m = trace.names.length;
		const lo = Math.max(0, Math.min(trace.maps.length - 1, Math.floor(layer)));
		const hi = Math.min(trace.maps.length - 1, lo + 1);
		const q = idx[i] * m + idx[j];
		const p = trace.maps[lo][q] + (trace.maps[hi][q] - trace.maps[lo][q]) * (layer - lo);
		hovered = { u: shownNames[i], v: shownNames[j], p };
	}
</script>

<div class="table" bind:clientWidth={w} bind:clientHeight={h}>
	{#if trace && side > 0}
		<div
			class="field"
			style:left="{LEFT}px"
			style:top="{TOP}px"
			style:width="{side}px"
			style:height="{side}px"
		>
			<canvas
				bind:this={canvas}
				style:width="{side}px"
				style:height="{side}px"
				onpointermove={move}
				onpointerleave={() => (hovered = null)}
				aria-label="The pair state read as answers: {shownNames.length} by {shownNames.length} towns, rows from, columns to"
			></canvas>
			{#if labels}
				<span class="axis from">from</span>
				{#if goalCol < shownNames.length * 0.78 && hCol < shownNames.length * 0.78}
					<span class="axis to">to</span>
				{/if}
				{#if curRow >= 0}
					<span class="row-label" style:top="{(curRow + 0.5) * cell}px">now {trace.current}</span>
				{/if}
				{#if litRow >= 0 && litRow !== curRow && apart(litRow, curRow)}
					<span class="row-label soft" style:top="{(litRow + 0.5) * cell}px">{hoverRow}</span>
				{/if}
				{#if hRow >= 0 && hRow !== curRow && hRow !== litRow && apart(hRow, curRow) && apart(hRow, litRow)}
					<span class="row-label soft" style:top="{(hRow + 0.5) * cell}px">{hovered?.u}</span>
				{/if}
				{#if goalCol >= 0}
					<span class="col-label goal" style:left="{(goalCol + 0.5) * cell}px">goal {goal}</span>
				{/if}
				{#if hCol >= 0 && hCol !== goalCol && (goalCol < 0 || Math.abs(hCol - goalCol) * cell >= 40)}
					<span class="col-label" style:left="{(hCol + 0.5) * cell}px">{hovered?.v}</span>
				{/if}
			{/if}
		</div>
	{/if}
</div>

<style>
	.table {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 0;
	}
	.field {
		position: absolute;
	}
	canvas {
		display: block;
		image-rendering: pixelated;
		cursor: crosshair;
	}
	.axis {
		position: absolute;
		font-size: var(--text-2xs);
		color: var(--color-ink-soft);
		white-space: nowrap;
	}
	.axis.from {
		right: calc(100% + 8px);
		top: -18px;
	}
	.axis.to {
		right: 0;
		bottom: calc(100% + 4px);
	}
	.row-label {
		position: absolute;
		right: calc(100% + 6px);
		transform: translateY(-50%);
		font-size: var(--text-2xs);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		color: var(--color-ink);
		transition: top var(--move) var(--ease);
	}
	.row-label.soft {
		font-weight: 400;
		color: var(--color-ink-soft);
		transition: none;
	}
	.col-label {
		position: absolute;
		bottom: calc(100% + 4px);
		transform: translateX(-50%);
		font-size: var(--text-2xs);
		font-variant-numeric: tabular-nums;
		white-space: nowrap;
		color: var(--color-ink-soft);
	}
	.col-label.goal {
		color: var(--color-rose);
		font-weight: 500;
	}
</style>
