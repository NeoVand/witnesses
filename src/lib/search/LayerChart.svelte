<script lang="ts">
	/**
	 * What the model would answer if it stopped after each layer: the read-out of the current vertex's row, one line per
	 * branch out of the current vertex. Rose is the branch the model finally takes, cobalt the strongest alternative;
	 * the others are quiet. Under the axis, the answer each layer would give. Lines are drawn up to the playhead. For the
	 * repeated layer the columns are its iterations (`unit`).
	 */
	import type { Trace } from './trace.ts';

	let {
		trace,
		branches,
		layer,
		layers = 8,
		unit = 'layer',
		height,
		onseek
	}: {
		trace: Trace | null;
		/** the current vertex's children */
		branches: number[];
		layer: number;
		layers?: number;
		/** what a column is: a layer of the stack, or an iteration of the repeated layer */
		unit?: 'layer' | 'iteration';
		height?: number;
		onseek?: (l: number) => void;
	} = $props();

	let width = $state(0);
	let measured = $state(0);
	const H = $derived(height ?? measured);
	const M = { l: 28, r: 56, t: 12, b: 50 };
	const iw = $derived(Math.max(10, width - M.l - M.r));
	const ih = $derived(Math.max(10, H - M.t - M.b));
	const x = (l: number) => M.l + (l / layers) * iw;
	const y = (p: number) => M.t + (1 - p) * ih;

	const ci = $derived(trace ? trace.names.indexOf(trace.current) : -1);
	const m = $derived(trace?.names.length ?? 0);
	/** p[l] for each branch */
	const series = $derived(
		trace && ci >= 0
			? branches
					.map((v) => ({ v, j: trace.names.indexOf(v) }))
					.filter((s) => s.j >= 0)
					.map((s) => ({ v: s.v, p: trace.maps.map((mp) => mp[ci * m + s.j]) }))
			: []
	);
	const answer = $derived(trace?.answer);
	/** the strongest alternative: the other branch with the highest probability at any layer */
	const rival = $derived(
		series.filter((s) => s.v !== answer).sort((a, b) => Math.max(...b.p) - Math.max(...a.p))[0]?.v
	);
	/** what each layer would answer: the arg max over every name of that layer's row */
	const says = $derived(
		trace && ci >= 0
			? trace.maps.map((mp) => {
					let b = 0;
					for (let k = 1; k < m; k++) if (mp[ci * m + k] > mp[ci * m + b]) b = k;
					return { v: trace.names[b], p: mp[ci * m + b] };
				})
			: []
	);
	const at = (p: number[], l: number) => {
		const lo = Math.max(0, Math.min(p.length - 1, Math.floor(l)));
		const hi = Math.min(p.length - 1, lo + 1);
		return p[lo] + (p[hi] - p[lo]) * (l - lo);
	};
	function line(p: number[]): string {
		const pts: string[] = [];
		const end = Math.min(layer, p.length - 1);
		for (let l = 0; l <= Math.floor(end); l++) pts.push(`${x(l).toFixed(1)},${y(p[l]).toFixed(1)}`);
		if (end > Math.floor(end)) pts.push(`${x(end).toFixed(1)},${y(at(p, end)).toFixed(1)}`);
		return pts.length > 1 ? `M${pts.join('L')}` : '';
	}
	const ordered = $derived([
		...series.filter((s) => s.v !== answer && s.v !== rival),
		...series.filter((s) => s.v === rival),
		...series.filter((s) => s.v === answer)
	]);
	const fmt = (p: number) => (p >= 0.995 ? '1.00' : p.toFixed(2));
	/** labels at the line ends, nudged apart so they never overlap */
	const ends = $derived.by(() => {
		const shown = series.filter((s) => s.v === answer || s.v === rival);
		const ys = shown.map((s) => ({
			v: s.v,
			p: at(s.p, Math.min(layer, layers)),
			y: y(at(s.p, Math.min(layer, layers)))
		}));
		ys.sort((a, b) => a.y - b.y);
		for (let i = 1; i < ys.length; i++) if (ys[i].y - ys[i - 1].y < 13) ys[i].y = ys[i - 1].y + 13;
		return ys;
	});
	const headX = $derived(x(Math.min(layer, layers)));
</script>

<div
	class="chart"
	bind:clientWidth={width}
	bind:clientHeight={measured}
	style:height={height ? `${height}px` : '100%'}
>
	{#if width > 0 && H > 0}
		<svg
			{width}
			height={H}
			role="img"
			aria-label="The answer read after each {unit}, for each branch"
		>
			{#each [0, 0.5, 1] as g (g)}
				<line class="grid" x1={M.l} x2={M.l + iw} y1={y(g)} y2={y(g)} />
				<text class="tick" x={M.l - 8} y={y(g) + 4} text-anchor="end">{g}</text>
			{/each}
			{#each Array.from({ length: layers + 1 }, (_, l) => l) as l (l)}
				<g
					class={['col', { on: Math.round(layer) === l }]}
					role="button"
					tabindex="0"
					aria-label="Show {unit} {l}"
					onclick={() => onseek?.(l)}
					onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && onseek?.(l)}
				>
					<rect
						class="hit"
						x={x(l) - iw / layers / 2}
						y={M.t}
						width={iw / layers}
						height={ih + M.b}
					/>
					<text class="tick" x={x(l)} y={M.t + ih + 15} text-anchor="middle">{l}</text>
					{#if says[l] && layer >= l - 0.02}
						<text
							class={['say', { right: says[l].v === answer }]}
							x={x(l)}
							y={M.t + ih + 33}
							text-anchor="middle">{says[l].v}</text
						>
					{/if}
				</g>
			{/each}
			<text class="axis" x={M.l + iw} y={M.t + ih + 50} text-anchor="end"
				>{unit}, and the answer it would give</text
			>
			<line class="head" x1={headX} x2={headX} y1={M.t} y2={M.t + ih} />
			{#each ordered as s (s.v)}
				<path class={['ln', { answer: s.v === answer, rival: s.v === rival }]} d={line(s.p)} />
				<circle
					class={['end', { answer: s.v === answer, rival: s.v === rival }]}
					cx={headX}
					cy={y(at(s.p, Math.min(layer, layers)))}
					r={s.v === answer || s.v === rival ? 3 : 1.8}
				/>
			{/each}
			{#each ends as e (e.v)}
				<text
					class={['lab', { answer: e.v === answer, rival: e.v !== answer }]}
					x={headX + 8}
					y={e.y + 4}>{e.v} <tspan class="pv">{fmt(e.p)}</tspan></text
				>
			{/each}
		</svg>
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
	.axis {
		font-size: var(--text-2xs);
		fill: var(--color-ink-soft);
		font-variant-numeric: tabular-nums;
	}
	.col {
		cursor: pointer;
		outline: none;
	}
	.col .hit {
		fill: transparent;
	}
	.col:hover .hit,
	.col:focus-visible .hit {
		fill: rgb(31 42 68 / 0.03);
	}
	.col.on .tick {
		fill: var(--color-ink);
		font-weight: 600;
	}
	.say {
		font-size: var(--text-2xs);
		font-variant-numeric: tabular-nums;
		fill: var(--color-ink-soft);
	}
	.say.right {
		fill: var(--color-rose);
		font-weight: 600;
	}
	.head {
		stroke: var(--color-ink);
		stroke-width: 1;
		stroke-dasharray: 1 3;
		opacity: 0.45;
	}
	.ln {
		fill: none;
		stroke: var(--color-mist);
		stroke-width: 1;
		stroke-linejoin: round;
		stroke-linecap: round;
	}
	.ln.rival {
		stroke: var(--color-cobalt);
		stroke-width: 1.5;
	}
	.ln.answer {
		stroke: var(--color-rose);
		stroke-width: 2;
	}
	.end {
		fill: var(--color-mist);
	}
	.end.rival {
		fill: var(--color-cobalt);
	}
	.end.answer {
		fill: var(--color-rose);
	}
	.lab {
		font-size: var(--text-2xs);
		font-weight: 600;
		font-variant-numeric: tabular-nums;
	}
	.lab.answer {
		fill: var(--color-rose);
	}
	.lab.rival {
		fill: var(--color-cobalt);
	}
	.pv {
		font-weight: 400;
	}
</style>
