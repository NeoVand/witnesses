<script lang="ts">
	/**
	 * The current town's roads and the walks that have reached each: one row per road out of the current town, its bar
	 * the light of the town it leads to at the playhead (the map's own scale: walk weight 10^LO dark, the solve's largest
	 * full), its number the walk weight itself. Once the road is chosen, each row gives the read-out's probability; the
	 * chosen road is rose.
	 */
	import { lightAt, LO, sci, type WalkFrames } from './walks.ts';

	let {
		frames,
		roads,
		M,
		solved,
		chosen,
		hover = null
	}: {
		frames: WalkFrames;
		/** the current town's roads: far town, the solve's walk weight, the read-out's probability (null: recorded) */
		roads: { v: number; x: number | null; p: number | null }[];
		M: number;
		solved: number;
		/** the chosen road is lit (0 to 1) */
		chosen: number;
		hover?: number | null;
	} = $props();

	const light = $derived(lightAt(frames, M, solved));
	/** the walk weight at the playhead, from the same frames the map draws */
	function weight(v: number): number | null {
		const s = frames.sums[v];
		if (!s) return null;
		const lo = Math.max(0, Math.min(frames.reach, Math.floor(M)));
		const hi = Math.min(frames.reach, lo + 1);
		const t = Math.max(0, Math.min(1, M - lo));
		let lx = s[lo] + (s[hi] - s[lo]) * t;
		if (solved > 0) lx += (frames.solve[v] - lx) * solved;
		return 10 ** lx;
	}
	const rows = $derived(
		[...roads]
			.sort((a, b) => (b.x ?? frames.solve[b.v] ?? -99) - (a.x ?? frames.solve[a.v] ?? -99))
			.slice(0, 10)
			.map((r) => ({ ...r, w: weight(r.v), b: light.get(r.v) ?? 0 }))
	);
	const more = $derived(Math.max(0, roads.length - 10));
	const fmt = (x: number | null) => (x === null || x < 10 ** LO ? '< 0.0003' : sci(x));
	const pfmt = (p: number) => (p >= 0.995 ? '1.00' : p < 0.005 ? '0.00' : p.toFixed(2));
</script>

<div
	class="roads"
	role="table"
	aria-label="The current town's roads: the walk weight that has reached the town each leads to"
>
	<div class="row head" role="row">
		<span role="columnheader">road</span>
		<span role="columnheader">walks that reached its town</span>
		<span class="r" role="columnheader">{chosen > 0 ? 'read-out' : 'weight'}</span>
	</div>
	{#each rows as r (r.v)}
		{@const ans = chosen > 0 && r.v === frames.answer}
		<div class={['row', { ans, hov: hover === r.v }]} role="row">
			<span class="to num" role="cell">{frames.current} → <b>{r.v}</b></span>
			<span class="track" role="cell">
				<span class="bar" style:width="{(r.b * 100).toFixed(1)}%"></span>
			</span>
			<span class="r num" role="cell"
				>{chosen > 0 && r.p !== null ? `p ${pfmt(r.p)}` : fmt(r.w)}</span
			>
		</div>
	{/each}
	{#if more}<p class="more">and {more} more roads, none of them lit</p>{/if}
</div>

<style>
	.roads {
		display: grid;
		gap: 5px;
		font-size: var(--text-xs);
	}
	.row {
		display: grid;
		grid-template-columns: 74px minmax(0, 1fr) 64px;
		align-items: center;
		gap: 12px;
		min-height: 18px;
		color: var(--color-ink-soft);
	}
	.head {
		min-height: 0;
		margin-bottom: 2px;
		color: var(--color-ink-soft);
	}
	.to b {
		font-weight: 500;
		color: var(--color-ink);
	}
	.ans .to b,
	.ans .r {
		color: var(--color-rose);
		font-weight: 600;
	}
	.hov .to b {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.track {
		position: relative;
		height: 6px;
		border-radius: 3px;
		background: var(--track);
		overflow: hidden;
	}
	.bar {
		position: absolute;
		inset: 0 auto 0 0;
		border-radius: 3px;
		background: var(--color-rose-light);
	}
	.ans .bar {
		background: var(--color-rose);
	}
	.r {
		text-align: right;
		white-space: nowrap;
	}
	.more {
		margin-top: 2px;
		color: var(--color-ink-soft);
	}
</style>
