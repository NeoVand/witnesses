<script lang="ts">
	/**
	 * The question as a road map (its layout from static/maps/), the stage of the demo. Grey roads on white land, pale
	 * water, woods; the current town is the dark disc, the goal the rose ring, every town's number beside it. While the
	 * layers sweep, each town is tinted by the read-out of the current town's row at the playhead (rose above the row's
	 * mean, cobalt below); the answer's road lights rose, and the walk drives the marker along it, one road a step.
	 * Hovering a town names it to the pair table; a pair hovered in the table lights its route here. Given `reach`, the
	 * roads within that many roads of the current town are drawn darker: how far a pair can see at the playhead.
	 *
	 * Edits: click a town to make it the goal, drag from one town to another to add a road (a new road, bridged where
	 * it crosses water), click a road to remove it.
	 *
	 * The propagator's walks (`light`, walks.ts): each road u → v glows rose by the light of v, the weight of the walks
	 * from the goal that have come back that far; with `front`, the walk length so far, a road lights from its far end
	 * as the walks cross it. `chosen` draws the answer's road on, bold, with its arrow. `trees` adds the woods' conifers.
	 */
	import { untrack } from 'svelte';
	import {
		at,
		centred,
		chevronD,
		length,
		lineD,
		newRoad,
		placeLabels,
		roadKey,
		route,
		shapeD,
		type MapLayout,
		type Pt
	} from './map.ts';
	import { current, type Problem } from './tokens.ts';
	import type { Trace } from './trace.ts';
	import { PALETTE, diverging, ease } from './theme.ts';
	import { toGoal, treeSpots } from './walks.ts';

	let {
		layout,
		question,
		problem,
		trace = null,
		layer = 0,
		answered = false,
		moveMs = 800,
		pair = null,
		routeLit = null,
		reach = null,
		hoverTown = $bindable(null),
		light = null,
		front = null,
		chosen = null,
		trees = false,
		numbers = 'all',
		hoverRoad = $bindable(null),
		onroadhover,
		marked = null,
		interactive = true,
		fixed,
		scale,
		label,
		onvertex,
		onedge,
		onadd
	}: {
		layout: MapLayout | null;
		/** the question as asked (the walk starts from its path) */
		question: Problem;
		/** the question after the steps walked */
		problem: Problem;
		trace?: Trace | null;
		layer?: number;
		/** the playhead has reached the last layer: the answer is lit */
		answered?: boolean;
		/** how long the marker takes to drive a road */
		moveMs?: number;
		/** a pair hovered in the table: its route is lit */
		pair?: { u: number; v: number } | null;
		/** a route to light in rose (towns, in order) */
		routeLit?: number[] | null;
		/** draw the roads that end within this many roads of the current town darker */
		reach?: number | null;
		hoverTown?: number | null;
		/** the propagator's walks: town → brightness in [0, 1] */
		light?: Map<number, number> | null;
		/** the walk length so far (a float): roads light from the goal's side as the walks cross them */
		front?: number | null;
		/** the answer's road, drawn on to `t` (0 to 1) */
		chosen?: { u: number; v: number; t: number } | null;
		/** draw small conifers in the woods */
		trees?: boolean;
		/** every town's number, or only the start's, the goal's and the answer's */
		numbers?: 'all' | 'key';
		/** the road under the pointer (when `onroadhover` is given) */
		hoverRoad?: { u: number; v: number } | null;
		onroadhover?: (r: { u: number; v: number } | null) => void;
		/** towns to ring, e.g. the start and the goal while their query tokens are hovered */
		marked?: readonly number[] | null;
		interactive?: boolean;
		/** draw for this width and let the drawing scale with its box (a static map, rendered on the server too) */
		fixed?: number;
		/** the marks' size against the layout's own drawing size (default: from the width, 0.72 to 1) */
		scale?: number;
		label?: string;
		onvertex?: (v: number) => void;
		onedge?: (u: number, v: number) => void;
		onadd?: (u: number, v: number) => void;
	} = $props();

	let measured = $state(0);
	const w = $derived(fixed ?? measured);
	/**
	 * The part of the sheet drawn: all of it, except on maps of many towns, whose sheets keep a wide margin of land or
	 * sea; there the view closes in on the towns
	 */
	const view = $derived.by((): [number, number, number, number] => {
		if (!layout) return [0, 0, 100, 80];
		const [SW, SH] = layout.size;
		if (Object.keys(layout.towns).length <= 150) return [0, 0, SW, SH];
		let y0 = Infinity;
		let y1 = -Infinity;
		for (const [, y] of Object.values(layout.towns)) {
			y0 = Math.min(y0, y);
			y1 = Math.max(y1, y);
		}
		const top = Math.max(0, y0 - 10);
		const bottom = Math.min(SH, y1 + 10);
		return [0, top, SW, bottom - top];
	});
	const W = $derived(view[2]);
	const H = $derived(view[3]);
	/** px per mm, and the marks' scale: full size from the layout's own drawing width up, smaller on a phone */
	const k = $derived(w / W);
	/** on maps of many towns (250) the marks may shrink further on a phone, so the towns keep room between them */
	const many = $derived(layout ? Object.keys(layout.towns).length > 150 : false);
	const f = $derived(
		scale ??
			(many
				? Math.max(0.45, Math.min(1, w / 1200))
				: Math.max(0.72, Math.min(1, w / (layout?.demo.px ?? 700))))
	);
	const mm = (px: number) => px / k;
	/** the key towns' numbers stay readable however small the marks: at least 10 px */
	const fl = $derived(Math.max(f, 10 / (layout?.demo.font ?? 11)));

	const cur = $derived(current(problem));
	const T = (t: number): Pt => layout?.towns[String(t)] ?? [0, 0];

	// ---- the roads that stand: the layout's, less those removed, plus those added ----------------------------
	const present = $derived(new Set(problem.edges.map(([u, v]) => roadKey(u, v))));
	const laid = $derived(
		new Map((layout?.roads ?? []).map((r) => [roadKey(r.edge[0], r.edge[1]), r.path] as const))
	);
	const roads = $derived.by(() => {
		const out: { u: number; v: number; key: string; path: Pt[]; added: boolean; wet: Pt[][] }[] =
			[];
		if (!layout) return out;
		for (const [u, v] of problem.edges) {
			const key = roadKey(u, v);
			const p = laid.get(key);
			if (p) out.push({ u, v, key, path: p, added: false, wet: [] });
			else if (layout.towns[String(u)] && layout.towns[String(v)]) {
				const r = newRoad(layout, T(u), T(v));
				out.push({ u, v, key, path: r.path, added: true, wet: r.wet });
			}
		}
		return out;
	});
	const byKey = $derived(new Map(roads.map((r) => [r.key, r] as const)));

	// ---- what is lit ---------------------------------------------------------------------------------------------
	const asked = $derived(
		new Set(question.path.slice(0, -1).map((v, i) => roadKey(v, question.path[i + 1])))
	);
	const walked = $derived(
		new Set([
			...problem.path
				.slice(question.path.length - 1, -1)
				.map((v, i) => roadKey(v, problem.path[question.path.length - 1 + i + 1])),
			...(routeLit ?? []).slice(0, -1).map((v, i) => roadKey(v, routeLit![i + 1]))
		])
	);
	const visited = $derived(new Set(problem.path.slice(0, -1)));
	const live = $derived(trace !== null && trace.current === cur);
	const answerRoad = $derived(live && answered && trace ? roadKey(cur, trace.answer) : null);
	const scores = $derived(
		live && trace ? centred(trace.maps, trace.names, cur, layer) : new Map<number, number>()
	);
	const hoverRoute = $derived.by(() => {
		if (!pair) return new Set<string>();
		const r = route(problem.edges, pair.u, pair.v);
		return new Set(r ? r.slice(0, -1).map((x, i) => roadKey(x, r[i + 1])) : []);
	});
	const hoverEnds = $derived(new Set([...(pair ? [pair.u, pair.v] : []), ...(marked ?? [])]));

	// ---- the walks' light ------------------------------------------------------------------------------------------
	/** roads from each town to the goal: the walk length at which the walks from the goal reach it */
	const dist = $derived(light ? toGoal(problem.edges, problem.goal) : new Map<number, number>());
	const glow = $derived.by(() => {
		const out: { key: string; d: string; b: number; r: number }[] = [];
		if (!light) return out;
		for (const r of roads) {
			const b = light.get(r.v) ?? 0;
			if (b < 0.01) continue;
			const dv = dist.get(r.v);
			const rr = front === null || dv === undefined ? 1 : Math.max(0, Math.min(1, front - dv));
			if (rr <= 0) continue;
			out.push({ key: r.key, d: lineD(r.path), b, r: Math.round(rr * 1000) / 1000 });
		}
		return out;
	});
	const spots = $derived(trees && layout && w ? treeSpots(layout, w / layout.size[0]) : []);
	const chosenRoad = $derived(chosen ? byKey.get(roadKey(chosen.u, chosen.v)) : undefined);
	/** the chosen road, stopped short of the answer's disc so its arrowhead sits on the road */
	const chosenPath = $derived.by(() => {
		if (!chosenRoad || !layout) return null;
		const p = chosenRoad.path;
		const total = length(p);
		const cut = Math.max(0, total - mm((layout.demo.town + 3.5) * f));
		const pts: Pt[] = [];
		const n = Math.max(8, Math.round(total / 0.4));
		for (let i = 0; i <= n; i++) {
			const q = at(p, (cut * i) / n);
			pts.push([q.x, q.y]);
		}
		const end = at(p, cut);
		return { d: lineD(pts), end, cut };
	});
	const mix = (b: number) => {
		const t = 0.4 + 0.6 * b;
		const a = [0x1f, 0x2a, 0x44];
		const c = [0xd6, 0x33, 0x6c];
		return `rgb(${a.map((x, i) => Math.round(x + (c[i] - x) * t)).join(' ')})`;
	};
	/** roads one can drive from the current town within `reach` roads */
	const seen = $derived.by(() => {
		const keys: string[] = [];
		if (reach !== null && reach >= 1) {
			const kids: Record<number, number[]> = {};
			for (const [u, v] of problem.edges) (kids[u] ??= []).push(v);
			const done: Record<number, true> = { [cur]: true };
			let frontier = [cur];
			for (let h = 0; h < reach && frontier.length; h++) {
				const next: number[] = [];
				for (const u of frontier)
					for (const v of kids[u] ?? []) {
						keys.push(roadKey(u, v));
						if (!done[v]) {
							done[v] = true;
							next.push(v);
						}
					}
				frontier = next;
			}
		}
		return new Set(keys);
	});

	// ---- the marker: it drives the road from the town it left to the one it reached --------------------------
	/** the marker's place while it drives; otherwise it sits on the current town */
	let anim = $state<Pt | null>(null);
	const marker = $derived(anim ?? T(cur));
	let raf = 0;
	let lastCur: number | null = null;
	let lastQuestion: Problem | null = null;
	let lastLayout: MapLayout | null = null;
	$effect(() => {
		const c = cur;
		const q = question;
		const L = layout;
		untrack(() => {
			cancelAnimationFrame(raf);
			anim = null;
			if (!L) return;
			const road = lastCur !== null ? byKey.get(roadKey(lastCur, c)) : undefined;
			const glide = L === lastLayout && q === lastQuestion && road && moveMs > 0;
			lastCur = c;
			lastQuestion = q;
			lastLayout = L;
			if (!glide) return;
			const path = road.path;
			const total = length(path);
			const t0 = performance.now();
			const step = (now: number) => {
				const t = Math.min(1, (now - t0) / moveMs);
				const p = at(path, ease(t) * total);
				anim = t < 1 ? [p.x, p.y] : null;
				if (t < 1) raf = requestAnimationFrame(step);
			};
			anim = path[0];
			raf = requestAnimationFrame(step);
		});
		return () => cancelAnimationFrame(raf);
	});
	const moving = $derived(anim !== null);

	// ---- the numbers beside the towns ---------------------------------------------------------------------------
	const labels = $derived.by(() => {
		if (!layout || !w) return new Map<number, { x: number; y: number }>();
		const kids = problem.edges.filter(([u]) => u === cur).map(([, v]) => v);
		if (numbers === 'key') {
			const keep = new Set([problem.goal, cur, question.start, ...(chosen ? [chosen.v] : [])]);
			// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a local map, built whole and never changed after
			const discs = new Map<number, number>();
			for (const t of Object.keys(layout.towns).map(Number))
				discs.set(t, (t === cur ? layout.demo.hub : layout.demo.town + layout.demo.ring) * f + 1);
			const all = placeLabels(layout, k, fl, [...keep], discs);
			return new Map([...all].filter(([t]) => keep.has(t)));
		}
		const priority = [
			problem.goal,
			cur,
			...(trace && live ? [trace.answer] : []),
			question.start,
			...problem.path,
			...kids
		];
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a local map, built whole and never changed after
		const discs = new Map<number, number>();
		for (const t of Object.keys(layout.towns).map(Number))
			discs.set(
				t,
				(t === cur
					? layout.demo.hub
					: t === problem.goal
						? layout.demo.town + layout.demo.ring
						: layout.demo.town) *
					f +
					1
			);
		return placeLabels(layout, k, f, priority, discs);
	});

	// ---- dragging a new road ------------------------------------------------------------------------------------
	let svg: SVGSVGElement | undefined = $state();
	let drag = $state<{ from: number; x: number; y: number; moved: boolean } | null>(null);
	let hot = $state<number | null>(null);
	const local = (e: PointerEvent): Pt => {
		const r = svg!.getBoundingClientRect();
		return [
			view[0] + ((e.clientX - r.left) / r.width) * W,
			view[1] + ((e.clientY - r.top) / r.height) * H
		];
	};
	function nearest(x: number, y: number): number | null {
		if (!layout) return null;
		let best: number | null = null;
		let bd = mm(16);
		for (const [t, p] of Object.entries(layout.towns)) {
			const d = Math.hypot(p[0] - x, p[1] - y);
			if (d < bd) {
				bd = d;
				best = Number(t);
			}
		}
		return best;
	}
	function down(e: PointerEvent, t: number) {
		if (!interactive) return;
		const [x, y] = local(e);
		drag = { from: t, x, y, moved: false };
		svg?.setPointerCapture(e.pointerId);
	}
	function move(e: PointerEvent) {
		if (!drag) return;
		const [x, y] = local(e);
		const [x0, y0] = T(drag.from);
		drag = { ...drag, x, y, moved: drag.moved || Math.hypot(x - x0, y - y0) > mm(10) };
		hot = nearest(x, y);
	}
	function up(e: PointerEvent) {
		if (!drag) return;
		const [x, y] = local(e);
		const to = nearest(x, y);
		const d = drag;
		drag = null;
		hot = null;
		if (!d.moved) onvertex?.(d.from);
		else if (to !== null && to !== d.from) onadd?.(d.from, to);
	}

	/** a parallel copy of a polyline, d mm to its left */
	function offset(p: readonly Pt[], d: number): Pt[] {
		return p.map((q, i) => {
			const a = p[Math.max(0, i - 1)];
			const b = p[Math.min(p.length - 1, i + 1)];
			const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
			return [q[0] - ((b[1] - a[1]) / L) * d, q[1] + ((b[0] - a[0]) / L) * d] as Pt;
		});
	}
	const r2 = (x: number) => Math.round(x * 100) / 100;
	const lit = (key: string) =>
		walked.has(key) || key === answerRoad || hoverRoute.has(key) || asked.has(key);
	const fmt = (v: number) => String(v);
</script>

<div class="map" bind:clientWidth={measured} style:aspect-ratio="{W} / {H}">
	{#if layout && w > 0}
		<svg
			bind:this={svg}
			viewBox="{view[0]} {view[1]} {W} {H}"
			width={fixed ? undefined : w}
			height={fixed ? undefined : (w * H) / W}
			class={{ fill: fixed !== undefined }}
			role="group"
			aria-label={label ??
				`A map of the question: ${Object.keys(layout.towns).length} towns; the walk is at ${cur}, the goal is ${problem.goal}`}
			onpointermove={move}
			onpointerup={up}
			onpointercancel={() => ((drag = null), (hot = null))}
			onpointerleave={() => (hoverTown = null)}
		>
			<!-- the land: woods, then each water's shore under every water's fill, so rivers run into lakes without a seam -->
			<g class="land" aria-hidden="true">
				{#each layout.woods as p, i (i)}<path class="wood" d={shapeD(p)} />{/each}
				{#each layout.ridges as p, i (i)}<path class="ridge" d={shapeD(p)} />{/each}
				{#if layout.water.sea}<path class="shore" d={shapeD(layout.water.sea)} />{/if}
				{#each layout.water.lakes as p, i (i)}<path class="shore" d={shapeD(p)} />{/each}
				{#each layout.water.rivers as r, i (i)}<path class="shore" d={shapeD(r.outline)} />{/each}
				{#if layout.water.sea}<path class="water" d={shapeD(layout.water.sea)} />{/if}
				{#each layout.water.lakes as p, i (i)}<path class="water" d={shapeD(p)} />{/each}
				{#each layout.water.rivers as r, i (i)}<path class="water" d={shapeD(r.outline)} />{/each}
				{#each layout.islands as p, i (i)}<path class="island" d={shapeD(p)} />{/each}
				{#each spots as [x, y], i (i)}
					{@const h = mm(10 * f)}
					<path
						class="tree"
						d="M{x},{y - 0.55 * h}L{x - 0.24 * h},{y - 0.05 * h}L{x + 0.24 * h},{y -
							0.05 * h}ZM{x},{y - 0.3 * h}L{x - 0.31 * h},{y + 0.28 * h}L{x + 0.31 * h},{y +
							0.28 * h}ZM{x - 0.04 * h},{y + 0.26 * h}h{0.08 * h}v{0.2 * h}h{-0.08 * h}Z"
					/>
				{/each}
				<!-- no place names: the maps label only towns -->
			</g>

			<!-- roads: a white casing under each, a grey road, a quiet one-way chevron -->
			<g class="roads">
				{#each roads as r (r.key)}
					{@const d = lineD(r.path)}
					<path class="casing" {d} stroke-width={(1.7 + 2.2) * f} />
					{#if r.added}
						{#each r.wet as run, i (i)}
							<path class="casing" d={lineD(run)} stroke-width={(1.7 + 7) * f} />
							<path class="parapet" d={lineD(offset(run, mm(3.2 * f)))} />
							<path class="parapet" d={lineD(offset(run, -mm(3.2 * f)))} />
						{/each}
					{/if}
					<path
						class={['road', { added: r.added, seen: seen.has(r.key) }]}
						{d}
						stroke-width={1.7 * f}
					/>
					{#if k >= 4.5}
						<path class={['chev', { seen: seen.has(r.key) }]} d={chevronD(r.path, mm(5.2 * f))} />
					{/if}
				{/each}
			</g>

			<!-- bridges: parapets along the road over the river -->
			<g class="bridges" aria-hidden="true">
				{#each layout.bridges as b, i (i)}
					{#if present.has(roadKey(b.road[0], b.road[1]))}
						{@const nx = -b.dir[1]}
						{@const ny = b.dir[0]}
						{@const off = mm(3.0 * f)}
						{@const half = b.width / 2 + 0.55}
						{#each [1, -1] as s (s)}
							{@const qx = b.at[0] + nx * s * off}
							{@const qy = b.at[1] + ny * s * off}
							<path
								class="parapet"
								d="M{r2(qx - b.dir[0] * half + nx * s * 0.3)},{r2(
									qy - b.dir[1] * half + ny * s * 0.3
								)}L{r2(qx - b.dir[0] * (half - 0.3))},{r2(qy - b.dir[1] * (half - 0.3))}L{r2(
									qx + b.dir[0] * (half - 0.3)
								)},{r2(qy + b.dir[1] * (half - 0.3))}L{r2(qx + b.dir[0] * half + nx * s * 0.3)},{r2(
									qy + b.dir[1] * half + ny * s * 0.3
								)}"
							/>
						{/each}
					{/if}
				{/each}
			</g>

			<!-- overpasses: the road on top keeps its white casing across the one beneath -->
			<g class="over" aria-hidden="true">
				{#each layout.overpasses as o, i (i)}
					{@const r = byKey.get(roadKey(o.over[0], o.over[1]))}
					{#if r && present.has(roadKey(o.under[0], o.under[1]))}
						{@const near = r.path.filter((q) => Math.hypot(q[0] - o.at[0], q[1] - o.at[1]) < 2.4)}
						<path class="casing butt" d={lineD(near)} stroke-width={(1.7 + 5) * f} />
						<path class="road butt" d={lineD(near)} stroke-width={1.7 * f} />
					{/if}
				{/each}
			</g>

			<!-- the walks' light: a wide wash and a halo, blurred, under a core on the road itself -->
			{#if glow.length}
				<defs>
					<filter id="wash" x="-20%" y="-20%" width="140%" height="140%">
						<feGaussianBlur stdDeviation={mm(5.5 * f)} />
					</filter>
					<filter id="halo" x="-20%" y="-20%" width="140%" height="140%">
						<feGaussianBlur stdDeviation={mm(1.8 * f)} />
					</filter>
				</defs>
				<g class="glow wash" filter="url(#wash)" aria-hidden="true">
					{#each glow as g (g.key)}
						<path
							d={g.d}
							pathLength="1"
							stroke-dasharray="{g.r} 2"
							stroke-dashoffset={g.r - 1}
							stroke-width={mm(9 * f)}
							stroke-opacity={0.5 * g.b}
						/>
					{/each}
				</g>
				<g class="glow" filter="url(#halo)" aria-hidden="true">
					{#each glow as g (g.key)}
						<path
							d={g.d}
							pathLength="1"
							stroke-dasharray="{g.r} 2"
							stroke-dashoffset={g.r - 1}
							stroke-width={mm(4.2 * f)}
							stroke-opacity={0.55 * g.b}
						/>
					{/each}
				</g>
				<g class="glow" aria-hidden="true">
					{#each glow as g (g.key)}
						<path
							d={g.d}
							pathLength="1"
							stroke-dasharray="{g.r} 2"
							stroke-dashoffset={g.r - 1}
							stroke-width={mm(1.9 * f)}
							stroke-opacity={Math.min(1, 0.35 + 0.65 * g.b)}
						/>
					{/each}
				</g>
			{/if}

			<!-- lit roads: the path asked, the steps walked, the answer, a pair's route -->
			<g class="lit">
				{#each roads as r (r.key)}
					{#if lit(r.key)}
						{@const d = lineD(r.path)}
						{@const kind =
							walked.has(r.key) || r.key === answerRoad
								? 'rose'
								: asked.has(r.key)
									? 'ink'
									: 'pair'}
						<!-- in map units (not a screen-px stroke), so the answer's road can be drawn on along its length -->
						<path
							class={['litroad', kind, { fresh: r.key === answerRoad }]}
							{d}
							pathLength="1"
							stroke-width={mm(2.8 * f)}
						/>
						{#if k >= 4.5}<path class={['chev', kind]} d={chevronD(r.path, mm(6 * f))} />{/if}
					{/if}
				{/each}
			</g>

			<!-- the answer's road, drawn on from the current town, with its arrow at the answer -->
			{#if chosen && chosenPath && chosen.t > 0}
				{@const e = chosenPath.end}
				{@const hl = mm(6.5 * f)}
				{@const hw = mm(3.6 * f)}
				<g class="chosen" aria-hidden="true">
					<path
						class="chosen-casing"
						d={chosenPath.d}
						pathLength="1"
						stroke-dasharray="{chosen.t} 2"
						stroke-width={mm(6 * f)}
					/>
					<path
						class="chosen-road"
						d={chosenPath.d}
						pathLength="1"
						stroke-dasharray="{chosen.t} 2"
						stroke-width={mm(3.4 * f)}
					/>
					{#if chosen.t >= 0.98}
						<path
							class="chosen-head"
							d="M{r2(e.x + e.dx * hl * 0.6)},{r2(e.y + e.dy * hl * 0.6)}L{r2(
								e.x - e.dx * hl * 0.4 - e.dy * hw
							)},{r2(e.y - e.dy * hl * 0.4 + e.dx * hw)}L{r2(
								e.x - e.dx * hl * 0.4 + e.dy * hw
							)},{r2(e.y - e.dy * hl * 0.4 - e.dx * hw)}Z"
						/>
					{/if}
				</g>
			{/if}

			<!-- the roads under the pointer: hovered (its gates, its token), clicked (removed, when edits are on) -->
			{#if onroadhover || (interactive && onedge)}
				<g class="hits">
					{#each roads as r (r.key)}
						<path
							class={['hit', { on: hoverRoad?.u === r.u && hoverRoad?.v === r.v }]}
							d={lineD(r.path)}
							stroke-width={14}
							role="presentation"
							onpointerenter={() => onroadhover?.((hoverRoad = { u: r.u, v: r.v }))}
							onpointerleave={() => onroadhover?.((hoverRoad = null))}
							onclick={() => interactive && onedge?.(r.u, r.v)}
						>
							{#if interactive && onedge}<title>{r.u} → {r.v}: click to remove this road</title
								>{/if}
						</path>
					{/each}
				</g>
			{/if}

			{#if drag?.moved && onadd}
				{@const [x0, y0] = T(drag.from)}
				<line class="drawing" x1={x0} y1={y0} x2={drag.x} y2={drag.y} />
			{/if}

			<!-- towns -->
			<g class="towns">
				{#each Object.entries(layout.towns) as [name, p] (name)}
					{@const t = Number(name)}
					{@const c = scores.get(t)}
					{@const isAnswer =
						(live && answered && trace?.answer === t && t !== cur) ||
						(chosen !== null && chosen.t >= 0.98 && chosen.v === t)}
					{@const lt = light?.get(t) ?? 0}
					<g
						class={[
							'town',
							{
								visited: visited.has(t),
								answer: isAnswer,
								hot: hot === t || hoverEnds.has(t)
							}
						]}
						transform="translate({p[0]} {p[1]})"
						role="button"
						tabindex={interactive ? 0 : -1}
						aria-disabled={!interactive}
						aria-label={interactive ? `town ${t}: make it the goal` : `town ${t}`}
						onpointerdown={(e) => down(e, t)}
						onpointerenter={() => (hoverTown = t)}
						onpointerleave={() => (hoverTown = null)}
						onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && onvertex?.(t)}
					>
						{#if marked?.includes(t)}
							<circle class="mark" r={mm(Math.max(layout.demo.hub * f, 8) + 5)} />
						{/if}
						{#if t === problem.goal}
							<circle
								class="ring"
								r={mm((t === cur ? layout.demo.hub + 3 : layout.demo.town + layout.demo.ring) * f)}
							/>
						{/if}
						<circle
							class="dot"
							r={mm(layout.demo.town * f)}
							style:stroke={isAnswer && chosen
								? PALETTE.paper
								: lt >= 0.3 && !isAnswer
									? mix(lt)
									: undefined}
							style:fill={isAnswer
								? PALETTE.rose
								: visited.has(t)
									? PALETTE.ink
									: c !== undefined && t !== cur
										? diverging(c)
										: undefined}
						/>
						<circle class="pad" r={mm(Math.max(10, layout.demo.town * f + 6))} />
						{#if interactive}<title
								>{t}: click to make it the goal{onadd
									? '; drag to another town to add a road'
									: ''}</title
							>{/if}
					</g>
				{/each}
			</g>

			<!-- numbers -->
			<g class="labels" aria-hidden="true">
				{#each labels as [t, q] (t)}
					{#if t !== cur || moving}
						<text
							class={['num', { goal: t === problem.goal }]}
							x={q.x}
							y={q.y}
							dy="0.36em"
							font-size={mm(layout.demo.font * (numbers === 'key' ? fl : f))}>{fmt(t)}</text
						>
					{/if}
				{/each}
			</g>

			<!-- the current town: the dark disc with its number, driven along each step's road -->
			<g class="marker" transform="translate({r2(marker[0])} {r2(marker[1])})" aria-hidden="true">
				<!-- never smaller than its number: 8 px across the disc's radius, 7.5 px type -->
				<circle r={mm(Math.max(layout.demo.hub * f, 8))} />
				<text dy="0.36em" font-size={mm(Math.max(9.5 * f, 7.5))}>{cur}</text>
			</g>
		</svg>
	{/if}
</div>

<style>
	.map {
		position: relative;
		width: 100%;
		touch-action: none;
		user-select: none;
	}
	svg {
		display: block;
		overflow: hidden;
	}
	svg.fill {
		width: 100%;
		height: auto;
	}
	.wood {
		fill: var(--map-wood);
	}
	.ridge {
		fill: var(--map-ridge);
	}
	.shore {
		fill: none;
		stroke: var(--map-shore);
		stroke-width: 1.1px;
		stroke-linejoin: round;
		vector-effect: non-scaling-stroke;
	}
	.water {
		fill: var(--map-water);
	}
	.island {
		fill: var(--color-paper);
		stroke: var(--map-shore);
		stroke-width: 0.9px;
		vector-effect: non-scaling-stroke;
	}
	.tree {
		fill: var(--map-tree);
	}
	.glow path {
		fill: none;
		stroke: var(--color-rose);
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.chosen path {
		fill: none;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.chosen-casing {
		stroke: var(--color-paper);
	}
	.chosen-road {
		stroke: var(--color-rose);
	}
	.chosen-head {
		fill: var(--color-rose);
	}
	.hit.on {
		stroke: rgb(37 87 167 / 0.22);
	}
	.casing,
	.road,
	.parapet,
	.hit {
		fill: none;
		stroke-linecap: round;
		stroke-linejoin: round;
		vector-effect: non-scaling-stroke;
	}
	.litroad {
		fill: none;
		stroke-linecap: round;
		stroke-linejoin: round;
	}
	.casing {
		stroke: var(--color-paper);
	}
	.butt {
		stroke-linecap: butt;
	}
	.road {
		stroke: var(--map-road);
		transition: stroke 700ms var(--ease);
	}
	.road.seen {
		stroke: var(--color-blue-grey);
	}
	.road.added {
		stroke: var(--color-ink-soft);
		stroke-dasharray: 5 3.5;
	}
	.parapet {
		stroke: var(--map-road-ink);
		stroke-width: 0.9px;
		stroke-linecap: butt;
	}
	.chev {
		fill: var(--map-road-ink);
		opacity: 0.75;
		transition: fill 700ms var(--ease);
	}
	.chev.seen {
		fill: var(--color-ink-soft);
	}
	.chev.rose {
		fill: color-mix(in srgb, var(--color-rose) 70%, var(--color-ink));
		opacity: 1;
	}
	.chev.ink,
	.chev.pair {
		fill: var(--color-ink);
		opacity: 1;
	}
	.litroad.rose {
		stroke: var(--color-rose);
	}
	.litroad.ink {
		stroke: var(--color-ink);
	}
	.litroad.pair {
		stroke: var(--color-ink);
	}
	/* the answer's road lights from the current town to the answer */
	.litroad.fresh {
		stroke-dasharray: 1 1;
		animation: draw 700ms var(--ease) both;
	}
	@keyframes draw {
		from {
			stroke-dashoffset: 1;
		}
		to {
			stroke-dashoffset: 0;
		}
	}
	.hit {
		stroke: transparent;
		cursor: pointer;
	}
	.hit:hover {
		stroke: rgb(31 42 68 / 0.07);
	}
	.drawing {
		stroke: var(--color-ink);
		stroke-width: 1.4px;
		stroke-dasharray: 4 4;
		vector-effect: non-scaling-stroke;
		pointer-events: none;
	}
	.town {
		cursor: pointer;
		outline: none;
	}
	.dot {
		fill: var(--color-paper);
		stroke: var(--color-ink);
		stroke-width: 1.2px;
		vector-effect: non-scaling-stroke;
	}
	.pad {
		fill: transparent;
	}
	.town:hover .dot,
	.town:focus-visible .dot,
	.town.hot .dot {
		stroke-width: 2px;
	}
	.mark {
		fill: rgb(37 87 167 / 0.12);
		stroke: var(--color-cobalt);
		stroke-width: 1.5px;
		vector-effect: non-scaling-stroke;
	}
	.ring {
		fill: none;
		stroke: var(--color-rose);
		stroke-width: 1.5px;
		vector-effect: non-scaling-stroke;
	}
	.num {
		font-family: var(--font-sans);
		font-variant-numeric: tabular-nums;
		text-anchor: middle;
		fill: var(--color-ink);
		pointer-events: none;
	}
	.num.goal {
		fill: var(--color-rose);
		font-weight: 600;
	}
	.marker {
		pointer-events: none;
	}
	.marker circle {
		fill: var(--color-ink);
	}
	.marker text {
		fill: var(--color-paper);
		font-weight: 600;
		text-anchor: middle;
		font-variant-numeric: tabular-nums;
	}
</style>
