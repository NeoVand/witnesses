/**
 * A question drawn as a road map: the layout (static/maps/<question>.json, one per exported question, laid out offline
 * with its own land: towns, roads, bridges, water, woods) and the plain geometry the map view needs from it. Units are
 * the layout's millimetres, y down; the view scales them to its box.
 *
 * Every road in a layout is an edge of the question, drawn from its first town to its second. A road the visitor adds
 * is drawn as a gentle curve between its towns, bridged where it crosses water.
 */

export type Pt = [number, number];

export interface MapLayout {
	size: Pt;
	step: number;
	start: number;
	goal: number;
	current: number;
	path: number[];
	route: number[] | null;
	towns: Record<string, Pt>;
	roads: { edge: [number, number]; rank: number; path: Pt[] }[];
	overpasses: { at: Pt; over: [number, number]; under: [number, number] }[];
	bridges: { at: Pt; dir: Pt; river_dir: Pt; width: number; road: [number, number] }[];
	water: {
		sea: Pt[] | null;
		coast: Pt[] | null;
		lakes: Pt[][];
		rivers: { centre: Pt[]; width: [number, number]; outline: Pt[] }[];
	};
	woods: Pt[][];
	ridges: Pt[][];
	islands: Pt[][];
	names: { text: string; at: Pt; rot: number; kind: string }[];
	/** a spot beside each town for its number, placed for the sheet drawn `demo.px` px wide */
	labels: Record<string, [number, number, string]>;
	planar: boolean;
	demo: {
		px: number;
		town: number;
		hub: number;
		ring: number;
		font: number;
		digit: number;
		cap: number;
		gap: number;
		pad: number;
	};
}

export const roadKey = (u: number, v: number) => `${u}>${v}`;

/** an open polyline as an SVG path */
export function lineD(p: readonly Pt[]): string {
	if (!p.length) return '';
	let d = `M${p[0][0]},${p[0][1]}`;
	for (let i = 1; i < p.length; i++) d += `L${p[i][0]},${p[i][1]}`;
	return d;
}

/** a closed outline as an SVG path */
export const shapeD = (p: readonly Pt[]) => (p.length ? lineD(p) + 'Z' : '');

export function length(p: readonly Pt[]): number {
	let s = 0;
	for (let i = 1; i < p.length; i++) s += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
	return s;
}

/** the point at arc length s along p, and the unit direction there */
export function at(p: readonly Pt[], s: number): { x: number; y: number; dx: number; dy: number } {
	let left = Math.max(0, s);
	for (let i = 1; i < p.length; i++) {
		const [x0, y0] = p[i - 1];
		const [x1, y1] = p[i];
		const L = Math.hypot(x1 - x0, y1 - y0);
		if (left <= L || i === p.length - 1) {
			const f = L > 0 ? Math.min(1, left / L) : 0;
			return {
				x: x0 + (x1 - x0) * f,
				y: y0 + (y1 - y0) * f,
				dx: L > 0 ? (x1 - x0) / L : 1,
				dy: L > 0 ? (y1 - y0) / L : 0
			};
		}
		left -= L;
	}
	const [x, y] = p[p.length - 1] ?? [0, 0];
	return { x, y, dx: 1, dy: 0 };
}

/** a small one-way chevron at the middle of a road, `size` long, as an SVG path */
export function chevronD(p: readonly Pt[], size: number): string {
	const m = at(p, length(p) / 2);
	const { x, y, dx, dy } = m;
	const nx = -dy;
	const ny = dx;
	const h = size / 2;
	const tip: Pt = [x + dx * h, y + dy * h];
	const back: Pt = [x - dx * h, y - dy * h];
	const r = (q: number) => Math.round(q * 100) / 100;
	return (
		`M${r(tip[0])},${r(tip[1])}L${r(back[0] + nx * size * 0.55)},${r(back[1] + ny * size * 0.55)}` +
		`L${r(back[0] + dx * size * 0.45)},${r(back[1] + dy * size * 0.45)}` +
		`L${r(back[0] - nx * size * 0.55)},${r(back[1] - ny * size * 0.55)}Z`
	);
}

export function inside(q: Pt, poly: readonly Pt[]): boolean {
	let c = false;
	for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
		const [xi, yi] = poly[i];
		const [xj, yj] = poly[j];
		if (yi > q[1] !== yj > q[1] && q[0] < ((xj - xi) * (q[1] - yi)) / (yj - yi) + xi) c = !c;
	}
	return c;
}

/** every body of water a road may cross on a bridge */
export function waters(m: MapLayout): Pt[][] {
	return [
		...(m.water.sea ? [m.water.sea] : []),
		...m.water.lakes,
		...m.water.rivers.map((r) => r.outline)
	];
}

/**
 * A road the visitor adds between towns a and b: a quadratic curve bowed to the side with fewer towns near it, with
 * the stretches that cross water (drawn as bridges).
 */
export function newRoad(m: MapLayout, a: Pt, b: Pt): { path: Pt[]; wet: Pt[][] } {
	const dx = b[0] - a[0];
	const dy = b[1] - a[1];
	const L = Math.hypot(dx, dy) || 1;
	const towns = Object.values(m.towns);
	const make = (k: number): Pt[] => {
		const cx = (a[0] + b[0]) / 2 - (dy / L) * k * L;
		const cy = (a[1] + b[1]) / 2 + (dx / L) * k * L;
		const n = Math.max(12, Math.round(L / 0.5));
		return Array.from({ length: n + 1 }, (_, i) => {
			const t = i / n;
			return [
				(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * cx + t * t * b[0],
				(1 - t) ** 2 * a[1] + 2 * (1 - t) * t * cy + t * t * b[1]
			] as Pt;
		});
	};
	// the bow that passes fewest towns, then the gentlest
	const crowd = (p: Pt[]) => {
		let c = 0;
		for (const t of towns) {
			if (t === a || t === b) continue;
			let d = Infinity;
			for (const q of p) d = Math.min(d, Math.hypot(q[0] - t[0], q[1] - t[1]));
			if (d < m.step * 0.32) c++;
		}
		return c;
	};
	let path = make(0.16);
	let best = Infinity;
	for (const k of [0.16, -0.16, 0.08, -0.08, 0.26, -0.26, 0.36, -0.36, 0]) {
		const p = make(k);
		const sc = crowd(p) * 10 + Math.abs(k) * 4;
		if (sc < best) {
			best = sc;
			path = p;
		}
	}
	const bodies = waters(m);
	const wet: Pt[][] = [];
	let run: Pt[] = [];
	for (const q of path) {
		if (bodies.some((w) => inside(q, w))) run.push(q);
		else if (run.length) {
			wet.push(run);
			run = [];
		}
	}
	if (run.length) wet.push(run);
	return { path, wet };
}

/** a shortest directed route from u to v over the given edges (towns), or null */
export function route(edges: readonly [number, number][], u: number, v: number): number[] | null {
	const out = new Map<number, number[]>();
	for (const [a, b] of edges) {
		if (!out.has(a)) out.set(a, []);
		out.get(a)!.push(b);
	}
	const prev = new Map<number, number>([[u, u]]);
	let frontier = [u];
	while (frontier.length && !prev.has(v)) {
		const next: number[] = [];
		for (const x of frontier)
			for (const y of out.get(x) ?? [])
				if (!prev.has(y)) {
					prev.set(y, x);
					next.push(y);
				}
		frontier = next;
	}
	if (!prev.has(v) || u === v) return null;
	const r = [v];
	while (r[r.length - 1] !== u) r.push(prev.get(r[r.length - 1])!);
	return r.reverse();
}

/**
 * Where each town's number goes at this scale: in the direction the layout chose, at the layout's distance scaled
 * with the marks; labels are kept in order of priority and any that would touch a kept label or another town's mark
 * is dropped (only on screens narrower than the layout's own drawing size does that happen).
 */
export function placeLabels(
	m: MapLayout,
	k: number,
	f: number,
	priority: readonly number[],
	discs: ReadonlyMap<number, number>
): Map<number, { x: number; y: number }> {
	const ref = m.demo.px / m.size[0];
	const out = new Map<number, { x: number; y: number }>();
	const kept: [number, number, number, number][] = [];
	const names = Object.keys(m.towns).map(Number);
	const order = [...new Set([...priority, ...names])];
	for (const t of order) {
		const lab = m.labels[String(t)];
		const p = m.towns[String(t)];
		if (!lab || !p) continue;
		const [ax, ay, txt] = lab;
		const d = Math.hypot(ax - p[0], ay - p[1]) || 1;
		const ux = (ax - p[0]) / d;
		const uy = (ay - p[1]) / d;
		const off = (d * ref * f) / k; // layout px, scaled with the marks, back to mm
		const x = p[0] + ux * off;
		const y = p[1] + uy * off;
		const w = ((m.demo.digit * txt.length + 2 * m.demo.pad) * f) / k;
		const h = ((m.demo.cap + 2 * m.demo.pad) * f) / k;
		const box: [number, number, number, number] = [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
		if (box[0] < 0 || box[1] < 0 || box[2] > m.size[0] || box[3] > m.size[1]) continue;
		let clash = kept.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1]);
		if (!clash)
			for (const u of names) {
				if (u === t) continue;
				const q = m.towns[String(u)];
				const r = (discs.get(u) ?? m.demo.town * f + 1) / k;
				const cx = Math.max(box[0], Math.min(q[0], box[2]));
				const cy = Math.max(box[1], Math.min(q[1], box[3]));
				if (Math.hypot(q[0] - cx, q[1] - cy) < r) {
					clash = true;
					break;
				}
			}
		if (clash) continue;
		kept.push(box);
		out.set(t, { x, y });
	}
	return out;
}

/** the read-out of the current town's row at the playhead, as log-probabilities about the row's mean */
export function centred(
	maps: readonly Float32Array[],
	names: readonly number[],
	cur: number,
	layer: number
): Map<number, number> {
	const out = new Map<number, number>();
	const m = names.length;
	const ci = names.indexOf(cur);
	if (ci < 0 || !maps.length) return out;
	const lo = Math.max(0, Math.min(maps.length - 1, Math.floor(layer)));
	const hi = Math.min(maps.length - 1, lo + 1);
	const f = layer - lo;
	// the recorded runs keep each probability to 1/255: the same floor for live runs keeps the two alike
	const floor = 0.5 / 255;
	const lg = new Float64Array(m);
	let mean = 0;
	for (let j = 0; j < m; j++) {
		const p = maps[lo][ci * m + j] + (maps[hi][ci * m + j] - maps[lo][ci * m + j]) * f;
		lg[j] = Math.log(Math.max(floor, p));
		mean += lg[j] / m;
	}
	for (let j = 0; j < m; j++) out.set(names[j], lg[j] - mean);
	return out;
}

/** the same map with every town renamed by `to` (old name → new name); roads, labels and bridges follow their towns */
export function renameLayout(m: MapLayout, to: ReadonlyMap<number, number>): MapLayout {
	const r = (t: number) => to.get(t) ?? t;
	const keyed = <T>(rec: Record<string, T>) =>
		Object.fromEntries(Object.entries(rec).map(([k, v]) => [String(r(Number(k))), v]));
	return {
		...m,
		start: r(m.start),
		goal: r(m.goal),
		current: r(m.current),
		path: m.path.map(r),
		route: m.route ? m.route.map(r) : null,
		towns: keyed(m.towns),
		labels: Object.fromEntries(
			Object.entries(m.labels).map(([k, [x, y]]) => [
				String(r(Number(k))),
				[x, y, String(r(Number(k)))] as [number, number, string]
			])
		),
		roads: m.roads.map((x) => ({ ...x, edge: [r(x.edge[0]), r(x.edge[1])] as [number, number] })),
		bridges: m.bridges.map((b) => ({
			...b,
			road: [r(b.road[0]), r(b.road[1])] as [number, number]
		})),
		overpasses: m.overpasses.map((o) => ({
			...o,
			over: [r(o.over[0]), r(o.over[1])] as [number, number],
			under: [r(o.under[0]), r(o.under[1])] as [number, number]
		}))
	};
}
