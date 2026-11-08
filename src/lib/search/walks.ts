/**
 * The propagator's walks as light on a map, shared by the project page's map and the demo. For the current town c the
 * layer solves x = s (I - G)^-1, the weight of every walk from the source s (on the goal) back along the roads; its
 * partial sums x_M = Σ_{m ≤ M} s Gᵐ add the walks one length at a time. A town's light is log10 x on a fixed scale
 * (Fig. 2a of the paper): dark at 10^LO, full at the largest x of the solve, with a gamma that keeps the faint walks
 * faint. A road u → v carries the walks from u that go on through v, so it is lit by x_v.
 */
import type { MapLayout, Pt } from './map.ts';
import { inside } from './map.ts';

/** the light's floor: walk weights below 10^LO are dark (the paper's Fig. 2a) */
export const LO = -3.6;
const GAMMA = 1.5;

/** a town's brightness for walk weight x, on the scale from 10^LO to 10^hi */
export function level(x: number, hi: number): number {
	const t = (Math.log10(Math.max(x, 1e-30)) - LO) / Math.max(1e-9, hi - LO);
	return Math.pow(Math.min(1, Math.max(0, t)), GAMMA);
}

/**
 * What the map animates for one question: per town, log10 of the partial sums x_M for M = 0 … reach (the walk length
 * at which the walks from the goal reach the current town), and of the solve's x; only towns the light ever reaches
 * are listed, every other town stays dark.
 */
export interface WalkFrames {
	/** town → log10 x_M, M = 0 … reach */
	sums: Record<string, number[]>;
	/** town → log10 x, the solve */
	solve: Record<string, number>;
	/** log10 of the largest x of the solve: the scale's top */
	hi: number;
	/** the walk length at which the walks from the goal first reach the current town (roads from it to the goal) */
	reach: number;
	current: number;
	goal: number;
	answer: number;
	/** the channel these walks are read from (the one that carries the answer) */
	channel: number;
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;

/** roads from each town to the goal (a breadth-first search along the roads backwards from the goal) */
export function toGoal(edges: readonly [number, number][], goal: number): Map<number, number> {
	const into = new Map<number, number[]>();
	for (const [u, v] of edges) {
		if (!into.has(v)) into.set(v, []);
		into.get(v)!.push(u);
	}
	const d = new Map<number, number>([[goal, 0]]);
	let frontier = [goal];
	while (frontier.length) {
		const next: number[] = [];
		for (const v of frontier)
			for (const u of into.get(v) ?? [])
				if (!d.has(u)) {
					d.set(u, d.get(v)! + 1);
					next.push(u);
				}
		frontier = next;
	}
	return d;
}

/**
 * The frames from one run: the walk channel's source row s, its gates G (m × m), the solve's x, over `names`.
 * `sums(M)` is supplied by the caller (partialSums in model.ts) so this module stays free of the model.
 */
export function framesFrom(opts: {
	names: readonly number[];
	X: ArrayLike<number>;
	sums: Float64Array[];
	edges: readonly [number, number][];
	current: number;
	goal: number;
	answer: number;
	channel: number;
}): WalkFrames {
	const { names, X, sums } = opts;
	let mx = 0;
	for (let i = 0; i < names.length; i++) mx = Math.max(mx, X[i]);
	const hi = Math.log10(Math.max(mx, 1e-30));
	const reach = sums.length - 1;
	const out: WalkFrames = {
		sums: {},
		solve: {},
		hi: r3(hi),
		reach,
		current: opts.current,
		goal: opts.goal,
		answer: opts.answer,
		channel: opts.channel
	};
	names.forEach((v, i) => {
		if (X[i] < 10 ** LO) return;
		out.solve[v] = r3(Math.log10(X[i]));
		out.sums[v] = sums.map((s) => r3(Math.log10(Math.max(s[i], 1e-30))));
	});
	return out;
}

/** the walk length to animate to: until the walks from the goal reach the current town (at least one step) */
export function reachOf(edges: readonly [number, number][], current: number, goal: number) {
	return Math.max(1, toGoal(edges, goal).get(current) ?? 24);
}

/**
 * The light at playhead M (a float from 0 to frames.reach, then blended toward the solve by `solved`, 0 to 1): town →
 * brightness in [0, 1]. Between two lengths the log weight moves smoothly.
 */
export function lightAt(f: WalkFrames, M: number, solved: number): Map<number, number> {
	const out = new Map<number, number>();
	const lo = Math.max(0, Math.min(f.reach, Math.floor(M)));
	const hiM = Math.min(f.reach, lo + 1);
	const t = Math.max(0, Math.min(1, M - lo));
	for (const [k, s] of Object.entries(f.sums)) {
		let lx = s[lo] + (s[hiM] - s[lo]) * t;
		if (solved > 0) lx = lx + (f.solve[k] - lx) * solved;
		const b = level(10 ** lx, f.hi);
		if (b > 0) out.set(Number(k), b);
	}
	return out;
}

/**
 * Spots for the woods' small conifers (the paper's map symbol): well inside each wood and off every road, farthest
 * from the wood's edge first, a few per wood by its area. `scale` is px per mm of the drawing.
 */
export function treeSpots(m: MapLayout, scale: number): Pt[] {
	const roadPts: Pt[] = [];
	for (const r of m.roads) {
		for (let i = 1; i < r.path.length; i++) {
			const [x0, y0] = r.path[i - 1];
			const [x1, y1] = r.path[i];
			const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 0.6));
			for (let k = 0; k < n; k++)
				roadPts.push([x0 + ((x1 - x0) * k) / n, y0 + ((y1 - y0) * k) / n]);
		}
	}
	const towns = Object.values(m.towns);
	const near = (p: Pt, pts: readonly Pt[], d: number) =>
		pts.some(
			(q) =>
				Math.abs(q[0] - p[0]) < d &&
				Math.abs(q[1] - p[1]) < d &&
				Math.hypot(q[0] - p[0], q[1] - p[1]) < d
		);
	const edgeDist = (p: Pt, poly: readonly Pt[]) => {
		let d = Infinity;
		for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
			const [ax, ay] = poly[j];
			const [bx, by] = poly[i];
			const dx = bx - ax;
			const dy = by - ay;
			const L = dx * dx + dy * dy || 1;
			const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L));
			d = Math.min(d, Math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy));
		}
		return d;
	};
	const out: Pt[] = [];
	const g = 6 / scale; // a candidate every 6 px
	for (const poly of m.woods) {
		let x0 = Infinity;
		let y0 = Infinity;
		let x1 = -Infinity;
		let y1 = -Infinity;
		let area = 0;
		for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
			x0 = Math.min(x0, poly[i][0]);
			y0 = Math.min(y0, poly[i][1]);
			x1 = Math.max(x1, poly[i][0]);
			y1 = Math.max(y1, poly[i][1]);
			area += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
		}
		area = (Math.abs(area) / 2) * scale * scale; // px²
		const want = Math.max(1, Math.min(7, Math.round(area / 2600)));
		const cand: [Pt, number][] = [];
		for (let x = x0; x <= x1; x += g)
			for (let y = y0; y <= y1; y += g) {
				const p: Pt = [x, y];
				if (!inside(p, poly)) continue;
				const de = edgeDist(p, poly);
				if (de * scale < 8 || near(p, roadPts, 9 / scale) || near(p, towns, 11 / scale)) continue;
				cand.push([p, de]);
			}
		cand.sort((a, b) => b[1] - a[1]);
		const pick: Pt[] = [];
		for (const [p] of cand) {
			if (pick.every((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) * scale > 22)) pick.push(p);
			if (pick.length >= want) break;
		}
		out.push(...pick);
	}
	return out.map(([x, y]) => [Math.round(x * 100) / 100, Math.round(y * 100) / 100] as Pt);
}

const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹⁻';
/** a small number as the page writes it: 0.97, 0.004, or 2 × 10⁻⁵ */
export function sci(x: number | null): string {
	if (x === null || !Number.isFinite(x)) return '–';
	if (x === 0) return '0';
	if (x >= 0.995) return x.toFixed(2);
	if (x >= 0.01) return x.toFixed(2);
	if (x >= 0.001) return x.toFixed(3);
	const e = Math.floor(Math.log10(x));
	const mant = Math.round(x / 10 ** e);
	const exp = String(mant === 10 ? e + 1 : e)
		.split('')
		.map((c) => (c === '-' ? SUP[10] : SUP[Number(c)]))
		.join('');
	return `${mant === 10 ? 1 : mant} × 10${exp}`;
}
