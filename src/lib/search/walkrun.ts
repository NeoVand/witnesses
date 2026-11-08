/**
 * One run of the propagator and everything the map and its panels draw from it: the walk channel's partial sums and
 * solve (as frames), its learned gates on every road both ways, on a sample of town pairs with no road and on the
 * question's own start and goal, and the read-out over the current town's roads. The worker runs it live; the build
 * runs it once for the page's opening (hero.server.ts), so the two are the same computation.
 */
import { partialSums, runProp, type PropWeights } from './model.ts';
import { parse, spec } from './tokens.ts';
import { framesFrom, reachOf, type WalkFrames } from './walks.ts';

export interface Gates {
	/** each road u → v: the gate of the step that walks it back (v → u) and of the step along it (u → v) */
	roads: { u: number; v: number; back: number; along: number }[];
	/** town pairs with no road either way, sampled evenly (at most 600) */
	none: number[];
	/** the question's own pair, a step from the start to the goal */
	pair: number | null;
	start: number;
	goal: number;
}

export interface WalkRun {
	tokens: number[];
	names: number[];
	current: number;
	answer: number;
	frames: WalkFrames;
	/** the current town's roads: each road's far town, its walk weight x (the solve) and the read-out's probability */
	roads: { v: number; x: number; p: number }[];
	/** the read-out over every name present (aligned with `names`): its logits, as the model writes them */
	logits: number[];
	gates: Gates;
	ms: number;
}

const r4 = (x: number) => Number(x.toPrecision(4));

export function walkRun(W: PropWeights, V: number, tokens: number[]): WalkRun {
	const r = runProp(W, tokens);
	const p = parse(tokens, spec(V));
	const h = r.working;
	const m = r.names.length;
	const reach = reachOf(p.edges, r.current, p.goal);
	const sums = partialSums(r.S[h], r.G[h], m, reach);
	const frames = framesFrom({
		names: r.names,
		X: r.X[h],
		sums,
		edges: p.edges,
		current: r.current,
		goal: p.goal,
		answer: r.answer,
		channel: h
	});
	const at = new Map(r.names.map((v, i) => [v, i]));
	const G = r.G[h];
	const g = (u: number, v: number) => {
		const i = at.get(u);
		const j = at.get(v);
		return i === undefined || j === undefined ? null : G[i * m + j];
	};
	const has = new Set(p.edges.map(([u, v]) => `${u}>${v}`));
	const none: number[] = [];
	const stride = Math.max(1, Math.floor((m * m) / 600));
	for (let k = 0; k < m * m; k += stride) {
		const u = r.names[Math.floor(k / m)];
		const v = r.names[k % m];
		if (u === v || has.has(`${u}>${v}`) || has.has(`${v}>${u}`)) continue;
		if ((u === p.start && v === p.goal) || (u === p.goal && v === p.start)) continue;
		none.push(r4(G[k]));
	}
	const gates: Gates = {
		roads: p.edges.map(([u, v]) => ({ u, v, back: r4(g(v, u) ?? 0), along: r4(g(u, v) ?? 0) })),
		none,
		pair: g(p.start, p.goal),
		start: p.start,
		goal: p.goal
	};
	// the read-out over the names, as probabilities, at the current town's roads
	let mx = -Infinity;
	for (const l of r.logits) mx = Math.max(mx, l);
	let z = 0;
	for (const l of r.logits) z += Math.exp(l - mx);
	const roads = p.edges
		.filter(([u]) => u === r.current)
		.map(([, v]) => {
			const i = at.get(v) ?? -1;
			return {
				v,
				x: i >= 0 ? r4(r.X[h][i]) : 0,
				p: i >= 0 ? r4(Math.exp(r.logits[i] - mx) / z) : 0
			};
		});
	return {
		tokens,
		names: r.names,
		current: r.current,
		answer: r.answer,
		frames,
		roads,
		logits: Array.from(r.logits, (x) => Math.round(x * 1e4) / 1e4),
		gates,
		ms: r.ms
	};
}
