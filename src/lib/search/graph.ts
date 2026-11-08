/**
 * Plain graph facts the page states beside the model's answer: who reaches whom, how far, which next steps lead to the
 * goal, and the benchmark's lookahead (a port of `lookahead_depth` in Saparov et al.'s train.py).
 */
import { current, vertices, type Problem } from './tokens.ts';

export type Adj = Map<number, number[]>;

export function children(p: Problem): Adj {
	const out: Adj = new Map();
	for (const v of vertices(p)) out.set(v, []);
	for (const [u, v] of p.edges) out.get(u)!.push(v);
	return out;
}

/** shortest hop counts from `from` (BFS); unreachable vertices are absent */
export function hops(adj: Adj, from: number): Map<number, number> {
	const d = new Map<number, number>([[from, 0]]);
	let frontier = [from];
	while (frontier.length) {
		const next: number[] = [];
		for (const u of frontier)
			for (const v of adj.get(u) ?? [])
				if (!d.has(v)) {
					d.set(v, d.get(u)! + 1);
					next.push(v);
				}
		frontier = next;
	}
	return d;
}

/** all-pairs hop counts among the given vertices (Infinity where there is no path) */
export function distances(adj: Adj, vs: readonly number[]): Float64Array {
	const m = vs.length;
	const at = new Map(vs.map((v, i) => [v, i]));
	const D = new Float64Array(m * m).fill(Infinity);
	vs.forEach((u, i) => {
		for (const [v, h] of hops(adj, u)) {
			const k = at.get(v);
			if (k !== undefined) D[i * m + k] = h;
		}
	});
	return D;
}

/** the next steps from the current vertex that still reach the goal */
export function goodSteps(p: Problem, adj = children(p)): number[] {
	const cur = current(p);
	if (cur === p.goal) return [];
	return (adj.get(cur) ?? []).filter((c) => c === p.goal || hops(adj, c).has(p.goal));
}

export function reaches(adj: Adj, u: number, v: number): boolean {
	return hops(adj, u).has(v);
}

/** would adding u → v close a cycle (or repeat an edge)? */
export function blocked(p: Problem, u: number, v: number, adj = children(p)): string | null {
	if (u === v) return 'a vertex cannot point to itself';
	if ((adj.get(u) ?? []).includes(v)) return `${u} → ${v} is already an edge`;
	if (hops(adj, v).has(u)) return `${u} → ${v} would close a cycle`;
	return null;
}

/**
 * The benchmark's lookahead of stepping from `vertex` to `next` toward `goal`: how many steps one must search before
 * every branch but the right one is ruled out (Saparov et al.'s definition, run on the same frontier rules).
 */
export function lookahead(adj: Adj, vertex: number, next: number, goal: number): number {
	let frontier: [number, number][] = (adj.get(vertex) ?? []).map((v) => [v, v]);
	const visited = new Set(adj.get(vertex) ?? []);
	let depth = 0;
	while (frontier.length) {
		if (frontier.every(([, b]) => b === next)) return depth;
		depth++;
		const nf: [number, number][] = [];
		for (const [v, branch] of frontier) {
			if (v === goal) return depth;
			for (const c of adj.get(v) ?? []) {
				if (!visited.has(c)) {
					nf.push([c, branch]);
					visited.add(c);
				} else if (branch === next) {
					for (let i = 0; i < nf.length; i++) if (nf[i][0] === c) nf[i] = [c, branch];
				}
			}
		}
		frontier = nf;
	}
	return depth;
}
