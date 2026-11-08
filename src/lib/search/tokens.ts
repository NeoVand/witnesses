/**
 * The benchmark's input format (Saparov et al., "Transformers Struggle to Learn to Search", ICLR 2025): a directed
 * acyclic graph written as edges `E u v`, then the question `Q start goal`, then `P` and the path so far, left-padded
 * to a fixed length. With V vertices the input has 3V + 5 tokens; vertex names are token ids 1..V (0 is reserved), and
 * the markers follow them: P = V + 1, E = V + 2, padding = V + 3, Q = V + 4.
 */

export interface Spec {
	/** the largest vertex name */
	V: number;
	/** the input length, 3V + 5 */
	n: number;
	path: number;
	edge: number;
	pad: number;
	query: number;
	vocab: number;
	/** name slots in the model's pair state: token ids 0..V */
	names: number;
}

export function spec(V: number): Spec {
	return {
		V,
		n: 3 * V + 5,
		path: V + 1,
		edge: V + 2,
		pad: V + 3,
		query: V + 4,
		vocab: V + 5,
		names: V + 1
	};
}

/** One question: the graph, where to go, and how far the path has come. */
export interface Problem {
	/** edges in the order the input writes them */
	edges: [number, number][];
	start: number;
	goal: number;
	/** the path so far, from the start; its last vertex is the current one */
	path: number[];
}

export function parse(tokens: readonly number[], s: Spec): Problem {
	const t = tokens.filter((x) => x !== s.pad);
	const edges: [number, number][] = [];
	let i = 0;
	while (i < t.length && t[i] === s.edge) {
		edges.push([t[i + 1], t[i + 2]]);
		i += 3;
	}
	if (t[i] !== s.query) throw new Error(`tokens: expected Q at ${i}, found ${t[i]}`);
	const start = t[i + 1];
	const goal = t[i + 2];
	if (t[i + 3] !== s.path) throw new Error(`tokens: expected P at ${i + 3}, found ${t[i + 3]}`);
	const path = t.slice(i + 4);
	if (!path.length) throw new Error('tokens: empty path');
	return { edges, start, goal, path };
}

/** the problem as the model reads it, or null when it no longer fits in n tokens */
export function encode(p: Problem, s: Spec): number[] | null {
	const body: number[] = [];
	for (const [u, v] of p.edges) body.push(s.edge, u, v);
	body.push(s.query, p.start, p.goal, s.path, ...p.path);
	if (body.length > s.n) return null;
	return [...Array(s.n - body.length).fill(s.pad), ...body];
}

/** the free token budget: how many more path steps or edges (3 tokens each) still fit */
export function room(p: Problem, s: Spec): number {
	return s.n - (3 * p.edges.length + 4 + p.path.length);
}

export const current = (p: Problem) => p.path[p.path.length - 1];

/** every vertex the input names, in order of first appearance */
export function vertices(p: Problem): number[] {
	const seen = new Set<number>();
	for (const [u, v] of p.edges) seen.add(u).add(v);
	seen.add(p.start).add(p.goal);
	for (const v of p.path) seen.add(v);
	return [...seen];
}

/**
 * New random names for a question's towns (an injective map into 1..V) and its edges in a new random order: the same
 * graph, written as the benchmark could have written it. Returns the renamed problem and the map old → new.
 */
export function renamed(
	p: Problem,
	V: number,
	rand = Math.random
): { problem: Problem; to: Map<number, number> } {
	const pool = Array.from({ length: V }, (_, i) => i + 1);
	for (let i = pool.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[pool[i], pool[j]] = [pool[j], pool[i]];
	}
	const to = new Map<number, number>();
	vertices(p).forEach((v, i) => to.set(v, pool[i]));
	const r = (v: number) => to.get(v)!;
	const edges = p.edges.map(([u, v]) => [r(u), r(v)] as [number, number]);
	for (let i = edges.length - 1; i > 0; i--) {
		const j = Math.floor(rand() * (i + 1));
		[edges[i], edges[j]] = [edges[j], edges[i]];
	}
	return { problem: { edges, start: r(p.start), goal: r(p.goal), path: p.path.map(r) }, to };
}
