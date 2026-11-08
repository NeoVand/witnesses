/**
 * A layered drawing of a directed acyclic graph (the Sugiyama recipe, kept small): every vertex gets a rank so that
 * edges point one way, long edges pass through invisible bends, the order within each rank is improved by barycentre
 * sweeps, and positions are relaxed toward neighbours without breaking the order. The drawing runs left to right or top
 * to bottom, whichever leaves the vertices more room in the box it must fit.
 */

export interface Placed {
	/** vertex name → position, in layout units (one unit between ranks, one between neighbours in a rank) */
	at: Map<number, { rank: number; slot: number }>;
	/** each edge as the slots it passes through, rank by rank, from u to v */
	routes: Map<string, { rank: number; slot: number }[]>;
	ranks: number;
	/** the most vertices (and bends) in one rank */
	breadth: number;
	/** vertices in drawing order: rank by rank, top to bottom */
	order: number[];
}

export const edgeKey = (u: number, v: number) => `${u}>${v}`;

/** ranks by longest path from the sources, then each vertex pulled as close to its children as it can go */
function rank(vs: number[], edges: [number, number][], first: number): Map<number, number> {
	const kids = new Map<number, number[]>(vs.map((v) => [v, []]));
	const parents = new Map<number, number[]>(vs.map((v) => [v, []]));
	for (const [u, v] of edges) {
		kids.get(u)!.push(v);
		parents.get(v)!.push(u);
	}
	// topological order (Kahn); a cycle, which the page never builds, falls back to input order
	const indeg = new Map(vs.map((v) => [v, parents.get(v)!.length]));
	const queue = vs
		.filter((v) => indeg.get(v) === 0)
		.sort((a, b) => (a === first ? -1 : b === first ? 1 : 0));
	const topo: number[] = [];
	while (queue.length) {
		const u = queue.shift()!;
		topo.push(u);
		for (const v of kids.get(u)!) {
			indeg.set(v, indeg.get(v)! - 1);
			if (indeg.get(v) === 0) queue.push(v);
		}
	}
	for (const v of vs) if (!topo.includes(v)) topo.push(v);
	const r = new Map<number, number>();
	for (const v of topo) r.set(v, Math.max(0, ...parents.get(v)!.map((u) => (r.get(u) ?? -1) + 1)));
	// pull sources and early vertices right, next to their nearest child (shorter edges), except the start
	for (const v of [...topo].reverse()) {
		const ks = kids.get(v)!;
		if (!ks.length || v === first) continue;
		const room = Math.min(...ks.map((k) => r.get(k)!)) - 1;
		if (room > r.get(v)!) r.set(v, room);
	}
	return r;
}

function crossings(
	layers: number[][],
	links: [number, number][],
	pos: Map<number, number>,
	of: Map<number, number>
) {
	let c = 0;
	const byRank = new Map<number, [number, number][]>();
	for (const [a, b] of links) {
		const k = of.get(a)!;
		if (!byRank.has(k)) byRank.set(k, []);
		byRank.get(k)!.push([pos.get(a)!, pos.get(b)!]);
	}
	for (const es of byRank.values())
		for (let i = 0; i < es.length; i++)
			for (let j = i + 1; j < es.length; j++)
				if ((es[i][0] - es[j][0]) * (es[i][1] - es[j][1]) < 0) c++;
	void layers;
	return c;
}

export function place(vs: number[], edges: [number, number][], first: number): Placed {
	const r = rank(vs, edges, first);
	// nodes: real vertices keep their names; bends get negative ids
	let nextBend = -1;
	const of = new Map<number, number>(r);
	const links: [number, number][] = [];
	const chains = new Map<string, number[]>();
	for (const [u, v] of edges) {
		const chain = [u];
		for (let k = r.get(u)! + 1; k < r.get(v)!; k++) {
			const b = nextBend--;
			of.set(b, k);
			chain.push(b);
		}
		chain.push(v);
		for (let i = 0; i + 1 < chain.length; i++) links.push([chain[i], chain[i + 1]]);
		chains.set(edgeKey(u, v), chain);
	}
	const ranks = Math.max(0, ...of.values()) + 1;
	const layers: number[][] = Array.from({ length: ranks }, () => []);
	// first order: depth first from the start, so a branch stays together
	const down = new Map<number, number[]>();
	const up = new Map<number, number[]>();
	for (const [a, b] of links) {
		if (!down.has(a)) down.set(a, []);
		if (!up.has(b)) up.set(b, []);
		down.get(a)!.push(b);
		up.get(b)!.push(a);
	}
	const seen = new Set<number>();
	const visit = (x: number) => {
		if (seen.has(x)) return;
		seen.add(x);
		layers[of.get(x)!].push(x);
		for (const y of down.get(x) ?? []) visit(y);
	};
	visit(first);
	for (const v of [...of.keys()].sort((a, b) => of.get(a)! - of.get(b)!)) visit(v);

	const pos = new Map<number, number>();
	const index = () => layers.forEach((l) => l.forEach((x, i) => pos.set(x, i)));
	index();
	let best = layers.map((l) => [...l]);
	let bestC = crossings(layers, links, pos, of);
	const sweep = (dir: 1 | -1) => {
		const ks =
			dir === 1 ? [...Array(ranks).keys()].slice(1) : [...Array(ranks).keys()].reverse().slice(1);
		for (const k of ks) {
			const nb = dir === 1 ? up : down;
			const bary = new Map<number, number>();
			for (const x of layers[k]) {
				const ns = (nb.get(x) ?? []).map((y) => pos.get(y)!);
				bary.set(x, ns.length ? ns.reduce((a, b) => a + b, 0) / ns.length : pos.get(x)!);
			}
			layers[k].sort((a, b) => bary.get(a)! - bary.get(b)! || pos.get(a)! - pos.get(b)!);
			layers[k].forEach((x, i) => pos.set(x, i));
		}
	};
	for (let it = 0; it < 10; it++) {
		sweep(it % 2 === 0 ? 1 : -1);
		const c = crossings(layers, links, pos, of);
		if (c < bestC) {
			bestC = c;
			best = layers.map((l) => [...l]);
		}
	}
	best.forEach((l, k) => (layers[k] = l));
	index();

	// slots: start from the order, then relax toward neighbours' mean, keeping order and one unit apart
	const breadth = Math.max(1, ...layers.map((l) => l.length));
	const y = new Map<number, number>();
	for (const l of layers) l.forEach((x, i) => y.set(x, i - (l.length - 1) / 2));
	for (let it = 0; it < 24; it++) {
		const ks = it % 2 === 0 ? [...Array(ranks).keys()] : [...Array(ranks).keys()].reverse();
		for (const k of ks) {
			const l = layers[k];
			const want = l.map((x) => {
				const ns = [...(up.get(x) ?? []), ...(down.get(x) ?? [])].map((q) => y.get(q)!);
				return ns.length ? ns.reduce((a, b) => a + b, 0) / ns.length : y.get(x)!;
			});
			// keep one unit apart: push down from the top, then pull back up from the bottom, then average
			const a = [...want];
			for (let i = 1; i < a.length; i++) a[i] = Math.max(a[i], a[i - 1] + 1);
			const b = [...want];
			for (let i = b.length - 2; i >= 0; i--) b[i] = Math.min(b[i], b[i + 1] - 1);
			l.forEach((x, i) => y.set(x, (a[i] + b[i]) / 2));
			for (let i = 1; i < l.length; i++)
				if (y.get(l[i])! < y.get(l[i - 1])! + 1) y.set(l[i], y.get(l[i - 1])! + 1);
		}
	}
	const ys = [...y.values()];
	const lo = Math.min(...ys);
	const at = new Map<number, { rank: number; slot: number }>();
	for (const v of vs) at.set(v, { rank: r.get(v)!, slot: y.get(v)! - lo });
	const routes = new Map<string, { rank: number; slot: number }[]>();
	for (const [k, chain] of chains)
		routes.set(
			k,
			chain.map((x) => ({ rank: of.get(x)!, slot: y.get(x)! - lo }))
		);
	const order = layers.flatMap((l) => l.filter((x) => x >= 0));
	return { at, routes, ranks, breadth: Math.max(breadth, Math.max(...ys) - lo + 1), order };
}
