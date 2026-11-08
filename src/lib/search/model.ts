/**
 * The pair-state model, in plain TypeScript on Float32Arrays: the same forward pass as the training code, run on the
 * exported weights (static/model/pairn_v49.json). test-data/reference_v49.json holds the training code's own logits and
 * maps for the first questions, and model.spec.ts checks this pass against them.
 *
 * Tokens are pooled into vertex names by identity: name u collects every token that writes u, each read with its two
 * neighbours. A state z[u][v] of D numbers is kept for every ordered pair of names. Each layer joins two relations
 * through every middle vertex k,
 *
 *   z[u][v] += sigmoid(G t[u][v]) * Σ_k (A t[u][k]) (B t[k][v]) / sqrt(#names),  t = the layer-normalised state,
 *
 * then applies a small MLP to each pair. The answer is read from the row of the current vertex: one score per name,
 * w · norm(z[cur][v]). Applying that read-out to every pair, after every layer, gives the maps the demo draws.
 *
 * The second model, the OR join (runLoop, below), shares the start and the read-out and repeats one layer whose join is
 * a soft OR; test-data/reference_orjoin.json holds its training code's logits after every iteration.
 */
import type { Spec } from './tokens.ts';

export interface Block {
	a: Float32Array;
	b: Float32Array;
	g: Float32Array;
	f1: Float32Array;
	f2: Float32Array;
}

export interface Weights {
	spec: Spec;
	D: number;
	L: number;
	/** token embedding width */
	E: number;
	/** rank of the starting scores */
	r: number;
	near: number[];
	tok: Float32Array;
	pos: Float32Array;
	U: Float32Array;
	V: Float32Array;
	C: Float32Array;
	A: Float32Array;
	Bm: Float32Array;
	c: Float32Array;
	w: Float32Array;
	blocks: Block[];
}

/** the export: `meta` and a list of arrays named by their JAX paths (`['blocks'][0]['a']`), row-major */
export interface Exported {
	meta: {
		vertices: number;
		tokens: number;
		pad: number;
		vocab: number;
		names: number;
		cfg: { D: number; layers: number; emb: number; r: number; kind: string };
		near_offsets: number[];
	};
	weights: { name: string; shape: number[]; data: number[] }[];
}

export function loadWeights(x: Exported, spec: Spec): Weights {
	const { meta } = x;
	if (meta.cfg.kind !== 'pairn') throw new Error(`weights: kind ${meta.cfg.kind}, expected pairn`);
	if (meta.vertices !== spec.V || meta.tokens !== spec.n || meta.pad !== spec.pad)
		throw new Error('weights: the export was made for another input size');
	const byName = new Map<string, Float32Array>();
	for (const a of x.weights) {
		const size = a.shape.reduce((p, q) => p * q, 1);
		if (a.data.length !== size)
			throw new Error(`weights: ${a.name} has ${a.data.length} numbers, shape ${a.shape}`);
		byName.set(a.name, Float32Array.from(a.data));
	}
	const get = (k: string) => {
		const v = byName.get(k);
		if (!v) throw new Error(`weights: missing ${k}`);
		return v;
	};
	const L = meta.cfg.layers;
	return {
		spec,
		D: meta.cfg.D,
		L,
		E: meta.cfg.emb,
		r: meta.cfg.r,
		near: meta.near_offsets,
		tok: get("['tok']"),
		pos: get("['pos']"),
		U: get("['U']"),
		V: get("['V']"),
		C: get("['C']"),
		A: get("['A']"),
		Bm: get("['Bm']"),
		c: get("['c']"),
		w: get("['w']"),
		blocks: Array.from({ length: L }, (_, l) => ({
			a: get(`['blocks'][${l}]['a']`),
			b: get(`['blocks'][${l}]['b']`),
			g: get(`['blocks'][${l}]['g']`),
			f1: get(`['blocks'][${l}]['f1']`),
			f2: get(`['blocks'][${l}]['f2']`)
		}))
	};
}

const EPS = 1e-6;

/** layer norm without parameters, in place, over `len` numbers from `at` */
function normInto(
	src: ArrayLike<number>,
	at: number,
	len: number,
	dst: Float32Array | Float64Array,
	to: number
) {
	let mean = 0;
	for (let i = 0; i < len; i++) mean += src[at + i];
	mean /= len;
	let v = 0;
	for (let i = 0; i < len; i++) {
		const d = src[at + i] - mean;
		v += d * d;
	}
	const s = 1 / Math.sqrt(v / len + EPS);
	for (let i = 0; i < len; i++) dst[to + i] = (src[at + i] - mean) * s;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
/** the tanh approximation (jax.nn.gelu's default) */
const gelu = (x: number) =>
	0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

export interface Run {
	/** the names present in the input, ascending; the model's pair state is over these */
	names: number[];
	/** the current vertex (the last token) */
	current: number;
	/** the pair state after the start and after each layer: L + 1 arrays of m × m × D */
	states: Float32Array[];
	/** the read-out applied to every pair of every state: L + 1 arrays of m × m scores, row u = "if u were current" */
	maps: Float32Array[];
	/** the current vertex's row of each map: L + 1 arrays of m scores */
	readouts: Float32Array[];
	/** the answer: the name with the largest final score */
	answer: number;
	ms: number;
}

/** what the start reads: the token features, the name pooling and the starting pair scores (both models share them) */
export interface StartWeights {
	spec: Spec;
	D: number;
	E: number;
	r: number;
	near: number[];
	tok: Float32Array;
	/** absolute position embeddings, or null for a model that reads inputs of any length */
	pos: Float32Array | null;
	U: Float32Array;
	V: Float32Array;
	C: Float32Array;
	A: Float32Array;
	Bm: Float32Array;
	c: Float32Array;
}

export interface Start {
	/** the names present in the input, ascending */
	names: number[];
	/** name → its index in `names`, or -1 */
	at: Int32Array;
	/** the starting pair state, m × m × D */
	z: Float64Array;
}

/**
 * The start: token features (each token's embedding, from the row `row(token)` of the table, read with its two
 * neighbours), pooled into the names they write; then D attention-style scores between names (rank r), plus, for each
 * nearby offset, a product of the two tokens' contexts summed into the pair of names they write.
 */
export function begin(
	W: StartWeights,
	tokens: readonly number[],
	row: (token: number) => number = (t) => t
): Start {
	const { spec, D, E, r } = W;
	const n = spec.n;
	if (tokens.length !== n) throw new Error(`run: ${tokens.length} tokens, expected ${n}`);

	// token features, each with its two neighbours
	const hn = new Float32Array(n * E);
	const tmp = new Float64Array(E);
	for (let i = 0; i < n; i++) {
		const base = row(tokens[i]) * E;
		for (let e = 0; e < E; e++) tmp[e] = W.tok[base + e] + (W.pos ? W.pos[i * E + e] : 0);
		normInto(tmp, 0, E, hn, i * E);
	}
	const ctx = new Float32Array(n * E);
	for (let i = 0; i < n; i++) {
		for (let f = 0; f < E; f++) tmp[f] = hn[i * E + f];
		if (i > 0)
			for (let e = 0; e < E; e++) {
				const h = hn[(i - 1) * E + e];
				for (let f = 0; f < E; f++) tmp[f] += h * W.C[e * E + f];
			}
		if (i < n - 1)
			for (let e = 0; e < E; e++) {
				const h = hn[(i + 1) * E + e];
				for (let f = 0; f < E; f++) tmp[f] += h * W.C[E * E + e * E + f];
			}
		normInto(tmp, 0, E, ctx, i * E);
	}

	// pool tokens into names
	const isName = (t: number) => t >= 0 && t < spec.names;
	const names = [...new Set(tokens.filter(isName))].sort((a, b) => a - b);
	const m = names.length;
	const at = new Int32Array(spec.names).fill(-1);
	names.forEach((v, k) => (at[v] = k));
	const g = new Float64Array(m * E);
	const cnt = new Int32Array(m);
	for (let i = 0; i < n; i++) {
		const k = isName(tokens[i]) ? at[tokens[i]] : -1;
		if (k < 0) continue;
		cnt[k]++;
		for (let e = 0; e < E; e++) g[k * E + e] += ctx[i * E + e];
	}
	for (let k = 0; k < m; k++) {
		const s = 1 / Math.sqrt(Math.max(cnt[k], 1));
		for (let e = 0; e < E; e++) g[k * E + e] *= s;
	}

	// the start: D attention-style scores between names (rank r) ...
	const proj = (M: Float32Array) => {
		const out = new Float64Array(m * D * r);
		for (let k = 0; k < m; k++)
			for (let e = 0; e < E; e++) {
				const x = g[k * E + e];
				if (x === 0) continue;
				const base = e * D * r;
				for (let q = 0; q < D * r; q++) out[k * D * r + q] += x * M[base + q];
			}
		return out;
	};
	const uu = proj(W.U);
	const ww = proj(W.V);
	const z = new Float64Array(m * m * D);
	const sr = 1 / Math.sqrt(r);
	for (let a = 0; a < m; a++)
		for (let b = 0; b < m; b++)
			for (let d = 0; d < D; d++) {
				let s = 0;
				for (let q = 0; q < r; q++) s += uu[(a * D + d) * r + q] * ww[(b * D + d) * r + q];
				z[(a * m + b) * D + d] = s * sr;
			}
	// ... plus, for each nearby offset, a product of the two tokens' contexts, summed into the pair of names they write
	const lin = (x: Float32Array, at0: number, M: Float32Array, base: number, out: Float64Array) => {
		out.fill(0);
		for (let e = 0; e < E; e++) {
			const v = x[at0 + e];
			for (let d = 0; d < D; d++) out[d] += v * M[base + e * D + d];
		}
	};
	const pa = new Float64Array(D);
	const pb = new Float64Array(D);
	W.near.forEach((o, oi) => {
		for (let i = 0; i < n; i++) {
			const j = i + o;
			if (j < 0 || j >= n || !isName(tokens[i]) || !isName(tokens[j])) continue;
			lin(ctx, i * E, W.A, oi * E * D, pa);
			lin(ctx, j * E, W.Bm, oi * E * D, pb);
			const p = (at[tokens[i]] * m + at[tokens[j]]) * D;
			for (let d = 0; d < D; d++) z[p + d] += pa[d] * pb[d] + W.c[oi * D + d];
		}
	});

	return { names, at, z };
}

/** One forward pass, keeping the state after every layer. */
export function run(W: Weights, tokens: readonly number[]): Run {
	const t0 = performance.now();
	const { D, L } = W;
	const n = W.spec.n;
	const { names, at, z: z0 } = begin(W, tokens);
	const m = names.length;
	const z = Float32Array.from(z0);
	const states = [z.slice()];
	const P = m * m;
	const H = 2 * D;
	const t = new Float32Array(P * D);
	// A t and B t, channel-major ([d][u][v]), so the join is one m × m matrix product per channel
	const Ac = new Float32Array(D * P);
	const Bc = new Float32Array(D * P);
	const Tc = new Float32Array(D * P);
	const acc = new Float64Array(H);
	const acc2 = new Float64Array(D);
	const nrm = new Float64Array(D);
	const sv = 1 / Math.sqrt(m);
	for (const blk of W.blocks) {
		for (let p = 0; p < P; p++) {
			normInto(z, p * D, D, t, p * D);
			acc.fill(0, 0, D);
			acc2.fill(0);
			for (let c = 0; c < D; c++) {
				const x = t[p * D + c];
				const row = c * D;
				for (let d = 0; d < D; d++) {
					acc[d] += x * blk.a[row + d];
					acc2[d] += x * blk.b[row + d];
				}
			}
			for (let d = 0; d < D; d++) {
				Ac[d * P + p] = acc[d];
				Bc[d * P + p] = acc2[d];
			}
		}
		// the join through every middle vertex k: Tc[d][u][v] = Σ_k Ac[d][u][k] Bc[d][k][v]
		Tc.fill(0);
		for (let d = 0; d < D; d++) {
			const base = d * P;
			for (let i = 0; i < m; i++) {
				const ti = base + i * m;
				for (let k = 0; k < m; k++) {
					const a = Ac[ti + k];
					if (a === 0) continue;
					const bk = base + k * m;
					for (let j = 0; j < m; j++) Tc[ti + j] += a * Bc[bk + j];
				}
			}
		}
		for (let p = 0; p < P; p++) {
			acc2.fill(0);
			for (let c = 0; c < D; c++) {
				const x = t[p * D + c];
				const row = c * D;
				for (let d = 0; d < D; d++) acc2[d] += x * blk.g[row + d];
			}
			for (let d = 0; d < D; d++) z[p * D + d] += sigmoid(acc2[d]) * Tc[d * P + p] * sv;
			// the per-pair MLP
			normInto(z, p * D, D, nrm, 0);
			acc.fill(0);
			for (let c = 0; c < D; c++) {
				const x = nrm[c];
				const row = c * H;
				for (let h = 0; h < H; h++) acc[h] += x * blk.f1[row + h];
			}
			acc2.fill(0);
			for (let h = 0; h < H; h++) {
				const x = gelu(acc[h]);
				const row = h * D;
				for (let d = 0; d < D; d++) acc2[d] += x * blk.f2[row + d];
			}
			for (let d = 0; d < D; d++) z[p * D + d] += acc2[d];
		}
		states.push(z.slice());
	}

	const maps = states.map((s) => {
		const out = new Float32Array(P);
		for (let p = 0; p < P; p++) {
			normInto(s, p * D, D, nrm, 0);
			let v = 0;
			for (let d = 0; d < D; d++) v += nrm[d] * W.w[d];
			out[p] = v;
		}
		return out;
	});
	const current = tokens[n - 1];
	const ci = at[current];
	if (ci < 0) throw new Error('run: the last token is not a vertex');
	const readouts = maps.map((mp) => mp.slice(ci * m, ci * m + m));
	const fin = readouts[L];
	let best = 0;
	for (let k = 1; k < m; k++) if (fin[k] > fin[best]) best = k;
	return {
		names,
		current,
		states,
		maps,
		readouts,
		answer: names[best],
		ms: performance.now() - t0
	};
}

// ---------------------------------------------------------------------------------------------------------------------
// The OR join, looped: one pair layer run K times (static/model/orjoin_v49.json). Its join over the middle vertex is a
// soft OR, the rate form 1 - exp(-Σ_k a[u][k] c[k][v]) with a = sigmoid(A t[u][k]) and c = sigmoid(B t[k][v]),
// t the layer-normalised state; then z[u][v] += sigmoid(G t[u][v]) · W_o O[u][v], and the same per-pair MLP. No
// 1/sqrt(#names): the OR is bounded however many routes run through the middle. There are no position embeddings and
// names are embedded by their own id from a pool of 128, so the one layer reads graphs of any size up to 128 vertices;
// the markers P, E, padding and Q have their own rows after the pool.

export interface LoopBlock extends Block {
	/** W_o: the OR's D channels mixed back into the state */
	o: Float32Array;
	/** omega, J × D: the series' weights (J = 1: the rate form) */
	om: Float32Array;
	J: number;
}

export interface LoopWeights extends StartWeights {
	/** names 1..pool have their own embedding rows; the markers follow at pool + 1..pool + 4 */
	pool: number;
	w: Float32Array;
	block: LoopBlock;
	/** the iterations the layer was trained with */
	trained: number[];
	params: number;
}

export interface LoopExported {
	meta: {
		vertices: number;
		name_pool: number;
		embed_rows: number;
		cfg: {
			kind: string;
			emb: number;
			D: number;
			r: number;
			loop: boolean;
			nopos: boolean;
			tri: string;
			J: number;
			iters_trained: number[];
		};
		near_offsets: number[];
	};
	weights: { name: string; shape: number[]; data: number[] }[];
}

/** the looped layer's weights, for questions written with `spec` (any size up to the pool) */
export function loadLoopWeights(x: LoopExported, spec: Spec): LoopWeights {
	const { meta } = x;
	const c = meta.cfg;
	if (c.kind !== 'pairn' || !c.loop || !c.nopos || c.tri !== 'newton')
		throw new Error('weights: not the looped OR join');
	if (spec.V > meta.name_pool) throw new Error(`weights: names from ${meta.name_pool} only`);
	const byName = new Map<string, Float32Array>();
	let params = 0;
	for (const a of x.weights) {
		const size = a.shape.reduce((p, q) => p * q, 1);
		if (a.data.length !== size)
			throw new Error(`weights: ${a.name} has ${a.data.length} numbers, shape ${a.shape}`);
		byName.set(a.name, Float32Array.from(a.data));
		params += size;
	}
	const get = (k: string) => {
		const v = byName.get(k);
		if (!v) throw new Error(`weights: missing ${k}`);
		return v;
	};
	const tok = get("['tok']");
	if (tok.length !== meta.embed_rows * c.emb) throw new Error('weights: the embedding table');
	const blk = (k: string) => get(`['blocks'][0]['${k}']`);
	return {
		spec,
		D: c.D,
		E: c.emb,
		r: c.r,
		near: meta.near_offsets,
		tok,
		pos: null,
		U: get("['U']"),
		V: get("['V']"),
		C: get("['C']"),
		A: get("['A']"),
		Bm: get("['Bm']"),
		c: get("['c']"),
		w: get("['w']"),
		pool: meta.name_pool,
		block: {
			a: blk('a'),
			b: blk('b'),
			g: blk('g'),
			f1: blk('f1'),
			f2: blk('f2'),
			o: blk('o'),
			om: blk('om'),
			J: blk('om').length / c.D
		},
		trained: c.iters_trained,
		params
	};
}

export interface LoopRun {
	names: number[];
	current: number;
	/** the read-out applied to every pair, after the start and after each iteration: K + 1 arrays of m × m scores */
	maps: Float32Array[];
	/** the current vertex's row of each map */
	readouts: Float32Array[];
	/** the answer after each iteration (index 0: the start) */
	answers: number[];
	/** the answer after the last iteration */
	answer: number;
	ms: number;
}

/**
 * One forward pass of the looped layer, K iterations, reading out after each. The state is kept in float64 (the
 * training code's float32 differs from it by at most 2.1e-5 on the reference questions).
 */
export function runLoop(W: LoopWeights, tokens: readonly number[], K: number): LoopRun {
	const t0 = performance.now();
	const { D, spec, pool } = W;
	const V = spec.V;
	// names keep their own rows; the markers P, E, padding and Q sit after the pool
	const { names, at, z } = begin(W, tokens, (t) => (t > V ? pool + (t - V) : t));
	const m = names.length;
	const P = m * m;
	const H = 2 * D;
	const blk = W.block;
	const t = new Float64Array(P * D);
	const G = new Float64Array(P * D);
	// sigmoid(A t) and sigmoid(B t), channel-major ([d][u][v]), so the join is one m × m product per channel and order
	const Ac = new Float64Array(D * P);
	const Bc = new Float64Array(D * P);
	const Lc = new Float64Array(D * P);
	const Aq = new Float64Array(D * P);
	const Bq = new Float64Array(D * P);
	const acc = new Float64Array(H);
	const acc2 = new Float64Array(D);
	const acc3 = new Float64Array(D);
	const nrm = new Float64Array(D);
	const O = new Float64Array(D);

	const readAll = () => {
		const out = new Float32Array(P);
		for (let p = 0; p < P; p++) {
			normInto(z, p * D, D, nrm, 0);
			let v = 0;
			for (let d = 0; d < D; d++) v += nrm[d] * W.w[d];
			out[p] = v;
		}
		return out;
	};
	const maps = [readAll()];
	for (let it = 0; it < K; it++) {
		for (let p = 0; p < P; p++) {
			normInto(z, p * D, D, t, p * D);
			acc.fill(0, 0, D);
			acc2.fill(0);
			acc3.fill(0);
			for (let c = 0; c < D; c++) {
				const x = t[p * D + c];
				const row = c * D;
				for (let d = 0; d < D; d++) {
					acc[d] += x * blk.a[row + d];
					acc2[d] += x * blk.b[row + d];
					acc3[d] += x * blk.g[row + d];
				}
			}
			for (let d = 0; d < D; d++) {
				Ac[d * P + p] = sigmoid(acc[d]);
				Bc[d * P + p] = sigmoid(acc2[d]);
				G[p * D + d] = sigmoid(acc3[d]);
			}
		}
		// L[d][u][v] = Σ_q omega_q[d] / q · Σ_k a[d][u][k]^q c[d][k][v]^q  (J = 1: the plain sum of a · c)
		Lc.fill(0);
		for (let q = 1; q <= blk.J; q++) {
			for (let i = 0; i < D * P; i++) {
				Aq[i] = q === 1 ? Ac[i] : Ac[i] ** q;
				Bq[i] = q === 1 ? Bc[i] : Bc[i] ** q;
			}
			for (let d = 0; d < D; d++) {
				const base = d * P;
				const wq = blk.om[(q - 1) * D + d] / q;
				for (let i = 0; i < m; i++) {
					const li = base + i * m;
					for (let k = 0; k < m; k++) {
						const a = Aq[li + k] * wq;
						if (a === 0) continue;
						const bk = base + k * m;
						for (let j = 0; j < m; j++) Lc[li + j] += a * Bq[bk + j];
					}
				}
			}
		}
		for (let p = 0; p < P; p++) {
			// the soft OR, 1 - exp(-L) (L floored at -20, as in training), mixed by W_o and gated
			for (let c = 0; c < D; c++) O[c] = -Math.expm1(-Math.max(Lc[c * P + p], -20));
			acc2.fill(0);
			for (let c = 0; c < D; c++) {
				const x = O[c];
				const row = c * D;
				for (let d = 0; d < D; d++) acc2[d] += x * blk.o[row + d];
			}
			for (let d = 0; d < D; d++) z[p * D + d] += G[p * D + d] * acc2[d];
			// the per-pair MLP
			normInto(z, p * D, D, nrm, 0);
			acc.fill(0);
			for (let c = 0; c < D; c++) {
				const x = nrm[c];
				const row = c * H;
				for (let h = 0; h < H; h++) acc[h] += x * blk.f1[row + h];
			}
			acc2.fill(0);
			for (let h = 0; h < H; h++) {
				const x = gelu(acc[h]);
				const row = h * D;
				for (let d = 0; d < D; d++) acc2[d] += x * blk.f2[row + d];
			}
			for (let d = 0; d < D; d++) z[p * D + d] += acc2[d];
		}
		maps.push(readAll());
	}

	const current = tokens[spec.n - 1];
	const ci = at[current];
	if (ci < 0) throw new Error('run: the last token is not a vertex');
	const readouts = maps.map((mp) => mp.slice(ci * m, ci * m + m));
	const answers = readouts.map((r) => {
		let best = 0;
		for (let k = 1; k < m; k++) if (r[k] > r[best]) best = k;
		return names[best];
	});
	return {
		names,
		current,
		maps,
		readouts,
		answers,
		answer: answers[K],
		ms: performance.now() - t0
	};
}

/** softmax of one row of scores */
export function softmax(xs: ArrayLike<number>): Float32Array {
	let mx = -Infinity;
	for (let i = 0; i < xs.length; i++) mx = Math.max(mx, xs[i]);
	const out = new Float32Array(xs.length);
	let s = 0;
	for (let i = 0; i < xs.length; i++) s += out[i] = Math.exp(xs[i] - mx);
	for (let i = 0; i < xs.length; i++) out[i] /= s;
	return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagator (static/model/prop_v49.json): one layer that sums every walk of every length in one linear solve. It
// shares the start and the read-out. In each of four channels h it forms, from the normalised pair state t,
//
//   gates   G[u][v] = t0 · sigmoid(pg_h · t[u][v] + pgb_h), each row then scaled so that it sums to at most t0,
//   source  S[u][v] = sigmoid(ps_h · t[u][v] + psb_h),
//   walks   X = S (I - G)^-1 = S + S G + S G² + …,
//
// with t0 = 0.99: a row sum below one keeps the spectral radius of G below one on any graph, so the walk sum converges.
// The features log(X + 1e-20) and 1 - exp(-X) of the four channels are mixed by po and gated, z[u][v] += sigmoid(g t)
// · po f, and the per-pair MLP follows. The read-out uses only the current town's row, so only that row of X is
// solved for: one LU of (I - G)ᵀ per channel and one right-hand side, the current row of S. No walk is ever formed.

export interface PropWeights extends StartWeights {
	pool: number;
	w: Float32Array;
	/** the gate and source maps, D × H, with their biases */
	pg: Float32Array;
	pgb: Float32Array;
	ps: Float32Array;
	psb: Float32Array;
	/** the 2H features mixed back into the state, 2H × D */
	po: Float32Array;
	g: Float32Array;
	f1: Float32Array;
	f2: Float32Array;
	/** channels */
	H: number;
	t0: number;
	params: number;
}

export interface PropExported {
	meta: {
		vertices: number;
		name_pool: number;
		embed_rows: number;
		params: number;
		cfg: {
			kind: string;
			emb: number;
			D: number;
			r: number;
			layers: number;
			nopos: boolean;
			tri: string;
			pheads: number;
			t0: number;
			learn_t0: boolean;
			rowcap: boolean;
		};
		near_offsets: number[];
	};
	weights: { name: string; shape: number[]; data: number[] }[];
}

/** the propagator's weights, for questions written with `spec` (any size up to the pool of 256 names) */
export function loadPropWeights(x: PropExported, spec: Spec): PropWeights {
	const { meta } = x;
	const c = meta.cfg;
	if (c.kind !== 'pairn' || c.tri !== 'prop' || !c.rowcap || c.learn_t0 || !c.nopos)
		throw new Error('weights: not the row-capped propagator');
	if (spec.V > meta.name_pool) throw new Error(`weights: names from ${meta.name_pool} only`);
	const byName = new Map<string, Float32Array>();
	let params = 0;
	for (const a of x.weights) {
		const size = a.shape.reduce((p, q) => p * q, 1);
		if (a.data.length !== size)
			throw new Error(`weights: ${a.name} has ${a.data.length} numbers, shape ${a.shape}`);
		byName.set(a.name, Float32Array.from(a.data));
		params += size;
	}
	const get = (k: string) => {
		const v = byName.get(k);
		if (!v) throw new Error(`weights: missing ${k}`);
		return v;
	};
	const tok = get("['tok']");
	if (tok.length !== meta.embed_rows * c.emb) throw new Error('weights: the embedding table');
	const blk = (k: string) => get(`['blocks'][0]['${k}']`);
	return {
		spec,
		D: c.D,
		E: c.emb,
		r: c.r,
		near: meta.near_offsets,
		tok,
		pos: null,
		U: get("['U']"),
		V: get("['V']"),
		C: get("['C']"),
		A: get("['A']"),
		Bm: get("['Bm']"),
		c: get("['c']"),
		w: get("['w']"),
		pool: meta.name_pool,
		pg: blk('pg'),
		pgb: blk('pgb'),
		ps: blk('ps'),
		psb: blk('psb'),
		po: blk('po'),
		g: blk('g'),
		f1: blk('f1'),
		f2: blk('f2'),
		H: c.pheads,
		t0: c.t0,
		params
	};
}

export interface PropRun {
	/** the names present, ascending: the order of every row below */
	names: number[];
	current: number;
	/** the read-out over the names, w · norm(z[current][v]) */
	logits: Float32Array;
	answer: number;
	/** per channel, the current town's row of X = S (I - G)^-1 */
	X: Float64Array[];
	/** per channel, the current town's row of the source S */
	S: Float64Array[];
	/** per channel, the gates after the row cap, m × m (row u, column v: the weight of one step u → v of a walk) */
	G: Float32Array[];
	/** the channel that carries the answer: the largest margin of log X at the answer over the other children */
	working: number;
	ms: number;
}

/** solve A y = b in place (A is n × n, row-major, overwritten by its LU factors; partial pivoting) */
function solveInPlace(A: Float64Array, b: Float64Array, n: number) {
	for (let k = 0; k < n; k++) {
		let p = k;
		let best = Math.abs(A[k * n + k]);
		for (let i = k + 1; i < n; i++) {
			const v = Math.abs(A[i * n + k]);
			if (v > best) {
				best = v;
				p = i;
			}
		}
		if (p !== k) {
			for (let j = 0; j < n; j++) {
				const t = A[k * n + j];
				A[k * n + j] = A[p * n + j];
				A[p * n + j] = t;
			}
			const t = b[k];
			b[k] = b[p];
			b[p] = t;
		}
		const piv = A[k * n + k];
		for (let i = k + 1; i < n; i++) {
			const f = A[i * n + k] / piv;
			if (f === 0) continue;
			A[i * n + k] = f;
			const ri = i * n;
			const rk = k * n;
			for (let j = k + 1; j < n; j++) A[ri + j] -= f * A[rk + j];
			b[i] -= f * b[k];
		}
	}
	for (let i = n - 1; i >= 0; i--) {
		let s = b[i];
		for (let j = i + 1; j < n; j++) s -= A[i * n + j] * b[j];
		b[i] = s / A[i * n + i];
	}
}

/** One forward pass of the propagator. */
export function runProp(W: PropWeights, tokens: readonly number[]): PropRun {
	const t0ms = performance.now();
	const { D, H, spec, pool, t0 } = W;
	const V = spec.V;
	const { names, at, z } = begin(W, tokens, (t) => (t > V ? pool + (t - V) : t));
	const m = names.length;
	const P = m * m;
	const current = tokens[spec.n - 1];
	const ci = at[current];
	if (ci < 0) throw new Error('run: the last token is not a vertex');

	// the normalised state, and every channel's gates (row-capped) and the current row of its source
	const t = new Float64Array(P * D);
	for (let p = 0; p < P; p++) normInto(z, p * D, D, t, p * D);
	const G = Array.from({ length: H }, () => new Float32Array(P));
	const Gd = Array.from({ length: H }, () => new Float64Array(P));
	const S = Array.from({ length: H }, () => new Float64Array(m));
	for (let p = 0; p < P; p++) {
		for (let h = 0; h < H; h++) {
			let a = W.pgb[h];
			for (let c = 0; c < D; c++) a += t[p * D + c] * W.pg[c * H + h];
			Gd[h][p] = t0 * sigmoid(a);
		}
	}
	for (let v = 0; v < m; v++) {
		const p = ci * m + v;
		for (let h = 0; h < H; h++) {
			let a = W.psb[h];
			for (let c = 0; c < D; c++) a += t[p * D + c] * W.ps[c * H + h];
			S[h][v] = sigmoid(a);
		}
	}
	const X: Float64Array[] = [];
	const A = new Float64Array(P);
	for (let h = 0; h < H; h++) {
		const g = Gd[h];
		for (let u = 0; u < m; u++) {
			let s = 0;
			for (let v = 0; v < m; v++) s += g[u * m + v];
			const k = t0 / Math.max(s, t0);
			for (let v = 0; v < m; v++) g[u * m + v] *= k;
		}
		G[h].set(g);
		// x (I - G) = s, as (I - G)ᵀ xᵀ = sᵀ
		for (let i = 0; i < m; i++)
			for (let j = 0; j < m; j++) A[i * m + j] = (i === j ? 1 : 0) - g[j * m + i];
		const x = Float64Array.from(S[h]);
		solveInPlace(A, x, m);
		for (let v = 0; v < m; v++) x[v] = Math.max(0, x[v]);
		X.push(x);
	}

	// the current row of the state: the walks' features, gated, then the per-pair MLP, then the read-out
	const F = 2 * H;
	const Hm = 2 * D;
	const f = new Float64Array(F);
	const zr = new Float64Array(D);
	const nrm = new Float64Array(D);
	const acc = new Float64Array(Hm);
	const acc2 = new Float64Array(D);
	const logits = new Float32Array(m);
	for (let v = 0; v < m; v++) {
		const p = ci * m + v;
		for (let h = 0; h < H; h++) {
			f[h] = Math.log(X[h][v] + 1e-20);
			f[H + h] = -Math.expm1(-X[h][v]);
		}
		for (let d = 0; d < D; d++) {
			let gate = 0;
			for (let c = 0; c < D; c++) gate += t[p * D + c] * W.g[c * D + d];
			let mix = 0;
			for (let c = 0; c < F; c++) mix += f[c] * W.po[c * D + d];
			zr[d] = z[p * D + d] + sigmoid(gate) * mix;
		}
		normInto(zr, 0, D, nrm, 0);
		acc.fill(0);
		for (let c = 0; c < D; c++) {
			const x = nrm[c];
			for (let k = 0; k < Hm; k++) acc[k] += x * W.f1[c * Hm + k];
		}
		acc2.fill(0);
		for (let k = 0; k < Hm; k++) {
			const x = gelu(acc[k]);
			for (let d = 0; d < D; d++) acc2[d] += x * W.f2[k * D + d];
		}
		for (let d = 0; d < D; d++) zr[d] += acc2[d];
		normInto(zr, 0, D, nrm, 0);
		let s = 0;
		for (let d = 0; d < D; d++) s += nrm[d] * W.w[d];
		logits[v] = s;
	}
	let best = 0;
	for (let k = 1; k < m; k++) if (logits[k] > logits[best]) best = k;
	const answer = names[best];

	// the working channel: the one whose walks most favour the answer over the current town's other roads
	const kids = new Set<number>();
	for (let i = 0; i + 2 < spec.n; i++)
		if (tokens[i] === spec.edge && tokens[i + 1] === current) kids.add(tokens[i + 2]);
	kids.delete(answer);
	let working = 0;
	let wm = -Infinity;
	for (let h = 0; h < H; h++) {
		const lx = (v: number) => Math.log(X[h][at[v]] + 1e-20);
		let other = -Infinity;
		for (const k of kids) if (at[k] >= 0) other = Math.max(other, lx(k));
		const margin = kids.size ? lx(answer) - other : lx(answer);
		if (margin > wm) {
			wm = margin;
			working = h;
		}
	}
	return { names, current, logits, answer, X, S, G, working, ms: performance.now() - t0ms };
}

/**
 * The walk sum one length at a time: x_M = Σ_{m ≤ M} s Gᵐ for M = 0 … Mmax, the partial sums whose limit is the solve's
 * x = s (I - G)^-1. s is a row over the m names, G an m × m matrix (row u: the steps from u).
 */
export function partialSums(s: ArrayLike<number>, G: ArrayLike<number>, m: number, Mmax: number) {
	const out: Float64Array[] = [];
	let y = Float64Array.from(s);
	const x = Float64Array.from(s);
	out.push(x.slice());
	for (let M = 1; M <= Mmax; M++) {
		const ny = new Float64Array(m);
		for (let u = 0; u < m; u++) {
			const a = y[u];
			if (a === 0) continue;
			const r = u * m;
			for (let v = 0; v < m; v++) ny[v] += a * G[r + v];
		}
		y = ny;
		for (let v = 0; v < m; v++) x[v] += y[v];
		out.push(x.slice());
	}
	return out;
}
