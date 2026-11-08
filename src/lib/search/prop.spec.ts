import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadPropWeights, partialSums, runProp, type PropExported } from './model.ts';
import { parse, spec } from './tokens.ts';
import { goodSteps } from './graph.ts';

const X = JSON.parse(readFileSync('static/model/prop_v49.json', 'utf8')) as PropExported;
const SIZES = [49, 125, 250] as const;
const W = Object.fromEntries(SIZES.map((V) => [V, loadPropWeights(X, spec(V))]));

interface Case {
	V: number;
	question: number;
	tokens: number[];
	names: number[];
	current: number;
	answer: number;
	logits: number[];
	/** per channel, the current row of X and of S, over `names` */
	X: number[][];
	S: number[][];
	working: number;
	gates?: { edge: [number, number]; back: number[]; along: number[] }[];
}
const ref = JSON.parse(readFileSync('test-data/reference_prop.json', 'utf8')) as {
	cases: Case[];
	answers: Record<string, number[]>;
};
const questions = (V: number) =>
	(
		JSON.parse(readFileSync(`static/model/graphs_v${V}.json`, 'utf8')) as {
			questions: { tokens: number[]; answer: number }[];
		}
	).questions;

/** a difference relative to the value's size, for walk weights that span twenty orders of magnitude */
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1e-3, Math.abs(b));

describe('the propagator in the browser', () => {
	it('holds the trained run’s 28,472 parameters: one layer, four channels, t0 = 0.99, rows capped', () => {
		expect(W[49].params).toBe(28_472);
		expect(W[49].H).toBe(4);
		expect(W[49].t0).toBe(0.99);
		expect(W[49].pool).toBe(256);
	});

	it('matches the training code’s logits, walk sums and sources at 49, 125 and 250 vertices (below 1e-3)', () => {
		for (const V of SIZES) expect(ref.cases.some((c) => c.V === V)).toBe(true);
		let logit = 0;
		let walks = 0;
		let source = 0;
		for (const c of ref.cases) {
			const out = runProp(W[c.V], c.tokens);
			expect(out.names).toEqual(c.names);
			expect(out.current).toBe(c.current);
			expect(out.answer).toBe(c.answer);
			expect(out.working).toBe(c.working);
			c.names.forEach((_, j) => {
				logit = Math.max(logit, Math.abs(out.logits[j] - c.logits[j]));
				for (let h = 0; h < 4; h++) {
					walks = Math.max(walks, rel(out.X[h][j], c.X[h][j]));
					source = Math.max(source, Math.abs(out.S[h][j] - c.S[h][j]));
				}
			});
		}
		console.log(
			`propagator parity: logits ${logit.toExponential(2)}, X (relative) ${walks.toExponential(2)}, S ${source.toExponential(2)}`
		);
		expect(logit).toBeLessThan(1e-3);
		expect(walks).toBeLessThan(1e-3);
		expect(source).toBeLessThan(1e-3);
	});

	it('matches the training code’s gates on every road of the 250-vertex question, both ways', () => {
		const c = ref.cases.find((x) => x.V === 250 && x.gates)!;
		const out = runProp(W[250], c.tokens);
		const m = out.names.length;
		const at = new Map(out.names.map((v, i) => [v, i]));
		let worst = 0;
		for (const { edge, back, along } of c.gates!) {
			const [u, v] = edge.map((x) => at.get(x)!);
			for (let h = 0; h < 4; h++) {
				worst = Math.max(worst, Math.abs(out.G[h][v * m + u] - back[h]));
				worst = Math.max(worst, Math.abs(out.G[h][u * m + v] - along[h]));
			}
		}
		expect(worst).toBeLessThan(1e-3);
	});

	it('sums the walks one length at a time to the solve’s answer', () => {
		const c = ref.cases.find((x) => x.V === 250)!;
		const out = runProp(W[250], c.tokens);
		const h = out.working;
		const m = out.names.length;
		const sums = partialSums(out.S[h], out.G[h], m, 1500);
		const last = sums[sums.length - 1];
		let worst = 0;
		for (let v = 0; v < m; v++) worst = Math.max(worst, rel(last[v], out.X[h][v]));
		expect(worst).toBeLessThan(1e-3);
	});

	it('answers every exported question as the training code does (300 at 49, 37 at 125, 1 at 250)', () => {
		for (const V of SIZES) {
			const qs = questions(V);
			const said = ref.answers[String(V)];
			expect(said.length).toBe(qs.length);
			let same = 0;
			let toward = 0;
			for (const [i, q] of qs.entries()) {
				const out = runProp(W[V], q.tokens);
				if (out.answer === said[i]) same++;
				if (goodSteps(parse(q.tokens, spec(V))).includes(out.answer)) toward++;
			}
			expect(same).toBe(qs.length);
			expect(toward).toBe(qs.length);
		}
	}, 300_000);
});
