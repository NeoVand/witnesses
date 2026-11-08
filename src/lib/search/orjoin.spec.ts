import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadLoopWeights, runLoop, type LoopExported } from './model.ts';
import { parse, spec } from './tokens.ts';
import { goodSteps } from './graph.ts';

const X = JSON.parse(readFileSync('static/model/orjoin_v49.json', 'utf8')) as LoopExported;
const W49 = loadLoopWeights(X, spec(49));
const W125 = loadLoopWeights(X, spec(125));

interface Case {
	V: number;
	tokens: number[];
	names: number[];
	answer: number;
	/** after K = 1..10, over `names` */
	logits: number[][];
	maps?: Record<string, number[][]>;
}
const ref = JSON.parse(readFileSync('test-data/reference_orjoin.json', 'utf8')) as {
	iterations: number;
	cases: Case[];
};
interface Question {
	lookahead: number | null;
	tokens: number[];
	answer: number;
}
const load = (f: string) =>
	JSON.parse(readFileSync(f, 'utf8')) as { meta: { kind?: string }; questions: Question[] };

describe('the OR join in the browser', () => {
	it('holds the trained run’s 24,896 parameters, one block, the rate form', () => {
		expect(W49.params).toBe(24_896);
		expect(W49.block.J).toBe(1);
		expect(W49.pool).toBe(128);
	});

	it('matches the training code’s logits after every iteration, at 49 and 125 vertices (below 1e-4)', () => {
		expect(ref.cases.filter((c) => c.V === 49).length).toBeGreaterThanOrEqual(10);
		expect(ref.cases.filter((c) => c.V === 125).length).toBeGreaterThanOrEqual(10);
		let worst = 0;
		for (const c of ref.cases) {
			const out = runLoop(c.V === 49 ? W49 : W125, c.tokens, ref.iterations);
			expect(out.names).toEqual(c.names);
			for (let k = 1; k <= ref.iterations; k++)
				c.names.forEach((_, j) => {
					worst = Math.max(worst, Math.abs(out.readouts[k][j] - c.logits[k - 1][j]));
				});
			const m = c.names.length;
			for (const [K, rows] of Object.entries(c.maps ?? {}))
				rows.forEach((row, i) =>
					row.forEach((v, j) => {
						worst = Math.max(worst, Math.abs(out.maps[Number(K)][i * m + j] - v));
					})
				);
		}
		console.log(
			`OR join parity: ${ref.cases.length} questions, K = 1..10, worst ${worst.toExponential(2)}`
		);
		expect(worst).toBeLessThan(1e-4);
	}, 120_000);

	it('answers every 49-town test question with a step toward the goal after the registered K = 8', () => {
		const S = spec(49);
		const { questions } = load('static/model/graphs_v49.json');
		let right = 0;
		for (const q of questions) {
			const out = runLoop(W49, q.tokens, 8);
			if (out.answer === q.answer && goodSteps(parse(q.tokens, S)).includes(out.answer)) right++;
		}
		expect(right).toBe(questions.length);
	}, 240_000);

	it('answers every exported 125-town question after K = 9', () => {
		const S = spec(125);
		const { questions } = load('static/model/graphs_v125.json');
		let right = 0;
		for (const q of questions) {
			const out = runLoop(W125, q.tokens, 9);
			if (out.answer === q.answer && goodSteps(parse(q.tokens, S)).includes(out.answer)) right++;
		}
		expect(right).toBe(questions.length);
	}, 240_000);
});
