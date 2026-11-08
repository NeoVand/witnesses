import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { loadWeights, run, type Exported } from './model.ts';
import { parse, spec, encode } from './tokens.ts';
import { goodSteps } from './graph.ts';

const S = spec(49);
const W = loadWeights(
	JSON.parse(readFileSync('static/model/pairn_v49.json', 'utf8')) as Exported,
	S
);

interface Reference {
	questions: number[];
	final_logits: number[][];
	layer_readouts: number[][][];
	layer_maps: number[][][][];
	present: number[][];
}
interface Question {
	lookahead: number | null;
	tokens: number[];
	answer: number;
	edges: [number, number][];
	start: number;
	goal: number;
	path: number[];
}
const ref = JSON.parse(readFileSync('test-data/reference_v49.json', 'utf8')) as Reference;
const graphs = JSON.parse(readFileSync('static/model/graphs_v49.json', 'utf8')) as {
	questions: Question[];
};

describe('the forward pass in the browser', () => {
	it('matches the training code’s logits, read-outs and maps (max difference below 1e-3)', () => {
		let worst = 0;
		ref.questions.forEach((qi, b) => {
			const q = graphs.questions[qi];
			const out = run(W, q.tokens);
			const present = ref.present[b].flatMap((p, v) => (p > 0 ? [v] : []));
			expect(out.names).toEqual(present);
			const m = present.length;
			present.forEach((v, k) => {
				worst = Math.max(worst, Math.abs(out.readouts[W.L][k] - ref.final_logits[b][v]));
				for (let l = 0; l <= W.L; l++) {
					worst = Math.max(worst, Math.abs(out.readouts[l][k] - ref.layer_readouts[l][b][v]));
					present.forEach((u, i) => {
						worst = Math.max(worst, Math.abs(out.maps[l][i * m + k] - ref.layer_maps[l][b][u][v]));
					});
				}
			});
			expect(out.answer).toBe(q.answer);
		});
		expect(worst).toBeLessThan(1e-3);
	});

	it('answers every exported question as the export says, and every answer is a step toward the goal', () => {
		let right = 0;
		for (const q of graphs.questions) {
			const out = run(W, q.tokens);
			const p = parse(q.tokens, S);
			if (out.answer === q.answer && goodSteps(p).includes(out.answer)) right++;
		}
		expect(right).toBe(graphs.questions.length);
	}, 120_000);

	it('reads the benchmark format back and writes it again unchanged', () => {
		for (const q of graphs.questions.slice(0, 40)) {
			const p = parse(q.tokens, S);
			expect(p.start).toBe(q.start);
			expect(p.goal).toBe(q.goal);
			expect(p.path).toEqual(q.path);
			expect(p.edges).toEqual(q.edges);
			expect(encode(p, S)).toEqual(q.tokens);
		}
	});
});
