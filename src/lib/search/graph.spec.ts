import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { children, goodSteps, lookahead } from './graph.ts';
import { place } from './layout.ts';
import { current, parse, spec, vertices } from './tokens.ts';

const S = spec(49);
const { questions } = JSON.parse(readFileSync('static/model/graphs_v49.json', 'utf8')) as {
	questions: { lookahead: number | null; tokens: number[]; answer: number }[];
};

describe('the graph facts the page states', () => {
	it('computes the benchmark’s lookahead as its own code does', () => {
		for (const q of questions) {
			if (q.lookahead === null) continue;
			const p = parse(q.tokens, S);
			expect(lookahead(children(p), current(p), q.answer, p.goal)).toBe(q.lookahead);
		}
	});

	it('finds the exported answer among the steps that reach the goal', () => {
		for (const q of questions) expect(goodSteps(parse(q.tokens, S))).toContain(q.answer);
	});
});

describe('the layered drawing', () => {
	it('places every vertex once, with every edge pointing to a later rank', () => {
		for (const q of questions) {
			const p = parse(q.tokens, S);
			const out = place(vertices(p), p.edges, p.start);
			expect(out.at.size).toBe(vertices(p).length);
			expect(new Set(out.order).size).toBe(out.order.length);
			for (const [u, v] of p.edges) expect(out.at.get(u)!.rank).toBeLessThan(out.at.get(v)!.rank);
			for (const a of out.at.values()) expect(Number.isFinite(a.slot)).toBe(true);
		}
	});
});
