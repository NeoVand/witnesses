/**
 * The recording the demo opens on, so it never shows an empty panel while the weights load: one test question and
 * the walk from its start to its goal, one forward pass per step, every map kept (src/lib/search/opening.json).
 * It runs the same forward pass as the page (src/lib/search/model.ts).
 *
 *   node scripts/record.ts [question index]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { loadWeights, run, type Exported } from '../src/lib/search/model.ts';
import { current, encode, parse, spec } from '../src/lib/search/tokens.ts';
import { fromRun, pack } from '../src/lib/search/trace.ts';

const S = spec(49);
const W = loadWeights(
	JSON.parse(readFileSync('static/model/pairn_v49.json', 'utf8')) as Exported,
	S
);
const { questions } = JSON.parse(readFileSync('static/model/graphs_v49.json', 'utf8')) as {
	questions: { lookahead: number | null; tokens: number[]; answer: number }[];
};
const index = Number(process.argv[2] ?? 62);
const q = questions[index];
const p = parse(q.tokens, S);
const steps = [];
for (;;) {
	const tokens = encode(p, S);
	if (!tokens) break;
	const r = run(W, tokens);
	steps.push(pack(fromRun(r, tokens)));
	if (r.answer === p.goal || !p.edges.some(([u, v]) => u === current(p) && v === r.answer)) break;
	p.path.push(r.answer);
}
const out = { format: 'two-things-recording-1', question: index, lookahead: q.lookahead, steps };
writeFileSync('src/lib/search/opening.json', JSON.stringify(out) + '\n');
console.log(
	`question ${index} (lookahead ${q.lookahead}): ${steps.length} steps, ${JSON.stringify(out).length} bytes`
);
