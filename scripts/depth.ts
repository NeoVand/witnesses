/**
 * The depth table the demo draws: for every lookahead, the share of the exported test questions whose answer the
 * read-out already gets right after each layer (the state after the start counts as layer 0). It runs the same
 * forward pass as the page (src/lib/search/model.ts) on static/model/graphs_v49.json.
 *
 *   node scripts/depth.ts      # writes src/lib/search/depth.json
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { loadWeights, run, type Exported } from '../src/lib/search/model.ts';
import { spec } from '../src/lib/search/tokens.ts';

const S = spec(49);
const W = loadWeights(
	JSON.parse(readFileSync('static/model/pairn_v49.json', 'utf8')) as Exported,
	S
);
const { questions } = JSON.parse(readFileSync('static/model/graphs_v49.json', 'utf8')) as {
	questions: { lookahead: number | null; tokens: number[]; answer: number }[];
};
const rows = new Map<number, { n: number; right: number[] }>();
for (const q of questions) {
	if (q.lookahead === null) continue;
	const out = run(W, q.tokens);
	const row = rows.get(q.lookahead) ?? { n: 0, right: Array(W.L + 1).fill(0) };
	row.n++;
	out.readouts.forEach((r, l) => {
		let b = 0;
		for (let k = 1; k < r.length; k++) if (r[k] > r[b]) b = k;
		if (out.names[b] === q.answer) row.right[l]++;
	});
	rows.set(q.lookahead, row);
}
const lookaheads = [...rows.keys()].sort((a, b) => a - b);
const table = {
	note: 'share of the exported test questions answered right by the read-out after each layer, by lookahead',
	layers: W.L,
	lookaheads,
	questions: lookaheads.map((L) => rows.get(L)!.n),
	right: lookaheads.map((L) => rows.get(L)!.right)
};
writeFileSync('src/lib/search/depth.json', JSON.stringify(table) + '\n');
for (const L of lookaheads) console.log(String(L).padStart(2), rows.get(L)!.right.join(' '));
