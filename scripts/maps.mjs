/*
 * Every question's map, in the browser: the demo opens each of the 300 exported questions in turn (/demo?q=1 to
 * ?q=300) and checks that its map draws one town for every vertex and one road for every edge of the question, with a
 * number for every town at this width and no error in the console. It drives a running dev or preview server.
 * SIZE=125 checks the 125-town questions (/demo?size=125&q=1 to ?q=37) instead, and that the repeated layer answers
 * each one with a road toward its goal.
 *
 *   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs node scripts/maps.mjs [base url]
 *   SIZE=125 PLAYWRIGHT=... node scripts/maps.mjs [base url]
 */
import { readFileSync } from 'node:fs';

const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright');
const BASE = process.argv[2] ?? 'http://127.0.0.1:5194';
const SIZE = Number(process.env.SIZE ?? 49);
const COUNT = JSON.parse(
	readFileSync(new URL(`../static/model/graphs_v${SIZE}.json`, import.meta.url), 'utf8')
).questions.length;

const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
const bad = [];
const [from, to] = (process.env.RANGE ?? `1-${COUNT}`).split('-').map(Number);
let answered = 0;
const t0 = Date.now();
for (let q = from; q <= to; q++) {
	if ((q - from) % 25 === 0)
		console.log(`question ${q}, ${((Date.now() - t0) / 1000).toFixed(0)} s`);
	await page.goto(`${BASE}/demo?q=${q}${SIZE === 49 ? '' : `&size=${SIZE}`}`, {
		waitUntil: 'domcontentloaded'
	});
	await page.waitForFunction(
		(n) => {
			const t = document.querySelector('.stage-map .facts')?.textContent ?? '';
			return t.includes(`test question ${n} of`) || t.includes(`graph ${n} of`);
		},
		q,
		{ timeout: 15000 }
	);
	const text = await page.locator('.stage-map .facts span:nth-child(2)').innerText();
	const towns = Number(/(\d+) towns/.exec(text)?.[1]);
	const roads = Number(/(\d+) roads/.exec(text)?.[1]);
	const ok = await page
		.waitForFunction(
			(n) =>
				document.querySelectorAll('.stage-map g.town').length === n &&
				document.querySelectorAll('.stage-map text.num').length >= n - 1,
			towns,
			{ timeout: 8000 }
		)
		.then(() => true)
		.catch(() => false);
	const drawn = await page.evaluate(() => ({
		towns: document.querySelectorAll('.stage-map g.town').length,
		roads: document.querySelectorAll('.stage-map .roads path.road').length,
		numbers: document.querySelectorAll('.stage-map text.num').length
	}));
	if (!ok || drawn.roads !== roads) bad.push({ q, towns, roads, drawn });
	if (SIZE !== 49) {
		// the repeated layer's first answer: a road that leads to the goal
		const said = await page
			.waitForFunction(
				() => {
					const v = document.querySelector('.stage-map .verdict')?.textContent ?? '';
					return /The model answers/.test(v) ? v : null;
				},
				null,
				{ timeout: 20000 }
			)
			.then((h) => h.jsonValue())
			.catch(() => '');
		if (said.includes('a road that leads to the goal')) answered++;
		else bad.push({ q, said });
	}
}
console.log(
	`maps checked: ${to - from + 1}; mismatches: ${bad.length}; console errors: ${errors.length}` +
		(SIZE === 49 ? '' : `; answered with a road toward the goal: ${answered}`)
);
if (bad.length) console.log(JSON.stringify(bad.slice(0, 10)));
if (errors.length) console.log(errors.slice(0, 10).join('\n'));
await browser.close();
process.exit(bad.length || errors.length ? 1 : 0);
