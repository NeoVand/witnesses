/*
 * Screenshots of the site for review, into docs/, and the social card, static/media/card.png: the project page, the
 * paper and the demo, at 1440 × 900 and at phone width (375 × 812). It drives a running dev or preview server.
 * ONLY=home,paper,prop,depth,card takes only those.
 *
 *   PLAYWRIGHT=/path/to/node_modules/playwright/index.mjs node scripts/shots.mjs [base url]
 *
 * Playwright is not a dependency of this repo; point PLAYWRIGHT at any copy. The base URL includes the base path of a
 * production preview (http://127.0.0.1:4173/witnesses).
 */
import { mkdirSync, readFileSync } from 'node:fs';

const { chromium } = await import(process.env.PLAYWRIGHT ?? 'playwright');
const BASE = process.argv[2] ?? 'http://127.0.0.1:5194';
const ONLY = (process.env.ONLY ?? 'home,paper,tutorial,prop,depth,card').split(',');
const OUT = new URL('../docs/', import.meta.url).pathname;
const CARD = new URL('../static/media/card.png', import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const sizes = [
	{ name: 'desktop', viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
	{
		name: 'phone',
		viewport: { width: 375, height: 812 },
		deviceScaleFactor: 2,
		isMobile: true,
		hasTouch: true
	}
];
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
let errors = 0;

for (const { name, ...ctx } of sizes) {
	const context = await browser.newContext(ctx);
	const page = await context.newPage();
	page.on('pageerror', (e) => (errors++, console.error(`[${name}] page error:`, e.message)));
	page.on('console', (m) => {
		if (m.type() !== 'error') return;
		errors++;
		console.error(`[${name}] console:`, m.text());
	});
	if (ONLY.includes('home')) await home(page, name);
	if (ONLY.includes('paper')) await paper(page, name);
	if (ONLY.includes('tutorial')) await tutorial(page, name);
	if (ONLY.includes('prop')) await prop(page, name);
	if (ONLY.includes('depth')) await depth(page, name);
	await context.close();
}
if (ONLY.includes('card')) await card();
await browser.close();
console.log(`screenshots in ${OUT}; ${errors} console or page errors`);

/** the project page: the map spreading, then its last frame (the chosen road lit), and the whole page */
async function home(page, name) {
	await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
	await wait(3200);
	await page.screenshot({ path: `${OUT}home-${name}-walks.png` });
	await wait(5200);
	await page.screenshot({ path: `${OUT}home-${name}.png` });
	await page.evaluate(() =>
		scrollTo(0, document.querySelector('.found')?.getBoundingClientRect().top ?? 0)
	);
	await wait(1500);
	await page.screenshot({ path: `${OUT}home-${name}-found.png` });
	await page.screenshot({ path: `${OUT}home-${name}-full.png`, fullPage: true });
}

async function paper(page, name) {
	await page.goto(`${BASE}/paper`, { waitUntil: 'networkidle' });
	await page.screenshot({ path: `${OUT}paper-${name}.png` });
	for (const id of ['thm-size', 'fig-sums', 'fig-search', 'tbl-zoo', 'stbl-search']) {
		await page.evaluate((i) => {
			document.getElementById(i)?.scrollIntoView({ block: 'start' });
			scrollBy(0, -60);
		}, id);
		await wait(700);
		await page.screenshot({ path: `${OUT}paper-${name}-${id}.png` });
	}
}

/** the tutorial: its first screen with the contents, a figure, an aside and the glossary */
async function tutorial(page, name) {
	await page.goto(`${BASE}/tutorial`, { waitUntil: 'networkidle' });
	await page.screenshot({ path: `${OUT}tutorial-${name}.png` });
	for (const [sel, tag] of [
		['figure.fig:nth-of-type(1)', 'figure'],
		['.aside', 'aside'],
		['#app-glossary', 'glossary']
	]) {
		await page.evaluate((q) => {
			document.querySelector(q)?.scrollIntoView({ block: 'start' });
			scrollBy(0, -60);
		}, sel);
		await wait(700);
		await page.screenshot({ path: `${OUT}tutorial-${name}-${tag}.png` });
	}
}

/** the propagator: the walks spreading on the 250-town map, the answer, a new goal, and a 49-town question */
async function prop(page, name) {
	await page.goto(`${BASE}/demo`, { waitUntil: 'networkidle' });
	if (name === 'phone')
		await page.evaluate(() =>
			document.querySelector('.part .stage')?.scrollIntoView({ block: 'start' })
		);
	await wait(2600);
	await page.screenshot({ path: `${OUT}demo-${name}-walks.png` });
	await page.locator('.part .verdict:has-text("The layer answers")').waitFor({ timeout: 30000 });
	await wait(400);
	await page.screenshot({ path: `${OUT}demo-${name}-answer.png` });
	// a new goal: the town twelve roads down another branch of the start; the walks start from there
	const q = JSON.parse(
		readFileSync(new URL('../static/model/graphs_v250.json', import.meta.url), 'utf8')
	).questions[0];
	const next = new Map(q.edges.map(([u, v]) => [u, v]));
	let goal = q.edges.find(([u, v]) => u === q.start && v !== q.answer)[1];
	for (let i = 1; i < 12 && next.has(goal); i++) goal = next.get(goal);
	await page.locator(`.part g.town[aria-label="town ${goal}: make it the goal"]`).click();
	await page.locator('.part .verdict:has-text("The layer answers")').waitFor({ timeout: 30000 });
	await wait(400);
	await page.screenshot({ path: `${OUT}demo-${name}-goal.png` });
	// a road hovered: its gates
	if (name === 'desktop') {
		await page.locator('.part .hits path').nth(12).hover({ force: true });
		await wait(300);
		await page.screenshot({ path: `${OUT}demo-${name}-gates.png` });
	}
	await page.goto(`${BASE}/demo?towns=49&question=63`, { waitUntil: 'networkidle' });
	if (name === 'phone')
		await page.evaluate(() =>
			document.querySelector('.part .stage')?.scrollIntoView({ block: 'start' })
		);
	await page.locator('.part .verdict:has-text("The layer answers")').waitFor({ timeout: 30000 });
	await wait(400);
	await page.screenshot({ path: `${OUT}demo-${name}-49.png` });
	await page.screenshot({ path: `${OUT}demo-${name}-full.png`, fullPage: true });
}

/** layer by layer: the 8 layers on the opening question, then the OR join on 125 towns */
async function depth(page, name) {
	await page.goto(`${BASE}/demo`, { waitUntil: 'networkidle' });
	await page.evaluate(() =>
		document.querySelector('.part2 .stage')?.scrollIntoView({ block: 'start' })
	);
	await wait(1500);
	await page.screenshot({ path: `${OUT}depth-${name}-sweep.png` });
	await page.locator('.part2 .verdict:has-text("Reached the goal")').waitFor({ timeout: 60000 });
	await wait(900);
	await page.screenshot({ path: `${OUT}depth-${name}-walked.png` });
	const { questions } = JSON.parse(
		readFileSync(new URL('../static/model/graphs_v125.json', import.meta.url), 'utf8')
	);
	const Q = questions.findIndex((x) => x.lookahead === 62);
	await page.goto(`${BASE}/demo?size=125&q=${Q + 1}`, { waitUntil: 'networkidle' });
	await page
		.locator(`.part2 .facts:has-text("test question ${Q + 1} of")`)
		.waitFor({ timeout: 20000 });
	await page.evaluate(() =>
		document.querySelector('.part2 .stage')?.scrollIntoView({ block: 'start' })
	);
	await page.locator('.part2 .verdict:has-text("The model answers")').waitFor({ timeout: 60000 });
	await wait(300);
	await page.screenshot({ path: `${OUT}depth-${name}-orjoin.png` });
}

/** the social card: the project page's map, its last frame, on white, 1200 × 630 */
async function card() {
	const context = await browser.newContext({
		viewport: { width: 1440, height: 900 },
		deviceScaleFactor: 2,
		reducedMotion: 'reduce'
	});
	const page = await context.newPage();
	await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
	await wait(800);
	const shot = await page.locator('.hero .frame').screenshot({ omitBackground: false });
	const img = `data:image/png;base64,${shot.toString('base64')}`;
	const sheet = await browser.newPage({
		viewport: { width: 1200, height: 630 },
		deviceScaleFactor: 1
	});
	await sheet.setContent(
		`<html><body style="margin:0;background:#fff;display:grid;place-items:center;width:1200px;height:630px">` +
			`<img src="${img}" style="max-width:1140px;max-height:590px"></body></html>`
	);
	await sheet.screenshot({ path: CARD });
	await sheet.close();
	await context.close();
	console.log(`card in ${CARD}`);
}
