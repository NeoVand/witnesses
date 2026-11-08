/*
 * Copies the tutorial in from where it is written: tutorial.md and refs.bib into tutorial/, the figures it draws itself
 * (the SVG or PNG the page shows and the PDF LaTeX includes) into tutorial/figures/out/, and its PDF to
 * static/attention-witnesses-tutorial.pdf. The paper's figures it reuses are served from paper/figures/out/ (synced
 * with the paper); one it names that neither folder holds is reported.
 *
 *   TUTORIAL_SRC=/path/to/the/tutorial node scripts/sync-tutorial.mjs
 *
 * TUTORIAL_SRC holds tutorial.md, refs.bib, figures/out/ and witnesses-tutorial.pdf. Nothing here edits the tutorial.
 */
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	unlinkSync
} from 'node:fs';
import { join } from 'node:path';

const SRC = process.env.TUTORIAL_SRC;
if (!SRC || !existsSync(join(SRC, 'tutorial.md'))) {
	console.error(
		'Set TUTORIAL_SRC to the folder holding tutorial.md, refs.bib, figures/out/ and the PDF.'
	);
	process.exit(1);
}
const ROOT = new URL('..', import.meta.url).pathname;
const DIR = join(ROOT, 'tutorial');
const OUT = join(DIR, 'figures', 'out');
const PAPER = join(ROOT, 'paper', 'figures', 'out');
mkdirSync(OUT, { recursive: true });

copyFileSync(join(SRC, 'tutorial.md'), join(DIR, 'tutorial.md'));
copyFileSync(join(SRC, 'refs.bib'), join(DIR, 'refs.bib'));

const md = readFileSync(join(DIR, 'tutorial.md'), 'utf8');
const used = new Set();
for (const m of md.matchAll(/^:::\s*figure\s*\{([^}]*)\}/gm))
	for (const k of m[1].matchAll(/\b(?:src|pdf)=([^\s}"]+)/g)) used.add(k[1]);
const own = [];
const missing = [];
for (const f of used) {
	const from = join(SRC, 'figures', 'out', f);
	if (existsSync(from)) {
		copyFileSync(from, join(OUT, f));
		own.push(f);
	} else if (!existsSync(join(PAPER, f.split('/').pop()))) missing.push(f);
}
let removed = 0;
for (const f of readdirSync(OUT))
	if (!own.includes(f)) {
		unlinkSync(join(OUT, f));
		removed++;
	}
const pdf = join(SRC, 'witnesses-tutorial.pdf');
if (existsSync(pdf)) copyFileSync(pdf, join(ROOT, 'static', 'attention-witnesses-tutorial.pdf'));

console.log(
	`tutorial.md, refs.bib; ${own.length} of its own figures, ${used.size - own.length - missing.length} from the paper` +
		(missing.length ? `; MISSING ${missing.join(', ')}` : '') +
		`; ${removed} old figure files removed; PDF ${existsSync(pdf) ? 'copied' : 'NOT FOUND'}`
);
