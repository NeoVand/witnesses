/*
 * Copies the paper in from where it is written: paper.md and refs.bib into paper/, the figures paper.md uses (the SVG
 * or PNG the page shows and the PDF LaTeX includes) into paper/figures/out/, and the author's PDF to
 * static/attention-witnesses.pdf. paper/figures/out/ then holds exactly the figures the paper uses.
 *
 *   PAPER_SRC=/path/to/the/paper node scripts/sync-paper.mjs
 *
 * PAPER_SRC holds paper.md, refs.bib, figures/out/ and the PDF (attention-witnesses*.pdf, the newest if several).
 * Nothing here edits the paper: the renderer (src/lib/paper/) is fixed instead.
 */
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	statSync,
	unlinkSync
} from 'node:fs';
import { join } from 'node:path';

const SRC = process.env.PAPER_SRC;
if (!SRC || !existsSync(join(SRC, 'paper.md'))) {
	console.error(
		'Set PAPER_SRC to the folder holding paper.md, refs.bib, figures/out/ and the PDF.'
	);
	process.exit(1);
}
const ROOT = new URL('..', import.meta.url).pathname;
const PAPER = join(ROOT, 'paper');
const OUT = join(PAPER, 'figures', 'out');
mkdirSync(OUT, { recursive: true });

copyFileSync(join(SRC, 'paper.md'), join(PAPER, 'paper.md'));
copyFileSync(join(SRC, 'refs.bib'), join(PAPER, 'refs.bib'));

// every figure the paper names: src= (the page) and pdf= (LaTeX)
const md = readFileSync(join(PAPER, 'paper.md'), 'utf8');
const used = new Set();
for (const m of md.matchAll(/^:::\s*figure\s*\{([^}]*)\}/gm))
	for (const k of m[1].matchAll(/\b(?:src|pdf)=([^\s}"]+)/g)) used.add(k[1]);
const missing = [];
for (const f of used) {
	const from = join(SRC, 'figures', 'out', f);
	if (existsSync(from)) copyFileSync(from, join(OUT, f));
	else missing.push(f);
}
let removed = 0;
for (const f of readdirSync(OUT))
	if (!used.has(f)) {
		unlinkSync(join(OUT, f));
		removed++;
	}

const pdfs = readdirSync(SRC)
	.filter((f) => /^attention-witnesses.*\.pdf$/.test(f))
	.sort((a, b) => statSync(join(SRC, b)).mtimeMs - statSync(join(SRC, a)).mtimeMs);
if (pdfs.length) copyFileSync(join(SRC, pdfs[0]), join(ROOT, 'static', 'attention-witnesses.pdf'));

console.log(
	`paper.md, refs.bib; ${used.size - missing.length} figures (${[...used].sort().join(', ')})` +
		(missing.length ? `; MISSING ${missing.join(', ')}` : '') +
		`; ${removed} old figure files removed; PDF ${pdfs[0] ?? 'NOT FOUND'} → static/attention-witnesses.pdf`
);
