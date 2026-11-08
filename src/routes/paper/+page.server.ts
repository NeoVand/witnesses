/**
 * The paper page, rendered at build time from paper/paper.md and paper/refs.bib (the same source as the arXiv
 * LaTeX, paper/build-tex.ts). The page is prerendered, so the parser and KaTeX never reach the browser.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import { asset } from '$app/paths';
import { building } from '$app/env';
import { parseBib } from '#lib/paper/bib.ts';
import { renderPaper } from '#lib/paper/html.ts';
import type { PageServerLoad } from './$types';

export const prerender = true;
/** the paper needs no client code: no hydration, so the rendered HTML is not shipped twice */
export const csr = false;

export const load: PageServerLoad = () => {
	const root = process.cwd();
	const md = readFileSync(resolvePath(root, 'paper/paper.md'), 'utf8');
	const bib = parseBib(readFileSync(resolvePath(root, 'paper/refs.bib'), 'utf8'));
	const out = renderPaper(
		md,
		bib,
		(f) => asset(`/figures/${f}` as Parameters<typeof asset>[0]),
		(f) => existsSync(resolvePath(root, 'paper/figures/out', f))
	);
	// the build refuses a paper with an unresolved reference; the dev server shows it and says what is missing
	const unresolved = out.missing.filter((m) => !m.startsWith('figure:'));
	if (unresolved.length) {
		if (building) throw new Error(`paper.md: unresolved ${unresolved.join(', ')}`);
		console.warn(`paper.md: unresolved ${unresolved.join(', ')}`);
	}
	return out;
};
