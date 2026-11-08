/**
 * The tutorial, a companion to the paper for students, rendered at build time from tutorial/tutorial.md and
 * tutorial/refs.bib by the paper's pipeline (src/lib/paper/): chapters, asides, figures (its own diagrams and the
 * paper's figures), the questions that close each chapter, the glossary, the answers and the reading list.
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import { asset } from '$app/paths';
import { building } from '$app/env';
import { parseBib } from '#lib/paper/bib.ts';
import { renderPaper } from '#lib/paper/html.ts';
import type { PageServerLoad } from './$types';

export const prerender = true;
/** the tutorial needs no client code: no hydration, so the rendered HTML is not shipped twice */
export const csr = false;

export const load: PageServerLoad = () => {
	const root = process.cwd();
	const md = readFileSync(resolvePath(root, 'tutorial/tutorial.md'), 'utf8');
	const bib = parseBib(readFileSync(resolvePath(root, 'tutorial/refs.bib'), 'utf8'));
	const figUrl = (f: string) => asset(`/figures/${f}` as Parameters<typeof asset>[0]);
	const exists = (f: string) =>
		existsSync(resolvePath(root, 'tutorial/figures/out', f)) ||
		existsSync(resolvePath(root, 'paper/figures/out', f));
	const out = renderPaper(md, bib, figUrl, exists, { labels: { figure: 'Figure' } });
	// the build refuses an unresolved reference; the dev server shows the page and says what is missing
	const unresolved = out.missing.filter((m) => !m.startsWith('figure:'));
	if (unresolved.length) {
		if (building) throw new Error(`tutorial.md: unresolved ${unresolved.join(', ')}`);
		console.warn(`tutorial.md: unresolved ${unresolved.join(', ')}`);
	}
	const lede = out.meta.lede
		? (/<p>([\s\S]*?)<\/p>/.exec(renderPaper(out.meta.lede, bib, figUrl, exists).html)?.[1] ?? '')
		: null;
	return { meta: out.meta, html: out.html, toc: out.toc, lede };
};
