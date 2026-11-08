/**
 * The demo opens on the propagator's run on the 250-town question, computed here at build time by the same forward
 * pass the page runs live (hero.server.ts), so the map is lit before the weights arrive. The equations the page shows
 * are the paper's own, read from paper/paper.md and set by KaTeX here, so the page ships no maths code.
 */
import { readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import katex from 'katex';
import { heroData } from '#lib/search/hero.server.ts';
import type { PageServerLoad } from './$types';

export const prerender = true;

/** the display equation labelled {#id} in paper.md, set by KaTeX */
function equation(md: string, id: string): string {
	const m = new RegExp(String.raw`^\$\$(.*)\$\$\s*\{#${id}\}\s*$`, 'm').exec(md);
	if (!m) throw new Error(`paper.md: no equation {#${id}}`);
	return katex.renderToString(m[1].trim(), { displayMode: true, throwOnError: true });
}

export const load: PageServerLoad = () => {
	const root = process.cwd();
	const md = readFileSync(resolvePath(root, 'paper/paper.md'), 'utf8');
	const h = heroData(root);
	return {
		opening: { run: h.run, lookahead: h.lookahead },
		layout250: h.layout,
		eq: { prop: equation(md, 'eq-prop'), pair: equation(md, 'eq-pair'), or: equation(md, 'eq-or') }
	};
};
