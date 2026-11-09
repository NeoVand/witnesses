/**
 * The project page's text comes from the paper itself (paper/paper.md): its title, abstract and first figure, and the
 * four findings, each one sentence of the abstract beside its figure, rendered at build time by the same pipeline as
 * the paper page, so the two never disagree. The map at the top is the trained propagator's own computation
 * (hero.server.ts).
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve as resolvePath } from 'node:path';
import { asset, resolve } from '$app/paths';
import { parseBib } from '#lib/paper/bib.ts';
import { renderPaper } from '#lib/paper/html.ts';
import { heroData } from '#lib/search/hero.server.ts';
import type { PageServerLoad } from './$types';

export const prerender = true;

/** the figure the project page shows under the abstract: the paper's first */
const KEY = /<figure class="fig[^"]*" id="([\w-]+)">([\s\S]*?)<\/figure>/;

/**
 * What we found: each finding is the abstract's sentence (or sentences) that start with `opens`, shown beside the
 * figure `fig`.
 * A sentence the paper no longer has stops the build, so the page never quotes words the paper dropped.
 */
const FOUND = [
	{ key: 'Search', fig: 'fig-search', opens: ['One propagator layer'] },
	{ key: 'SET', fig: 'fig-set', opens: ['Given the modulus'] },
	{ key: 'The two laws', fig: 'fig-laws', opens: ['The depth law is', 'We build a construction', 'The size law is'] },
	{ key: 'CLRS-30 and elsewhere', fig: 'fig-map', opens: ['On CLRS-30'] }
];

export const load: PageServerLoad = () => {
	const root = process.cwd();
	const md = readFileSync(resolvePath(root, 'paper/paper.md'), 'utf8');
	const bib = parseBib(readFileSync(resolvePath(root, 'paper/refs.bib'), 'utf8'));
	const exists = (f: string) => existsSync(resolvePath(root, 'paper/figures/out', f));
	const src = (f: string) => asset(`/figures/${f}` as Parameters<typeof asset>[0]);
	const out = renderPaper(md, bib, src, exists);
	// links inside the paper (citations, cross-references) point at the paper page from here
	const paper = resolve('/paper');
	const local = (html: string) => html.replace(/href="#/g, `href="${paper}#`);
	const abs = /<section class="abstract"[^>]*>([\s\S]*?)<\/section>/.exec(out.html);
	const abstract = abs ? local(abs[1].replace(/<h2 class="abs-h">Abstract<\/h2>/, '')) : '';
	const fig = KEY.exec(out.html);
	const figure = fig && !fig[2].includes('fig-missing') ? local(fig[2]) : null;

	// the abstract's sentences, split outside maths and parentheses
	const absMd = /::: abstract\n([\s\S]*?)\n:::/.exec(md)?.[1].trim() ?? '';
	const sentences: string[] = [];
	let depth = 0;
	let math = false;
	let at = 0;
	for (let i = 0; i < absMd.length; i++) {
		const c = absMd[i];
		if (c === '$') math = !math;
		else if (!math && c === '(') depth++;
		else if (!math && c === ')') depth--;
		else if (!math && depth === 0 && c === '.' && absMd[i + 1] === ' ') {
			sentences.push(absMd.slice(at, i + 1).trim());
			at = i + 2;
		}
	}
	sentences.push(absMd.slice(at).trim());
	/** one sentence as inline HTML: the paragraph the renderer makes of it, without the reference list it appends */
	const inline = (s: string) =>
		local(/<p>([\s\S]*?)<\/p>/.exec(renderPaper(s, bib, src, exists).html)?.[1] ?? '');
	const found = FOUND.map((f) => {
		const s = f.opens
			.map((o) => {
				const x = sentences.find((y) => y.startsWith(o));
				if (!x) throw new Error(`paper.md: the abstract has no sentence starting "${o}"`);
				return x;
			})
			.join(' ');
		const m = new RegExp(String.raw`^:::\s*figure\s*\{#${f.fig}\s([^}]*)\}`, 'm').exec(md);
		const file = m && /\bsrc=([^\s}"]+)/.exec(m[1])?.[1];
		const alt = m && /\balt="([^"]*)"/.exec(m[1])?.[1];
		const label = new RegExp(
			String.raw`<figure class="fig[^"]*" id="${f.fig}">[\s\S]*?<span class="fig-label">([^<|]*)`
		).exec(out.html)?.[1];
		if (!file || !label) throw new Error(`paper.md: no figure {#${f.fig}}`);
		// the figure's own size, so the page keeps its room before the (lazy) image arrives
		const head = readFileSync(resolvePath(root, 'paper/figures/out', file), 'utf8').slice(0, 2000);
		const vb = /viewBox="[\d.]+ [\d.]+ ([\d.]+) ([\d.]+)"/.exec(head);
		return {
			key: f.key,
			html: inline(s),
			src: src(file),
			width: vb ? Math.round(Number(vb[1])) : undefined,
			height: vb ? Math.round(Number(vb[2])) : undefined,
			alt: alt ?? f.key,
			fig: f.fig,
			label: label.trim()
		};
	});

	return {
		title: out.meta.title ?? '',
		authors: out.meta.authors ?? '',
		affiliation: out.meta.affiliation ?? '',
		abstract,
		figure,
		found,
		hero: (({ layout, question, frames, lookahead }) => ({ layout, question, frames, lookahead }))(
			heroData(root)
		)
	};
};
