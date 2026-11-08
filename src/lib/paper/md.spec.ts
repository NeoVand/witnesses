import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseBib } from './bib.ts';
import { compress, inline, number, parse } from './md.ts';
import { renderPaper } from './html.ts';
import { renderTex } from './tex.ts';

describe('the paper source', () => {
	it('parses divs, headings, tables and maths', () => {
		const doc = parse(
			'---\ntitle: T\n---\n## A {#sec-a}\n\nText $x$ [@k].\n\n::: theorem {#thm-a title="P"}\nBody.\n:::\n\n| a | b |\n|---|--:|\n| 1 | $|x|$ |\n\n$$\ny = x\n$$\n'
		);
		expect(doc.meta.title).toBe('T');
		expect(doc.blocks.map((b) => b.t)).toEqual(['heading', 'para', 'div', 'table', 'math']);
		const n = number(doc);
		expect(n.ids.get('sec-a')?.n).toBe('1');
		expect(n.ids.get('thm-a')?.n).toBe('1');
		expect(n.cites.get('k')).toBe(1);
	});

	it('sets a citation against the word before it and escapes dollars', () => {
		const xs = inline('word [@a; @b] costs \\$7');
		expect(xs[0]).toEqual({ t: 'text', s: 'word' });
		expect(xs[1]).toEqual({ t: 'cite', keys: ['a', 'b'] });
		expect(xs[2]).toEqual({ t: 'text', s: ' costs $7' });
	});

	it('names the authors of a narrative citation and numbers it with the rest', () => {
		const bib = parseBib(
			'@article{a2025x, author={Saparov, Abulhair and Pawar, S. and He, He}, title={T}, year={2025}}\n' +
				'@article{b2019y, author={Clift, James and Murfet, Daniel}, title={U}, year={2019}}'
		);
		const md = '@a2025x trained models [@b2019y]; see @b2019y, not x@y.com.';
		const xs = inline(md);
		expect(xs[0]).toEqual({ t: 'cite', keys: ['a2025x'], narrative: true });
		const html = renderPaper(
			md,
			bib,
			(f) => f,
			() => true
		).html;
		expect(html).toContain('<span class="cite-name">Saparov et al.</span><sup class="cite">');
		expect(html).toContain(
			'<span class="cite-name">Clift and Murfet</span><sup class="cite"><a href="#ref-2"'
		);
		expect(html).toContain('x@y.com');
		const tex = renderTex(md, bib, { figDir: '', figExists: () => true });
		expect(tex).toContain('Saparov et al.~\\cite{a2025x} trained');
	});

	it('compresses citation runs', () => {
		expect(compress([3, 1, 2, 7, 5, 6])).toBe('1–3,5–7');
		expect(compress([4, 5])).toBe('4,5');
	});

	it('numbers display equations and refers to them in both outputs', () => {
		const md = [
			'## A {#sec-a}',
			'',
			'By ({@eq-b}), $x$ is one:',
			'$$ x = 1 $$ {#eq-a}',
			'and then',
			'$$',
			'y = 2',
			'$$ {#eq-b}',
			'',
			'$$ z $$',
			'',
			'New paragraph.'
		].join('\n');
		const doc = parse(md);
		expect(doc.blocks.map((b) => b.t)).toEqual([
			'heading',
			'para',
			'math',
			'para',
			'math',
			'math',
			'para'
		]);
		const n = number(doc);
		expect(n.ids.get('eq-a')).toEqual({ kind: 'equation', n: '1' });
		expect(n.ids.get('eq-b')).toEqual({ kind: 'equation', n: '2' });
		const html = renderPaper(
			md,
			new Map(),
			(f) => f,
			() => true
		);
		expect(html.missing).toEqual([]);
		expect(html.html).toContain('(<a class="xref" href="#eq-b">2</a>)');
		expect(html.html).toContain('<div class="dmath" id="eq-b"');
		// the number sits beside the formula, not as KaTeX's tag; an unnumbered display has none
		expect(html.html.match(/<span class="eq-n"/g)?.length).toBe(2);
		expect(html.html).not.toContain('katex-tag');
		const tex = renderTex(md, new Map(), { figDir: '', figExists: () => true });
		expect(tex).toContain('(\\hyperref[eq-b]{2})');
		expect(tex).toContain('\\begin{equation}\ny = 2\n\\tag{2}\\label{eq-b}\n\\end{equation}');
		// a display stays in its paragraph: no blank line before it, none after it when the sentence goes on
		// each tagged equation names its own hyperref anchor, which amsmath's \tag would otherwise share
		expect(tex).toContain(
			'is one:\n\\renewcommand{\\theHequation}{eq-a}\n\\begin{equation}\nx = 1\n\\tag{1}\\label{eq-a}\n\\end{equation}\nand then\n\\renewcommand{\\theHequation}{eq-b}\n\\begin'
		);
		expect(tex).toContain('\\[\nz\n\\]\n\nNew paragraph.');
	});

	it('numbers lemmas and definitions, repeats a restated theorem, and letters appendices', () => {
		const md = [
			'## A {#sec-a}',
			'::: theorem {#thm-a title="Main"}',
			'Body.',
			':::',
			'::: appendix',
			'## Proofs {#app-p}',
			'::: definition {#def-a}',
			'D.',
			':::',
			'::: lemma {#lem-a}',
			'L.',
			':::',
			'::: theorem {restate=thm-a}',
			'Body.',
			':::',
			'## More {#app-q}',
			':::'
		].join('\n');
		const n = number(parse(md));
		expect(n.ids.get('app-p')).toEqual({ kind: 'appendix', n: 'A' });
		expect(n.ids.get('app-q')).toEqual({ kind: 'appendix', n: 'B' });
		expect(n.ids.get('lem-a')?.n).toBe('1');
		expect(n.ids.get('def-a')?.n).toBe('1');
		const html = renderPaper(
			md,
			new Map(),
			(f) => f,
			() => true
		).html;
		expect(html).toContain('<span class="thm-label">Lemma 1</span>');
		expect(html).toContain('<span class="thm-label">Definition 1</span>');
		expect(html.match(/<span class="thm-label">Theorem 1<\/span>/g)?.length).toBe(2);
		const tex = renderTex(md, new Map(), { figDir: '', figExists: () => true });
		expect(tex).toContain('\\appendix\n\\section{Proofs}\\label{app-p}');
		expect(tex).toContain('\\textbf{Lemma 1}');
	});

	it('renders the real paper to HTML and LaTeX with every reference resolved', () => {
		const md = readFileSync('paper/paper.md', 'utf8');
		const bib = parseBib(readFileSync('paper/refs.bib', 'utf8'));
		const html = renderPaper(
			md,
			bib,
			(f) => `/figures/${f}`,
			() => true
		);
		expect(html.missing).toEqual([]);
		const tex = renderTex(md, bib, { figDir: '', figExists: () => true });
		expect(tex).toContain('\\begin{thebibliography}');
		// the arXiv build is plain ASCII for pdfLaTeX
		const arxiv = renderTex(md, bib, { figDir: '', figExists: () => true, engine: 'pdflatex' });
		expect(arxiv).not.toMatch(/[\u0080-\u{10FFFF}]/u);
		expect(arxiv).toContain('\\pdfoutput=1');
		expect(arxiv).not.toContain('fontspec');
	});

	it('renders the tutorial with every reference resolved, its chapters numbered from 0, its asides and figures', () => {
		const md = readFileSync('tutorial/tutorial.md', 'utf8');
		const bib = parseBib(readFileSync('tutorial/refs.bib', 'utf8'));
		const out = renderPaper(
			md,
			bib,
			(f) => `/figures/${f}`,
			() => true,
			{ labels: { figure: 'Figure' } }
		);
		expect(out.missing).toEqual([]);
		expect(out.toc[0].n).toBe('0');
		expect(out.html).toContain('<aside class="aside"');
		expect(out.html).toContain('<span class="fig-label">Figure 1 |</span>');
	});

	it('maps Unicode to LaTeX commands for pdfLaTeX, in text and in maths, and refuses what it cannot map', () => {
		const opt = { figDir: '', figExists: () => true, engine: 'pdflatex' as const };
		const tex = renderTex(
			'Gr\u00f6bner: 3 \u00d7 4 \u2212 1 \u2013 \u201cok\u201d, $a \u2264 b$.',
			new Map(),
			opt
		);
		expect(tex).toContain(
			'Gr\\"{o}bner: 3 \\ensuremath{\\times} 4 \\ensuremath{-} 1 \\textendash{} \\textquotedblleft{}ok\\textquotedblright{}, $a \\le  b$.'
		);
		expect(() => renderTex('\u2603', new Map(), opt)).toThrow(/U\+2603/);
	});
});
