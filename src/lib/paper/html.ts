/**
 * paper.md to the web page's HTML, at build time (the paper route's server load; the page is prerendered). Maths is
 * rendered by KaTeX here, so the page ships no maths code. Layout classes are styled by src/routes/+page.svelte.
 */
import katex from 'katex';
import { citeName, detex, format, type Entry } from './bib.ts';
import {
	compress,
	inline,
	number,
	parse,
	wordCount,
	type Attrs,
	type Block,
	type Inline,
	type Numbering
} from './md.ts';

export interface Rendered {
	meta: Record<string, string>;
	html: string;
	toc: { id: string; n: string; title: string }[];
	words: number;
	missing: string[];
}

const esc = (s: string) =>
	s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** typographic touches for plain text: thin spaces are kept, straight quotes curled */
const typo = (s: string) =>
	// quotes are curled before escaping, which would otherwise hide the straight ones as &quot;
	esc(
		s
			.replace(/(^|[\s(—[])"/g, '$1“')
			.replace(/"/g, '”')
			.replace(/(^|[\s(—[])'/g, '$1‘')
			.replace(/'/g, '’')
			.replace(/ -- /g, ' – ')
			.replace(/---/g, '—')
	);

const KIND_LABEL: Record<string, string> = {
	section: 'Section',
	figure: 'Fig.',
	'ext-figure': 'Supplementary Fig.',
	table: 'Table',
	'ext-table': 'Supplementary Table',
	theorem: 'Theorem',
	'ext-theorem': 'Theorem',
	proposition: 'Proposition',
	'ext-proposition': 'Proposition',
	lemma: 'Lemma',
	'ext-lemma': 'Lemma',
	definition: 'Definition',
	'ext-definition': 'Definition',
	appendix: 'Appendix'
};

function tex(src: string, display: boolean): string {
	return katex.renderToString(src, {
		displayMode: display,
		throwOnError: true,
		output: 'html',
		strict: 'ignore'
	});
}

export function renderPaper(
	md: string,
	bib: Map<string, Entry>,
	figUrl: (f: string) => string,
	figExists: (f: string) => boolean,
	opts: { labels?: Record<string, string> } = {}
): Rendered {
	const LABELS = { ...KIND_LABEL, ...opts.labels };
	const doc = parse(md);
	const num = number(doc);
	const missing: string[] = [];
	let noteN = 0;

	const inl = (xs: Inline[]): string =>
		xs
			.map((x) => {
				switch (x.t) {
					case 'text':
						return typo(x.s);
					case 'math':
						return tex(x.tex, false);
					case 'code':
						return `<code>${esc(x.s)}</code>`;
					case 'strong':
						return `<strong>${inl(x.c)}</strong>`;
					case 'em':
						return `<em>${inl(x.c)}</em>`;
					case 'link':
						return `<a href="${esc(x.href)}">${inl(x.c)}</a>`;
					case 'br':
						return '<br>';
					case 'ref': {
						const r = num.ids.get(x.id);
						if (!r) {
							missing.push(`ref:${x.id}`);
							return '??';
						}
						return `<a class="xref" href="#${esc(x.id)}">${r.n}</a>`;
					}
					case 'cite': {
						const ns = x.keys.map((k) => {
							if (!bib.has(k)) missing.push(`cite:${k}`);
							return num.cites.get(k)!;
						});
						const title = x.keys
							.map((k) => {
								const e = bib.get(k);
								if (!e) return k;
								const f = format(e);
								return `${num.cites.get(k)}. ${detex(f.authors)} ${detex(f.title)} (${f.year})`;
							})
							.join('\n');
						const first = Math.min(...ns);
						const who =
							x.narrative && bib.has(x.keys[0])
								? `<span class="cite-name">${typo(detex(citeName(bib.get(x.keys[0])!)))}</span>`
								: '';
						return `${who}<sup class="cite"><a href="#ref-${first}" title="${esc(title)}">${compress(ns)}</a></sup>`;
					}
					case 'note': {
						noteN++;
						return `<label class="sn-mark" for="sn-${noteN}">${noteN}</label><input type="checkbox" id="sn-${noteN}" class="sn-toggle"><span class="sidenote"><span class="sn-n">${noteN}</span> ${inl(x.c)}</span>`;
					}
				}
			})
			.join('');
	const run = (s: string) => inl(inline(s));

	const label = (id: string | undefined) => {
		const r = id ? num.ids.get(id) : undefined;
		return r ? `${LABELS[r.kind]} ${r.n}` : '';
	};

	const toc: Rendered['toc'] = [];

	const blocks = (bs: Block[], ctx: { inFig?: boolean } = {}): string =>
		bs
			.map((b) => {
				switch (b.t) {
					case 'heading': {
						const id = b.attrs.id ?? '';
						const r = id ? num.ids.get(id) : undefined;
						if (b.level === 2) {
							toc.push({ id, n: r?.n ?? '', title: b.text });
							return `<h2 id="${esc(id)}" class="sec"><span class="sec-n">${r?.n ?? ''}</span><span class="sec-t">${run(b.text)}</span></h2>`;
						}
						const tag = `h${b.level}`;
						return `<${tag}${id ? ` id="${esc(id)}"` : ''}>${run(b.text)}</${tag}>`;
					}
					case 'para':
						return ctx.inFig ? run(b.text) : `<p>${run(b.text)}</p>`;
					case 'list': {
						const tag = b.ordered ? 'ol' : 'ul';
						return `<${tag}>${b.items.map((it) => `<li>${run(it)}</li>`).join('')}</${tag}>`;
					}
					case 'math': {
						const r = b.attrs.id ? num.ids.get(b.attrs.id) : undefined;
						if (!r) return `<div class="dmath">${tex(b.tex, true)}</div>`;
						// a numbered display: the number beside the formula, which scrolls on its own when it is wider than
						// the column (KaTeX's \tag would sit on top of an overflowing formula)
						return (
							`<div class="dmath" id="${esc(b.attrs.id!)}" style="display:flex;align-items:center;gap:1em;overflow:visible">` +
							`<div style="flex:1 1 auto;min-width:0;overflow-x:auto;overflow-y:hidden">${tex(b.tex, true)}</div>` +
							`<span class="eq-n" style="flex:none">${tex(`(${r.n})`, false)}</span></div>`
						);
					}
					case 'code':
						return `<pre class="code"><code>${esc(b.text)}</code></pre>`;
					case 'table': {
						const al = (i: number) =>
							b.align[i] === 'r' ? ' class="r"' : b.align[i] === 'c' ? ' class="c"' : '';
						const head = b.head.map((h, i) => `<th${al(i)}>${run(h)}</th>`).join('');
						const rows = b.rows
							.map((r) => `<tr>${r.map((c, i) => `<td${al(i)}>${run(c)}</td>`).join('')}</tr>`)
							.join('');
						return `<div class="tbl-wrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
					}
					case 'div':
						return div(b.kind, b.attrs, b.children);
				}
			})
			.join('\n');

	const div = (kind: string, a: Attrs, children: Block[]): string => {
		const id = a.id ? ` id="${esc(a.id)}"` : '';
		switch (kind) {
			case 'abstract':
				return `<section class="abstract"${id}><h2 class="abs-h">Abstract</h2>${blocks(children)}</section>`;
			case 'figure': {
				const src = a.kv.src;
				let img: string;
				if (src && figExists(src)) {
					// the figure opens at full size in its own tab: on a phone, pinch to read it
					img = `<a class="fig-open" href="${esc(figUrl(src))}" target="_blank" rel="noopener" title="Open the figure at full size"><img src="${esc(figUrl(src))}" alt="${esc(a.kv.alt ?? label(a.id))}" loading="lazy"></a>`;
				} else {
					missing.push(`figure:${src}`);
					img = `<div class="fig-missing">${esc(src ?? 'figure')} is being drawn</div>`;
				}
				const [first, ...rest] = children;
				const cap =
					first && first.t === 'para'
						? `<span class="fig-label">${label(a.id)} |</span> ${run(first.text)}` +
							(rest.length ? ' ' + blocks(rest, { inFig: true }) : '')
						: blocks(children);
				return `<figure class="fig${a.classes.includes('extended') ? ' ext' : ''}"${id}>${img}<figcaption>${cap}</figcaption></figure>`;
			}
			case 'table': {
				const [first, ...rest] = children;
				const cap =
					first && first.t === 'para'
						? `<p class="tbl-cap"><span class="fig-label">${label(a.id)} |</span> ${run(first.text)}</p>`
						: '';
				return `<div class="tbl"${id}>${cap}${blocks(first && first.t === 'para' ? rest : children)}</div>`;
			}
			case 'theorem':
			case 'proposition':
			case 'lemma':
			case 'definition': {
				const name = a.kv.title ? ` <span class="thm-name">(${run(a.kv.title)})</span>` : '';
				const lean = a.kv.lean
					? `<p class="lean"><span class="lean-k">Lean</span> ${a.kv.lean
							.split(/,\s*/)
							.map((n) => `<code>${esc(n)}</code>`)
							.join(', ')}</p>`
					: '';
				// a restated theorem repeats the label of the one it restates
				return `<div class="thm"${id}><p class="thm-head"><span class="thm-label">${label(a.kv.restate ?? a.id)}</span>${name}</p>${blocks(children)}${lean}</div>`;
			}
			case 'proof':
				return `<div class="proof"><p class="proof-head">${a.kv.title ? run(a.kv.title) : 'Proof sketch'}.</p>${blocks(children)}</div>`;
			case 'aside':
				// a note beside the argument (an analogy, a worked example): its title, then the note
				return `<aside class="aside"${id}>${a.kv.title ? `<p class="aside-t">${run(a.kv.title)}</p>` : ''}${blocks(children)}</aside>`;
			case 'status':
				return `<aside class="status"${id}><span class="status-k">${esc(a.kv.label ?? 'Today')}</span>${blocks(children)}</aside>`;
			default:
				return `<div class="${esc(kind)}"${id}>${blocks(children)}</div>`;
		}
	};

	let html = blocks(doc.blocks);

	// references, in order of first citation
	const keys = [...num.cites.entries()].sort((a, b) => a[1] - b[1]);
	const items = keys
		.map(([k, n]) => {
			const e = bib.get(k);
			if (!e) return `<li id="ref-${n}" value="${n}">${esc(k)} (missing)</li>`;
			const f = format(e);
			const t = (s: string) =>
				typo(detex(s.replace(/\\textbf\{([^}]*)\}/g, '\uE000$1\uE001')))
					.replace(/\uE000/g, '<b>')
					.replace(/\uE001/g, '</b>');
			const link = f.href
				? ` <a class="ref-link" href="${esc(f.href)}">${esc(f.href.replace(/^https?:\/\//, '').replace(/\/$/, ''))}</a>`
				: '';
			return `<li id="ref-${n}" value="${n}">${t(f.authors)} ${t(f.title)}. ${t(f.venue)} (${esc(f.year)}).${link}</li>`;
		})
		.join('\n');
	html += `\n<h2 id="references" class="sec"><span class="sec-n"></span><span class="sec-t">References</span></h2>\n<ol class="refs">${items}</ol>`;
	toc.push({ id: 'references', n: '', title: 'References' });

	return { meta: doc.meta, html, toc, words: wordCount(doc), missing };
}

export { number, parse };
export type { Numbering };
