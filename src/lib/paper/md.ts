/**
 * The paper's source format, parsed once for both outputs (the web page, html.ts; the arXiv LaTeX, tex.ts). It is
 * the Markdown subset paper/paper.md uses, with a few pandoc-like extensions:
 *
 * - front matter between `---` lines: `key: value`;
 * - headings `## Title {#id .unnumbered}`; level 2 sections are numbered unless `.unnumbered`;
 * - fenced divs `::: kind {#id key="value" .class}` … `:::` (figure, theorem, proposition, lemma, definition, proof,
 *   abstract, status, table, appendix), which may nest; `restate=id` on a theorem repeats that theorem's number;
 * - display maths `$$ … $$` on its own lines, numbered when followed by `{#eq-name}`; inline maths `$…$`;
 * - citations `[@key]`, `[@a; @b]`, and narrative citations `@key` ("Saparov et al.¹"); cross-references `{@id}` (the number of a section, appendix, figure, table,
 *   theorem or equation: write `({@eq-name})` for an equation);
 *   sidenotes `^[…]`;
 * - pipe tables, `-` and `1.` lists, `**strong**`, `*em*`, `` `code` ``, `[text](url)`.
 *
 * No dependencies, so `node paper/build-tex.ts` can import it with Node's type stripping.
 */

export type Attrs = { id?: string; classes: string[]; kv: Record<string, string> };

export type Block =
	| { t: 'heading'; level: number; text: string; attrs: Attrs }
	| { t: 'para'; text: string }
	| { t: 'list'; ordered: boolean; items: string[] }
	| { t: 'table'; head: string[]; align: ('l' | 'r' | 'c')[]; rows: string[][] }
	| { t: 'math'; tex: string; attrs: Attrs; cont: boolean }
	| { t: 'code'; lang: string; text: string }
	| { t: 'div'; kind: string; attrs: Attrs; children: Block[] };

export interface Doc {
	meta: Record<string, string>;
	blocks: Block[];
}

export type Inline =
	| { t: 'text'; s: string }
	| { t: 'math'; tex: string }
	| { t: 'code'; s: string }
	| { t: 'strong'; c: Inline[] }
	| { t: 'em'; c: Inline[] }
	| { t: 'link'; href: string; c: Inline[] }
	| { t: 'cite'; keys: string[]; narrative?: boolean }
	| { t: 'ref'; id: string }
	| { t: 'note'; c: Inline[] }
	| { t: 'br' };

export function parseAttrs(s: string | undefined): Attrs {
	const a: Attrs = { classes: [], kv: {} };
	if (!s) return a;
	const re = /#([\w:-]+)|\.([\w-]+)|([\w-]+)=("([^"]*)"|[^\s}]+)/g;
	for (const m of s.matchAll(re)) {
		if (m[1]) a.id = m[1];
		else if (m[2]) a.classes.push(m[2]);
		else if (m[3]) a.kv[m[3]] = m[5] ?? m[4];
	}
	return a;
}

const FENCE = /^:::\s*([\w-]+)?\s*(\{.*\})?\s*$/;
const HEADING = /^(#{1,4})\s+(.*?)\s*(\{[^}]*\})?\s*$/;
const LIST = /^(\s*)(-|\d+\.)\s+(.*)$/;
/** the end of a display: `$$`, optionally followed by attributes such as `{#eq-name}` */
const MATH_END = /\$\$\s*(\{[^}]*\})?\s*$/;
/** theorem-like environments, numbered by kind */
export const THEOREM_LIKE = ['theorem', 'proposition', 'lemma', 'definition'];

function isTableSep(line: string) {
	return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$/.test(line);
}

/** split a table row on unescaped pipes outside inline maths and code */
export function splitRow(line: string): string[] {
	let s = line.trim();
	if (s.startsWith('|')) s = s.slice(1);
	if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
	const cells: string[] = [];
	let cur = '';
	let math = false;
	let code = false;
	for (let i = 0; i < s.length; i++) {
		const ch = s[i];
		if (ch === '\\' && s[i + 1] === '|') {
			cur += '|';
			i++;
			continue;
		}
		if (ch === '`') code = !code;
		if (ch === '$' && !code) math = !math;
		if (ch === '|' && !math && !code) {
			cells.push(cur.trim());
			cur = '';
		} else cur += ch;
	}
	cells.push(cur.trim());
	return cells;
}

export function parse(src: string): Doc {
	const lines = src.replace(/\r\n?/g, '\n').split('\n');
	const meta: Record<string, string> = {};
	let i = 0;
	if (lines[0]?.trim() === '---') {
		i = 1;
		while (i < lines.length && lines[i].trim() !== '---') {
			const m = /^([\w-]+):\s*(.*)$/.exec(lines[i]);
			if (m) meta[m[1]] = m[2].trim();
			i++;
		}
		i++;
	}
	const [blocks] = parseBlocks(lines, i, false);
	return { meta, blocks };
}

function parseBlocks(lines: string[], start: number, inDiv: boolean): [Block[], number] {
	const out: Block[] = [];
	let i = start;
	while (i < lines.length) {
		const line = lines[i];
		const trimmed = line.trim();
		if (trimmed === '') {
			i++;
			continue;
		}
		if (trimmed.startsWith('<!--')) {
			while (i < lines.length && !lines[i].includes('-->')) i++;
			i++;
			continue;
		}
		const fence = FENCE.exec(trimmed);
		if (fence) {
			if (!fence[1]) {
				if (inDiv) return [out, i + 1];
				throw new Error(`stray ::: at line ${i + 1}`);
			}
			const [children, next] = parseBlocks(lines, i + 1, true);
			out.push({ t: 'div', kind: fence[1], attrs: parseAttrs(fence[2]), children });
			i = next;
			continue;
		}
		const h = HEADING.exec(line);
		if (h) {
			out.push({ t: 'heading', level: h[1].length, text: h[2], attrs: parseAttrs(h[3]) });
			i++;
			continue;
		}
		if (trimmed.startsWith('```')) {
			const lang = trimmed.slice(3).trim();
			const body: string[] = [];
			i++;
			while (i < lines.length && !lines[i].trim().startsWith('```')) body.push(lines[i++]);
			out.push({ t: 'code', lang, text: body.join('\n') });
			i++;
			continue;
		}
		if (trimmed.startsWith('$$')) {
			const body: string[] = [];
			const rest = trimmed.slice(2);
			let end = MATH_END.exec(rest);
			if (end) {
				body.push(rest.slice(0, end.index));
			} else {
				if (rest) body.push(rest);
				i++;
				while (i < lines.length && !(end = MATH_END.exec(lines[i].trim()))) body.push(lines[i++]);
				if (end) body.push(lines[i].trim().slice(0, end.index));
			}
			i++;
			// a display followed directly by text continues its paragraph
			const cont = i < lines.length && lines[i].trim() !== '' && !FENCE.test(lines[i].trim());
			out.push({ t: 'math', tex: body.join('\n').trim(), attrs: parseAttrs(end?.[1]), cont });
			continue;
		}
		if (trimmed.startsWith('|') && i + 1 < lines.length && isTableSep(lines[i + 1])) {
			const head = splitRow(line);
			const align = splitRow(lines[i + 1]).map((c) =>
				c.startsWith(':') && c.endsWith(':') ? 'c' : c.endsWith(':') ? 'r' : 'l'
			);
			const rows: string[][] = [];
			i += 2;
			while (i < lines.length && lines[i].trim().startsWith('|')) rows.push(splitRow(lines[i++]));
			out.push({ t: 'table', head, align, rows });
			continue;
		}
		const li = LIST.exec(line);
		if (li && li[1].length === 0) {
			const ordered = li[2] !== '-';
			const items: string[] = [];
			while (i < lines.length) {
				const m = LIST.exec(lines[i]);
				if (m && m[1].length === 0 && (m[2] !== '-') === ordered) {
					items.push(m[3]);
					i++;
				} else if (lines[i].trim() !== '' && /^\s+/.test(lines[i]) && items.length) {
					items[items.length - 1] += ' ' + lines[i].trim();
					i++;
				} else break;
			}
			out.push({ t: 'list', ordered, items });
			continue;
		}
		const para: string[] = [];
		while (i < lines.length) {
			const l = lines[i];
			const tl = l.trim();
			if (
				tl === '' ||
				FENCE.test(tl) ||
				HEADING.test(l) ||
				tl.startsWith('$$') ||
				tl.startsWith('```') ||
				(LIST.test(l) && !/^\s/.test(l))
			)
				break;
			para.push(tl);
			i++;
		}
		out.push({ t: 'para', text: para.join(' ') });
	}
	if (inDiv) throw new Error('unclosed ::: div');
	return [out, i];
}

/** find the index of the closing bracket for an opening one at `i`, honouring nesting and inline maths */
function closing(s: string, i: number, open: string, close: string): number {
	let depth = 0;
	let math = false;
	for (let j = i; j < s.length; j++) {
		const ch = s[j];
		if (ch === '\\') {
			j++;
			continue;
		}
		if (ch === '$') math = !math;
		if (math) continue;
		if (ch === open) depth++;
		else if (ch === close) {
			depth--;
			if (depth === 0) return j;
		}
	}
	return -1;
}

/** Inline tokens of a text run. */
export function inline(s: string): Inline[] {
	const out: Inline[] = [];
	let buf = '';
	const flush = () => {
		if (buf) out.push({ t: 'text', s: buf });
		buf = '';
	};
	for (let i = 0; i < s.length; i++) {
		const ch = s[i];
		const rest = s.slice(i);
		if (ch === '\\' && i + 1 < s.length && '$*`[]{}\\_^|'.includes(s[i + 1])) {
			buf += s[i + 1];
			i++;
			continue;
		}
		if (ch === '$') {
			const j = s.indexOf('$', i + 1);
			if (j > i) {
				flush();
				out.push({ t: 'math', tex: s.slice(i + 1, j) });
				i = j;
				continue;
			}
		}
		if (ch === '`') {
			const j = s.indexOf('`', i + 1);
			if (j > i) {
				flush();
				out.push({ t: 'code', s: s.slice(i + 1, j) });
				i = j;
				continue;
			}
		}
		if (rest.startsWith('**')) {
			const j = s.indexOf('**', i + 2);
			if (j > i) {
				flush();
				out.push({ t: 'strong', c: inline(s.slice(i + 2, j)) });
				i = j + 1;
				continue;
			}
		}
		if (ch === '*' && s[i + 1] !== ' ') {
			const j = s.indexOf('*', i + 1);
			if (j > i + 1) {
				flush();
				out.push({ t: 'em', c: inline(s.slice(i + 1, j)) });
				i = j;
				continue;
			}
		}
		if (rest.startsWith('^[')) {
			const j = closing(s, i + 1, '[', ']');
			if (j > 0) {
				flush();
				out.push({ t: 'note', c: inline(s.slice(i + 2, j)) });
				i = j;
				continue;
			}
		}
		if (rest.startsWith('{@')) {
			const m = /^\{@([\w:-]+)\}/.exec(rest);
			if (m) {
				flush();
				out.push({ t: 'ref', id: m[1] });
				i += m[0].length - 1;
				continue;
			}
		}
		if (rest.startsWith('[@')) {
			const j = s.indexOf(']', i);
			if (j > 0) {
				// a citation is a superscript set against the word before it
				buf = buf.replace(/\s+$/, '');
				flush();
				const keys = s
					.slice(i + 1, j)
					.split(';')
					.map((k) => k.trim().replace(/^@/, ''))
					.filter(Boolean);
				out.push({ t: 'cite', keys });
				i = j;
				continue;
			}
		}
		if (ch === '[') {
			const j = closing(s, i, '[', ']');
			if (j > 0 && s[j + 1] === '(') {
				const k = s.indexOf(')', j + 1);
				if (k > 0) {
					flush();
					out.push({ t: 'link', href: s.slice(j + 2, k), c: inline(s.slice(i + 1, j)) });
					i = k;
					continue;
				}
			}
		}
		// a narrative citation: `@key` after a space or an opening bracket, not inside a word or an address
		if (ch === '@' && (i === 0 || /[\s(—–-]/.test(s[i - 1]))) {
			const m = /^@([A-Za-z][\w:-]*[A-Za-z0-9])/.exec(rest);
			if (m) {
				flush();
				out.push({ t: 'cite', keys: [m[1]], narrative: true });
				i += m[0].length - 1;
				continue;
			}
		}
		if (rest.startsWith('  \n')) {
			flush();
			out.push({ t: 'br' });
			i += 2;
			continue;
		}
		buf += ch;
	}
	flush();
	return out;
}

/** Every inline run of a block list, in document order (for numbering citations by first use). */
export function* texts(blocks: Block[]): Generator<string> {
	for (const b of blocks) {
		if (b.t === 'heading' || b.t === 'para') yield b.text;
		else if (b.t === 'list') yield* b.items;
		else if (b.t === 'table') {
			yield* b.head;
			for (const r of b.rows) yield* r;
		} else if (b.t === 'div') {
			if (b.attrs.kv.title) yield b.attrs.kv.title;
			yield* texts(b.children);
		}
	}
}

export function* inlineWalk(xs: Inline[]): Generator<Inline> {
	for (const x of xs) {
		yield x;
		if ('c' in x) yield* inlineWalk(x.c);
	}
}

/** Numbers for sections, figures, tables, theorems and citations, fixed before either output is written. */
export interface Numbering {
	ids: Map<string, { kind: string; n: string }>;
	cites: Map<string, number>;
}

export function number(doc: Doc): Numbering {
	const ids = new Map<string, { kind: string; n: string }>();
	// a tutorial numbers its chapters from `first-section` (0 for a prologue chapter)
	let sec = Number(doc.meta['first-section'] ?? '1') - 1;
	let app = 0;
	let sub = 0;
	let cur = '';
	const count: Record<string, number> = {};
	const walk = (blocks: Block[], appendix: boolean) => {
		for (const b of blocks) {
			if (b.t === 'heading') {
				if (b.level === 2 && !b.attrs.classes.includes('unnumbered')) {
					// sections are numbered 1, 2, …; appendices A, B, …
					cur = appendix ? String.fromCharCode(65 + app++) : String(++sec);
					sub = 0;
					if (b.attrs.id) ids.set(b.attrs.id, { kind: appendix ? 'appendix' : 'section', n: cur });
				} else if (b.level === 3 && b.attrs.id) {
					sub++;
					ids.set(b.attrs.id, { kind: 'section', n: `${cur}.${sub}` });
				}
			} else if (b.t === 'math') {
				if (b.attrs.id) {
					count.equation = (count.equation ?? 0) + 1;
					ids.set(b.attrs.id, { kind: 'equation', n: String(count.equation) });
				}
			} else if (b.t === 'div') {
				const k = b.kind;
				if (['figure', 'table', ...THEOREM_LIKE].includes(k) && !b.attrs.kv.restate) {
					const ext = b.attrs.classes.includes('extended') || appendix;
					const key = (ext ? 'ext-' : '') + k;
					count[key] = (count[key] ?? 0) + 1;
					if (b.attrs.id) ids.set(b.attrs.id, { kind: key, n: String(count[key]) });
				}
				walk(b.children, appendix || k === 'appendix');
			}
		}
	};
	walk(doc.blocks, false);
	const cites = new Map<string, number>();
	for (const s of texts(doc.blocks))
		for (const x of inlineWalk(inline(s)))
			if (x.t === 'cite') for (const k of x.keys) if (!cites.has(k)) cites.set(k, cites.size + 1);
	return { ids, cites };
}

/** "3, 5–7" from citation numbers */
export function compress(ns: number[]): string {
	const s = [...new Set(ns)].sort((a, b) => a - b);
	const parts: string[] = [];
	for (let i = 0; i < s.length;) {
		let j = i;
		while (j + 1 < s.length && s[j + 1] === s[j] + 1) j++;
		parts.push(j - i >= 2 ? `${s[i]}–${s[j]}` : j > i ? `${s[i]},${s[j]}` : `${s[i]}`);
		i = j + 1;
	}
	return parts.join(',');
}

/** word count of the main text: paragraphs and lists outside figures, tables, theorems' Lean lines and Methods */
export function wordCount(doc: Doc, stopAt = 'methods'): number {
	let n = 0;
	for (const b of doc.blocks) {
		if (b.t === 'heading' && b.attrs.id === stopAt) break;
		if (b.t === 'para') n += b.text.split(/\s+/).filter(Boolean).length;
		if (b.t === 'list') for (const it of b.items) n += it.split(/\s+/).filter(Boolean).length;
		if (b.t === 'div' && ['theorem', 'proof'].includes(b.kind))
			for (const c of b.children) if (c.t === 'para') n += c.text.split(/\s+/).length;
	}
	return n;
}
