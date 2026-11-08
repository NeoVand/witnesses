/**
 * paper.md to LaTeX for arXiv (paper/build-tex.ts writes paper/tex/main.tex; tectonic builds it). Numbers for
 * sections, figures, tables, theorems and citations come from md.ts, as on the web page, so the two never drift:
 * figure and table numbers are written out, not left to LaTeX's counters, and the bibliography is emitted in
 * order of first citation.
 */
import { citeName, format, type Entry } from './bib.ts';
import { inline, number, parse, type Attrs, type Block, type Inline } from './md.ts';

const SPECIAL: Record<string, string> = {
	'\\': '\\textbackslash{}',
	'&': '\\&',
	'%': '\\%',
	$: '\\$',
	'#': '\\#',
	_: '\\_',
	'{': '\\{',
	'}': '\\}',
	'~': '\\textasciitilde{}',
	'^': '\\textasciicircum{}'
};

export const escText = (s: string) =>
	s
		.replace(/[\\&%$#_{}~^]/g, (c) => SPECIAL[c])
		.replace(/(^|[\s(—[])"/g, '$1``')
		.replace(/"/g, "''")
		.replace(/(^|[\s(—[])'/g, '$1`');

/** code: monospace, breakable after punctuation */
const escCode = (s: string) =>
	'\\texttt{' + escText(s).replace(/(\\_|\.|\/|,|=)/g, '$1\\allowbreak{}') + '}';

const KIND_LABEL: Record<string, string> = {
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
	'ext-definition': 'Definition'
};

export interface TexOptions {
	figDir: string;
	figExists: (pdf: string) => boolean;
	/**
	 * 'xetex' (default): fontspec, Unicode mapped in the preamble with newunicodechar, for tectonic.
	 * 'pdflatex': for arXiv, which compiles with pdfLaTeX. T1 Latin Modern through fontenc and inputenc, and every
	 * non-ASCII character replaced by a LaTeX command, so the file is plain ASCII; an unmapped character throws.
	 */
	engine?: 'xetex' | 'pdflatex';
}

export function renderTex(md: string, bib: Map<string, Entry>, opt: TexOptions): string {
	const doc = parse(md);
	const num = number(doc);
	const missing: string[] = [];
	const pdf = opt.engine === 'pdflatex';
	const mathTex = (s: string) => (pdf ? asciiMath(s) : s);

	const inl = (xs: Inline[]): string =>
		xs
			.map((x) => {
				switch (x.t) {
					case 'text':
						return escText(x.s);
					case 'math':
						return `$${mathTex(x.tex)}$`;
					case 'code':
						return escCode(x.s);
					case 'strong':
						return `\\textbf{${inl(x.c)}}`;
					case 'em':
						return `\\emph{${inl(x.c)}}`;
					case 'link':
						return `\\href{${x.href.replace(/[%#]/g, (c) => '\\' + c)}}{${inl(x.c)}}`;
					case 'br':
						return '\\\\';
					case 'ref': {
						const r = num.ids.get(x.id);
						if (!r) {
							missing.push(x.id);
							return '??';
						}
						return `\\hyperref[${x.id}]{${r.n}}`;
					}
					case 'cite': {
						const e = x.narrative ? bib.get(x.keys[0]) : undefined;
						return e ? `${citeName(e)}~\\cite{${x.keys[0]}}` : `\\cite{${x.keys.join(',')}}`;
					}
					case 'note':
						return `\\footnote{${inl(x.c)}}`;
				}
			})
			.join('');
	const run = (s: string) => inl(inline(s));
	const label = (id: string | undefined) => {
		const r = id ? num.ids.get(id) : undefined;
		return r ? `${KIND_LABEL[r.kind]} ${r.n}` : '';
	};

	const table = (b: Extract<Block, { t: 'table' }>) => {
		const width = b.head.map((_, i) =>
			Math.max(...[b.head[i], ...b.rows.map((r) => r[i] ?? '')].map((c) => c.length))
		);
		const wide = width.reduce((a, c) => a + c, 0) > 90;
		const spec = b.align
			.map((a, i) => (wide && width[i] > 24 && a === 'l' ? '>{\\raggedright\\arraybackslash}X' : a))
			.join('');
		const env = wide ? 'tabularx' : 'tabular';
		const open = wide ? `\\begin{tabularx}{\\linewidth}{${spec}}` : `\\begin{tabular}{${spec}}`;
		const row = (cs: string[]) => cs.map(run).join(' & ') + ' \\\\';
		// In a wide table, a header longer than its column's entries goes on two lines, so the column is as narrow
		// as its numbers and the wrapping first column gets the width (one line per model name where it fits).
		const data = b.head.map((_, i) => Math.max(0, ...b.rows.map((r) => (r[i] ?? '').length)));
		// Once one header splits, every multi-word number header splits too, so the header row reads as one block.
		const long = (h: string, i: number) =>
			wide && b.align[i] !== 'l' && h.includes(' ') && h.length > data[i] + 2;
		const anyLong = b.head.some(long);
		const headCell = (h: string, i: number) => {
			const sp = [...h.matchAll(/ /g)].map((m) => m.index ?? 0);
			if (!anyLong || b.align[i] === 'l' || !sp.length) return run(h);
			const cut = sp.reduce((a, c) =>
				Math.abs(c - h.length / 2) < Math.abs(a - h.length / 2) ? c : a
			);
			return `\\begin{tabular}[b]{@{}${b.align[i]}@{}}${run(h.slice(0, cut))}\\\\${run(h.slice(cut + 1))}\\end{tabular}`;
		};
		const headRow = b.head.map(headCell).join(' & ') + ' \\\\';
		const tight = b.head.length > 8 ? '\\footnotesize\\setlength{\\tabcolsep}{3pt}\n' : '';
		return [
			tight + open,
			'\\toprule',
			headRow,
			'\\midrule',
			...b.rows.map(row),
			'\\bottomrule',
			`\\end{${env}}`
		].join('\n');
	};

	/**
	 * Blocks joined by newlines. A display belongs to the paragraph around it: no blank line before it, and none
	 * after it when the source continues the sentence (`cont`); otherwise a blank line ends the paragraph.
	 */
	const blocks = (bs: Block[]): string => {
		const parts = bs.map(one);
		let out = parts[0] ?? '';
		for (let i = 1; i < bs.length; i++) {
			const prev = bs[i - 1];
			if (bs[i].t === 'math' || (prev.t === 'math' && prev.cont))
				out = out.trimEnd() + '\n' + parts[i];
			else if (prev.t === 'math') out += '\n\n' + parts[i];
			else out += '\n' + parts[i];
		}
		return out;
	};

	const one = (b: Block): string => {
		switch (b.t) {
			case 'heading': {
				const lab = b.attrs.id ? `\\label{${b.attrs.id}}` : '';
				const star = b.attrs.classes.includes('unnumbered') ? '*' : '';
				// in the appendix, numbered sections are lettered by LaTeX's \appendix, as md.ts letters them
				// Extended Data starts on its own page, after every float of the main text
				const brk = b.attrs.id === 'extended' ? '\\clearpage\n' : '';
				if (b.level === 2) return `${brk}\\section${star}{${run(b.text)}}${lab}`;
				if (b.level === 3) return `\\subsection*{${run(b.text)}}${lab}`;
				return `\\paragraph{${run(b.text)}}${lab}`;
			}
			case 'para':
				return run(b.text) + '\n';
			case 'list': {
				const env = b.ordered ? 'enumerate' : 'itemize';
				return `\\begin{${env}}\n${b.items.map((it) => `\\item ${run(it)}`).join('\n')}\n\\end{${env}}`;
			}
			case 'math': {
				const r = b.attrs.id ? num.ids.get(b.attrs.id) : undefined;
				const tex = mathTex(b.tex);
				// amsmath undoes the counter step of a \tag'ged equation, so hyperref would reuse one anchor name
				// (equation.1, or equation.<section>.1) and equation links would land on the wrong equation
				const anchor = r ? `\\renewcommand{\\theHequation}{${b.attrs.id}}\n` : '';
				return r
					? `${anchor}\\begin{equation}\n${tex}\n\\tag{${r.n}}\\label{${b.attrs.id}}\n\\end{equation}`
					: `\\[\n${tex}\n\\]`;
			}
			case 'code':
				// verbatim prints commands literally, so a substituted character would show as its command
				if (pdf && NON_ASCII.test(b.text))
					throw new Error(`pdflatex: non-ASCII character in a code block: ${b.text.slice(0, 60)}`);
				return `\\begin{verbatim}\n${b.text}\n\\end{verbatim}`;
			case 'table':
				return `{\\small\n${table(b)}\n}`;
			case 'div':
				return div(b.kind, b.attrs, b.children);
		}
	};

	const div = (kind: string, a: Attrs, children: Block[]): string => {
		const lab = a.id ? `\\phantomsection\\label{${a.id}}` : '';
		switch (kind) {
			case 'abstract':
				return `\\begin{abstract}\n${blocks(children)}\n\\end{abstract}`;
			case 'figure': {
				const pdf = a.kv.pdf;
				const img =
					pdf && opt.figExists(pdf)
						? `\\includegraphics[width=\\linewidth]{${opt.figDir}${pdf}}`
						: `\\fbox{\\parbox[c][6cm][c]{0.9\\linewidth}{\\centering ${escText(pdf ?? 'figure')} is being drawn}}`;
				const [first, ...rest] = children;
				const cap =
					first && first.t === 'para'
						? `\\textbf{${label(a.id)} $|$} ${run(first.text)} ${blocks(rest)}`
						: blocks(children);
				// an Extended Data figure stays below its heading rather than floating to the top of the page
				const where = a.classes.includes('extended') ? '!hb' : 'tbp';
				return `\\begin{figure}[${where}]\n\\centering\n${img}\n\\caption*{${cap.trim()}}${lab}\n\\end{figure}`;
			}
			case 'table': {
				const [first, ...rest] = children;
				const hasCap = first && first.t === 'para';
				const cap = hasCap ? `\\caption*{\\textbf{${label(a.id)} $|$} ${run(first.text)}}` : '';
				return `\\begin{table}[${a.classes.includes('extended') ? '!htbp' : 'tbp'}]\n\\centering\n${cap}${lab}\n${blocks(hasCap ? rest : children)}\n\\end{table}`;
			}
			case 'theorem':
			case 'proposition':
			case 'lemma':
			case 'definition': {
				const name = a.kv.title ? ` (${run(a.kv.title)})` : '';
				const lean = a.kv.lean
					? `\n\n\\noindent{\\small\\raggedright\\textsc{Lean:} ${a.kv.lean
							.split(/,\s*/)
							.map(escCode)
							.join(', ')}\\par}`
					: '';
				return `\\begin{thmbox}${lab}\\noindent\\textbf{${label(a.kv.restate ?? a.id)}}${name}.\\enspace\\itshape ${blocks(children).trim()}\\upshape${lean}\n\\end{thmbox}`;
			}
			case 'proof':
				return `\\begin{proofbox}\\noindent\\textit{${a.kv.title ? run(a.kv.title) : 'Proof sketch'}.}\\enspace ${blocks(children).trim()}\n\\end{proofbox}`;
			case 'status':
				return `\\begin{statusbox}\\noindent\\textbf{${escText(a.kv.label ?? 'Today')}.}\\enspace ${blocks(children).trim()}\n\\end{statusbox}`;
			case 'appendix':
				return `\\appendix\n${blocks(children)}`;
			default:
				return blocks(children);
		}
	};

	const body = blocks(doc.blocks);
	const keys = [...num.cites.entries()].sort((x, y) => x[1] - y[1]);
	const items = keys
		.map(([k]) => {
			const e = bib.get(k);
			if (!e) return `\\bibitem{${k}} ${escText(k)} (missing).`;
			const f = format(e);
			const link = f.href ? ` \\url{${f.href}}` : '';
			const venue = f.venue.replace(/&/g, '\\&');
			return `\\bibitem{${k}} ${f.authors.replace(/&/g, '\\&')} ${f.title}. ${venue} (${f.year}).${link}`;
		})
		.join('\n');
	if (missing.length) throw new Error(`unknown references: ${missing.join(', ')}`);

	const m = doc.meta;
	// pdfLaTeX: \pdfoutput=1 in the first lines tells arXiv to produce PDF; fonts and input encoding replace fontspec,
	// and the Unicode table is applied to the text below instead of declared in the preamble
	const head = pdf
		? `% Generated by paper/build-tex.ts --arxiv from paper/paper.md and paper/refs.bib for pdfLaTeX. Do not edit by hand.
\\pdfoutput=1`
		: `% Generated by paper/build-tex.ts from paper/paper.md and paper/refs.bib. Do not edit by hand.`;
	const fonts = pdf
		? `\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{lmodern}
\\usepackage{textcomp}`
		: `\\usepackage{fontspec}`;
	const out = `${head}
\\documentclass[10pt]{article}
\\usepackage[a4paper,margin=22mm]{geometry}
${fonts}
\\usepackage{amsmath,amssymb}
\\usepackage{graphicx,booktabs,tabularx,array,xcolor}
\\usepackage[font=small,labelformat=empty,justification=raggedright,singlelinecheck=false]{caption}
${pdf ? '' : '\\usepackage{newunicodechar}\n'}\\usepackage{microtype}
\\usepackage{enumitem}
\\usepackage[numbers,super,sort&compress]{natbib}
\\usepackage[hidelinks]{hyperref}
\\usepackage{url}
\\setlist{nosep,leftmargin=*}
\\definecolor{ink}{HTML}{1F2A44}
\\definecolor{inksoft}{HTML}{4C5B78}
\\definecolor{rose}{HTML}{D6336C}
\\definecolor{tint}{HTML}{F2F5FB}
\\newenvironment{thmbox}{\\par\\medskip\\begingroup\\leftskip=1em\\rightskip=1em}{\\par\\endgroup\\medskip}
\\newenvironment{proofbox}{\\par\\smallskip\\begingroup\\small\\leftskip=1em\\rightskip=1em}{\\par\\endgroup\\medskip}
\\newenvironment{statusbox}{\\par\\smallskip\\begingroup\\small\\color{inksoft}\\leftskip=1em}{\\par\\endgroup\\smallskip}
${pdf ? '' : UNICODE + '\n'}\\setlength{\\parskip}{0.35em}
% figures: a figure-only page starts at the top, not centred; text pages may carry larger floats
\\makeatletter\\setlength{\\@fptop}{0pt}\\setlength{\\@fpsep}{12pt}\\setlength{\\@fpbot}{0pt plus 1fil}\\makeatother
\\renewcommand{\\topfraction}{0.9}\\renewcommand{\\textfraction}{0.08}\\renewcommand{\\floatpagefraction}{0.75}
\\setlength{\\parindent}{0pt}
\\renewcommand{\\bibnumfmt}[1]{#1.}
\\title{${((t: string) => {
		const i = t.indexOf(': ');
		return i > 0 ? `${run(t.slice(0, i + 1))}\\\\[0.35em]{\\Large ${run(t.slice(i + 2))}}` : run(t);
	})(m.title ?? '')}}
\\author{${run(m.authors ?? '')}${m.affiliation ? `\\\\ {\\small ${run(m.affiliation)}}` : ''}}
\\date{${run(m.date ?? '')}}
\\begin{document}
\\maketitle
${body}
\\section*{References}
\\begingroup\\small
\\renewcommand{\\refname}{}
\\vspace{-2.5em}
\\begin{thebibliography}{${keys.length}}
\\setlength{\\itemsep}{1pt}
${items}
\\end{thebibliography}
\\endgroup
\\end{document}
`;
	return pdf ? asciiText(out) : out;
}

/** text-mode characters the Latin Modern text font lacks or sets badly, mapped once in the preamble */
const TEXT_CHARS = [
	['×', '\\ensuremath{\\times}'],
	['−', '\\ensuremath{-}'],
	['≤', '\\ensuremath{\\le}'],
	['≥', '\\ensuremath{\\ge}'],
	['≠', '\\ensuremath{\\neq}'],
	['≈', '\\ensuremath{\\approx}'],
	['±', '\\ensuremath{\\pm}'],
	['⇔', '\\ensuremath{\\Leftrightarrow}'],
	['⇒', '\\ensuremath{\\Rightarrow}'],
	['→', '\\ensuremath{\\to}'],
	['←', '\\ensuremath{\\leftarrow}'],
	['√', '\\ensuremath{\\surd}'],
	['✓', '\\ensuremath{\\checkmark}'],
	['·', '\\ensuremath{\\cdot}'],
	['÷', '\\ensuremath{\\div}'],
	['∈', '\\ensuremath{\\in}'],
	['τ', '\\ensuremath{\\tau}'],
	['δ', '\\ensuremath{\\delta}'],
	['Π', '\\ensuremath{\\Pi}'],
	['∞', '\\ensuremath{\\infty}'],
	['⌊', '\\ensuremath{\\lfloor}'],
	['⌋', '\\ensuremath{\\rfloor}'],
	['⟨', '\\ensuremath{\\langle}'],
	['⟩', '\\ensuremath{\\rangle}'],
	['…', '\\ldots{}'],
	['²', '\\textsuperscript{2}'],
	['³', '\\textsuperscript{3}'],
	['⁴', '\\textsuperscript{4}'],
	['⁵', '\\textsuperscript{5}'],
	['¹', '\\textsuperscript{1}'],
	['⁰', '\\textsuperscript{0}'],
	['₁', '\\textsubscript{1}'],
	['₂', '\\textsubscript{2}'],
	[' ', '\\,'],
	[' ', '\\,']
];
const UNICODE = TEXT_CHARS.map(([c, r]) => `\\newunicodechar{${c}}{${r}}`).join('\n');

/**
 * pdfLaTeX only. XeTeX sets these from the font; pdfLaTeX gets commands, so main.tex is plain ASCII and its result
 * does not depend on the installed inputenc tables.
 */
const GREEK: [string, string][] = [
	['α', 'alpha'],
	['β', 'beta'],
	['γ', 'gamma'],
	['δ', 'delta'],
	['ε', 'varepsilon'],
	['ζ', 'zeta'],
	['η', 'eta'],
	['θ', 'theta'],
	['κ', 'kappa'],
	['λ', 'lambda'],
	['μ', 'mu'],
	['ν', 'nu'],
	['ξ', 'xi'],
	['π', 'pi'],
	['ρ', 'rho'],
	['σ', 'sigma'],
	['τ', 'tau'],
	['φ', 'varphi'],
	['χ', 'chi'],
	['ψ', 'psi'],
	['ω', 'omega'],
	['Γ', 'Gamma'],
	['Δ', 'Delta'],
	['Θ', 'Theta'],
	['Λ', 'Lambda'],
	['Π', 'Pi'],
	['Σ', 'Sigma'],
	['Φ', 'Phi'],
	['Ψ', 'Psi'],
	['Ω', 'Omega']
];
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const SUB = '₀₁₂₃₄₅₆₇₈₉';
const TEXT_EXTRA: [string, string][] = [
	['–', '\\textendash{}'],
	['—', '\\textemdash{}'],
	['‘', '\\textquoteleft{}'],
	['’', '\\textquoteright{}'],
	['“', '\\textquotedblleft{}'],
	['”', '\\textquotedblright{}'],
	[' ', '~'],
	['°', '\\textdegree{}'],
	['§', '\\S{}'],
	['†', '\\textdagger{}'],
	['∑', '\\ensuremath{\\sum}'],
	['∏', '\\ensuremath{\\prod}'],
	['∀', '\\ensuremath{\\forall}'],
	['∃', '\\ensuremath{\\exists}'],
	['⊆', '\\ensuremath{\\subseteq}'],
	['∪', '\\ensuremath{\\cup}'],
	['∩', '\\ensuremath{\\cap}'],
	['ℝ', '\\ensuremath{\\mathbb{R}}'],
	['ℕ', '\\ensuremath{\\mathbb{N}}'],
	['ℤ', '\\ensuremath{\\mathbb{Z}}'],
	['⊥', '\\ensuremath{\\bot}'],
	['↦', '\\ensuremath{\\mapsto}'],
	['⟨', '\\ensuremath{\\langle}'],
	['⟩', '\\ensuremath{\\rangle}'],
	...GREEK.map(([c, n]): [string, string] => [c, `\\ensuremath{\\${n}}`]),
	...[...SUP].map((c, i): [string, string] => [c, `\\textsuperscript{${i}}`]),
	...[...SUB].map((c, i): [string, string] => [c, `\\textsubscript{${i}}`])
];
const TEXT_MAP = new Map<string, string>([...TEXT_EXTRA, ...(TEXT_CHARS as [string, string][])]);

/** inside math the same characters become math commands; a trailing space ends the control word */
const MATH_MAP = new Map<string, string>([
	['−', '-'],
	['–', '-'],
	['×', '\\times '],
	['·', '\\cdot '],
	['÷', '\\div '],
	['≤', '\\le '],
	['≥', '\\ge '],
	['≠', '\\neq '],
	['≈', '\\approx '],
	['±', '\\pm '],
	['⇔', '\\Leftrightarrow '],
	['⇒', '\\Rightarrow '],
	['→', '\\to '],
	['←', '\\leftarrow '],
	['↦', '\\mapsto '],
	['√', '\\surd '],
	['∈', '\\in '],
	['∞', '\\infty '],
	['⌊', '\\lfloor '],
	['⌋', '\\rfloor '],
	['⟨', '\\langle '],
	['⟩', '\\rangle '],
	['…', '\\ldots '],
	['∑', '\\sum '],
	['∏', '\\prod '],
	['⊆', '\\subseteq '],
	['ℝ', '\\mathbb{R}'],
	['ℕ', '\\mathbb{N}'],
	['ℤ', '\\mathbb{Z}'],
	[' ', '\\,'],
	[' ', '\\,'],
	[' ', '~'],
	...GREEK.map(([c, n]): [string, string] => [c, `\\${n} `]),
	...[...SUP].map((c, i): [string, string] => [c, `^{${i}}`]),
	...[...SUB].map((c, i): [string, string] => [c, `_{${i}}`])
]);

/** combining accents to LaTeX accent commands, for letters such as the ö in Gröbner */
const ACCENT_CMD: Record<string, string> = {
	'́': "'",
	'̀': '`',
	'̈': '"',
	'̂': '^',
	'̃': '~',
	'̄': '=',
	'̇': '.',
	'̊': 'r',
	'̋': 'H',
	'̌': 'v',
	'̆': 'u',
	'̧': 'c',
	'̨': 'k'
};
const LETTER_CMD: Record<string, string> = {
	ß: '{\\ss}',
	ø: '{\\o}',
	Ø: '{\\O}',
	ł: '{\\l}',
	Ł: '{\\L}',
	æ: '{\\ae}',
	Æ: '{\\AE}',
	œ: '{\\oe}',
	Œ: '{\\OE}',
	ı: '{\\i}'
};

const NON_ASCII = /[\u0080-\u{10FFFF}]/u;

const unmapped = (c: string, s: string, i: number) =>
	new Error(
		`pdflatex: no LaTeX command for ${JSON.stringify(c)} (U+${c
			.codePointAt(0)!
			.toString(16)
			.toUpperCase()
			.padStart(
				4,
				'0'
			)}) near "${s.slice(Math.max(0, i - 40), i + 40)}"; add it to TEXT_EXTRA or MATH_MAP in src/lib/paper/tex.ts`
	);

/** math source to ASCII; throws on a character with no math command */
function asciiMath(s: string): string {
	let out = '';
	let i = 0;
	for (const c of s) {
		if (!NON_ASCII.test(c)) out += c;
		else if (MATH_MAP.has(c)) out += MATH_MAP.get(c);
		else throw unmapped(c, s, i);
		i += c.length;
	}
	return out;
}

/** the whole document to ASCII (math is already ASCII); throws on a character with no command */
function asciiText(s: string): string {
	let out = '';
	let i = 0;
	for (const c of s) {
		if (!NON_ASCII.test(c)) out += c;
		else if (TEXT_MAP.has(c)) out += TEXT_MAP.get(c);
		else if (LETTER_CMD[c]) out += LETTER_CMD[c];
		else {
			const [base, ...marks] = c.normalize('NFD');
			if (!/^[A-Za-z]$/.test(base) || marks.length !== 1 || !ACCENT_CMD[marks[0]])
				throw unmapped(c, s, i);
			out += `\\${ACCENT_CMD[marks[0]]}{${base === 'i' ? '\\i' : base}}`;
		}
		i += c.length;
	}
	return out;
}
