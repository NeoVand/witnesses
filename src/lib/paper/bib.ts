/**
 * paper/refs.bib, read for both outputs. Entries are formatted in Nature's reference style (authors as "Surname, I.",
 * up to five names, else the first and "et al."; title; venue; year), with links to the arXiv, DOI or official page.
 * The HTML side decodes LaTeX accents to Unicode; the TeX side keeps the source.
 */

export interface Entry {
	type: string;
	key: string;
	f: Record<string, string>;
}

export function parseBib(src: string): Map<string, Entry> {
	const out = new Map<string, Entry>();
	const re = /@(\w+)\s*\{\s*([^,\s]+)\s*,/g;
	let m: RegExpExecArray | null;
	while ((m = re.exec(src))) {
		const type = m[1].toLowerCase();
		const key = m[2];
		let i = re.lastIndex;
		const f: Record<string, string> = {};
		// fields until the entry's closing brace
		while (i < src.length) {
			while (i < src.length && /[\s,]/.test(src[i])) i++;
			if (src[i] === '}') {
				i++;
				break;
			}
			const fm = /^([\w-]+)\s*=\s*/.exec(src.slice(i));
			if (!fm) throw new Error(`bib: cannot read a field of ${key} near ${src.slice(i, i + 40)}`);
			i += fm[0].length;
			let val: string;
			if (src[i] === '{') {
				let depth = 0;
				let j = i;
				for (; j < src.length; j++) {
					if (src[j] === '{') depth++;
					else if (src[j] === '}') {
						depth--;
						if (depth === 0) break;
					}
				}
				val = src.slice(i + 1, j);
				i = j + 1;
			} else if (src[i] === '"') {
				const j = src.indexOf('"', i + 1);
				val = src.slice(i + 1, j);
				i = j + 1;
			} else {
				const vm = /^[^,}\s]+/.exec(src.slice(i))!;
				val = vm[0];
				i += val.length;
			}
			f[fm[1].toLowerCase()] = val.replace(/\s+/g, ' ').trim();
		}
		re.lastIndex = i;
		out.set(key, { type, key, f });
	}
	return out;
}

const ACCENT: Record<string, string> = {
	"'": '\u0301',
	'`': '\u0300',
	'"': '\u0308',
	'^': '\u0302',
	'~': '\u0303',
	'=': '\u0304',
	'.': '\u0307',
	v: '\u030C',
	c: '\u0327',
	u: '\u0306',
	H: '\u030B',
	k: '\u0328'
};
const LETTER: Record<string, string> = {
	L: 'Ł',
	l: 'ł',
	ss: 'ß',
	o: 'ø',
	O: 'Ø',
	ae: 'æ',
	AE: 'Æ',
	aa: 'å',
	AA: 'Å',
	i: 'ı',
	oe: 'œ'
};

/** LaTeX text (as written in refs.bib) to plain Unicode */
export function detex(s: string): string {
	let t = s;
	t = t.replace(
		/\\([`'"^~=.])\s*\{?\\?([A-Za-z])\}?/g,
		(_, a, ch) => (ch === 'i' ? 'i' : ch) + ACCENT[a]
	);
	t = t.replace(/\\([vcuHk])\s*\{\\?([A-Za-z])\}/g, (_, a, ch) => ch + ACCENT[a]);
	t = t.replace(/\\([vcuHk])\s+([A-Za-z])/g, (_, a, ch) => ch + ACCENT[a]);
	t = t.replace(/\\(ss|ae|AE|aa|AA|oe|L|l|o|O|i)(?![A-Za-z])/g, (_, w) => LETTER[w]);
	t = t.replace(/\\&/g, '&').replace(/\\%/g, '%').replace(/\\_/g, '_').replace(/\\\$/g, '$');
	t = t.replace(/---/g, '—').replace(/--/g, '–').replace(/~/g, '\u00a0');
	t = t.replace(/\\textendash\b/g, '–').replace(/\\textemdash\b/g, '—');
	t = t.replace(/[{}]/g, '');
	return t.normalize('NFC');
}

interface Name {
	last: string;
	initials: string;
}

function splitNames(s: string): (Name | 'others')[] {
	return s.split(/\s+and\s+/).map((raw) => {
		const n = raw.trim();
		if (n === 'others') return 'others';
		if (/^\{.*\}$/.test(n)) return { last: n, initials: '' };
		let last: string;
		let first: string;
		if (n.includes(',')) {
			const [l, ...r] = n.split(',');
			last = l.trim();
			first = r.join(',').trim();
		} else {
			const parts = n.split(/\s+/);
			last = parts.pop()!;
			first = parts.join(' ');
		}
		const initials = first
			.split(/\s+/)
			.filter(Boolean)
			.map((w) =>
				w
					.split('-')
					.map((p) => {
						// keep a LaTeX-accented first letter whole: {\L}ukasz, {\'E}mile
						const acc = /^\{?(\\[^a-zA-Z]?\{?[A-Za-z]\}?|\\[A-Za-z]+)\}?/.exec(p);
						const ch = acc ? acc[0] : p.replace(/^[{]/, '')[0];
						return ch ? `${ch}.` : '';
					})
					.join('-')
			)
			.join(' ');
		return { last, initials };
	});
}

export interface Formatted {
	authors: string;
	title: string;
	venue: string;
	year: string;
	href: string | null;
}

/** Nature-style parts of a reference, in the source's TeX (pass through detex for the web) */
export function format(e: Entry): Formatted {
	const f = e.f;
	const names = splitNames(f.author ?? f.editor ?? '');
	const real = names.filter((n): n is Name => n !== 'others');
	const fmt = (n: Name) => (n.initials ? `${n.last}, ${n.initials}` : n.last);
	let authors: string;
	if (names.includes('others') || real.length > 5) authors = `${fmt(real[0])} et al.`;
	else if (real.length === 1) authors = fmt(real[0]);
	else authors = real.slice(0, -1).map(fmt).join(', ') + ' & ' + fmt(real[real.length - 1]);
	if (!/\.$/.test(authors)) authors += '.';
	const title = (f.title ?? '').replace(/\.$/, '');
	let venue: string;
	if (e.type === 'article') {
		venue = [
			f.journal,
			f.volume ? `\\textbf{${f.volume}}` : '',
			f.pages ? f.pages.replace('--', '–') : ''
		]
			.filter(Boolean)
			.join(', ');
		venue = venue.replace(/, \\textbf/, ' \\textbf');
	} else if (e.type === 'inproceedings' || e.type === 'incollection') {
		venue = `In ${f.booktitle ?? ''}${f.pages ? ', ' + f.pages.replace('--', '–') : ''}`;
	} else if (e.type === 'book') {
		venue = [f.edition ? `${f.edition} edn` : '', f.publisher].filter(Boolean).join(', ');
	} else if (e.type === 'phdthesis') {
		venue = `PhD thesis, ${f.school ?? ''}`;
	} else {
		venue = f.eprint ? `Preprint at arXiv:${f.eprint}` : (f.howpublished ?? f.note ?? '');
		if (!f.eprint && f.howpublished && f.note && !f.note.startsWith('Citation'))
			venue += `. ${f.note}`;
	}
	const href = f.doi
		? `https://doi.org/${f.doi}`
		: f.eprint
			? `https://arxiv.org/abs/${f.eprint}`
			: (f.url ?? null);
	return { authors, title, venue, year: f.year ?? '', href };
}

/** the authors as a sentence names them: "Saparov", "Clift and Murfet", "Saparov et al." (TeX source; detex for the web) */
export function citeName(e: Entry): string {
	const names = splitNames(e.f.author ?? e.f.editor ?? '');
	const real = names.filter((n): n is Name => n !== 'others');
	const last = (n: Name) => n.last.replace(/^\{(.*)\}$/, '$1');
	if (!real.length) return e.key;
	if (names.includes('others') || real.length > 2) return `${last(real[0])} et al.`;
	if (real.length === 2) return `${last(real[0])} and ${last(real[1])}`;
	return last(real[0]);
}
