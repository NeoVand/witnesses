/**
 * Writes paper/tex/main.tex from paper/paper.md and paper/refs.bib, the same source the web page renders.
 *
 *   node paper/build-tex.ts            # write main.tex
 *   node paper/build-tex.ts --pdf      # and build main.pdf with tectonic
 *   node paper/build-tex.ts --arxiv    # instead: paper/arxiv/ for pdfLaTeX (main.tex, figures/, the tarball)
 *
 * Node runs this file directly (type stripping); it imports only dependency-free modules.
 */
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readFileSync,
	rmSync,
	statSync,
	writeFileSync
} from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseBib } from '../src/lib/paper/bib.ts';
import { parse, wordCount } from '../src/lib/paper/md.ts';
import { renderTex } from '../src/lib/paper/tex.ts';

const here = dirname(fileURLToPath(import.meta.url));
const md = readFileSync(join(here, 'paper.md'), 'utf8');
const bib = parseBib(readFileSync(join(here, 'refs.bib'), 'utf8'));

if (process.argv.includes('--arxiv')) {
	arxiv();
	process.exit(0);
}

const out = join(here, 'tex');
mkdirSync(out, { recursive: true });

const figDir = '../figures/out/';
const tex = renderTex(md, bib, {
	figDir,
	figExists: (f) => existsSync(join(here, 'figures', 'out', f))
});
writeFileSync(join(out, 'main.tex'), tex);
console.log(`main.tex written: ${tex.length} characters; main text ${wordCount(parse(md))} words`);

if (process.argv.includes('--pdf')) {
	const tectonic = existsSync('/opt/homebrew/bin/tectonic')
		? '/opt/homebrew/bin/tectonic'
		: 'tectonic';
	execFileSync(tectonic, ['--keep-logs', 'main.tex'], { cwd: out, stdio: 'inherit' });
	// the site serves the same PDF beside the web version
	copyFileSync(join(out, 'main.pdf'), join(out, '..', '..', 'static', 'attention-witnesses.pdf'));
	console.log(`main.pdf built in ${out}, copied to static/attention-witnesses.pdf`);
}

/**
 * paper/arxiv/: main.tex for pdfLaTeX (plain ASCII, no fontspec), the figure PDFs it includes, and
 * attention-witnesses-arxiv.tar.gz with exactly those files. The bibliography is inline (thebibliography), so arXiv needs no
 * BibTeX run and no .bbl. paper/arxiv/check_pdflatex.py compiles the tarball with TeX Live on Modal.
 */
function arxiv() {
	const dir = join(here, 'arxiv');
	const figs = join(dir, 'figures');
	rmSync(figs, { recursive: true, force: true });
	mkdirSync(figs, { recursive: true });
	const tex = renderTex(md, bib, {
		figDir: 'figures/',
		figExists: (f) => existsSync(join(here, 'figures', 'out', f)),
		engine: 'pdflatex'
	});
	const used = [...tex.matchAll(/\\includegraphics(?:\[[^\]]*\])?\{figures\/([^}]+)\}/g)].map(
		(m) => m[1]
	);
	for (const f of used) copyFileSync(join(here, 'figures', 'out', f), join(figs, f));
	writeFileSync(join(dir, 'main.tex'), tex);
	const files = ['main.tex', ...used.map((f) => `figures/${f}`)];
	// macOS tar would add AppleDouble (._*) files for extended attributes; arXiv would see them as sources
	const mac = process.platform === 'darwin' ? ['--no-mac-metadata'] : [];
	const tgz = join(dir, 'attention-witnesses-arxiv.tar.gz');
	execFileSync('tar', [...mac, '-czf', tgz, ...files], {
		cwd: dir,
		env: { ...process.env, COPYFILE_DISABLE: '1' }
	});
	console.log(
		`paper/arxiv: main.tex ${tex.length} characters (ASCII), ${used.length} figures; ` +
			`attention-witnesses-arxiv.tar.gz ${(statSync(tgz).size / 1024).toFixed(0)} KiB with ${files.join(', ')}`
	);
}
