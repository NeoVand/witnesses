<script lang="ts">
	/**
	 * A document rendered from Markdown at build time (src/lib/paper/): the paper and the tutorial. Three columns on
	 * wide screens: labels in the left margin, the text, and room on the right that figures and sidenotes use. The
	 * title breaks after its colon, the subtitle smaller on its own line, as on the PDFs' title pages.
	 */
	import '@fontsource-variable/source-serif-4/opsz.css';
	import '@fontsource-variable/source-serif-4/opsz-italic.css';
	import type { Snippet } from 'svelte';
	import { splitTitle } from '#lib/site.ts';

	let {
		meta,
		html,
		lede = null,
		toc = null,
		links
	}: {
		meta: Record<string, string>;
		html: string;
		/** a short introduction under the byline, as HTML */
		lede?: string | null;
		/** a table of contents (the tutorial's chapters) */
		toc?: { id: string; n: string; title: string }[] | null;
		links: Snippet;
	} = $props();

	const title = $derived(splitTitle(meta.title ?? ''));
	const byline = $derived((meta.authors ?? '').replace(/'([^']*)'/g, '‘$1’'));
</script>

<main class="paper">
	<header class="front">
		{#if meta.date}<p class="kicker">{meta.date}</p>{/if}
		<h1>
			<span class="t-main">{title.main}</span>
			{#if title.sub}<span class="t-sub">{title.sub}</span>{/if}
		</h1>
		<p class="byline">
			{byline}{#if meta.affiliation}<span class="affil">, {meta.affiliation}</span>{/if}
		</p>
		<p class="links">{@render links()}</p>
		{#if lede}
			<!-- eslint-disable-next-line svelte/no-at-html-tags -- the document's own lede, rendered at build time -->
			<p class="lede">{@html lede}</p>
		{/if}
	</header>

	{#if toc?.length}
		<nav class="toc" aria-labelledby="toc-h">
			<h2 id="toc-h">Contents</h2>
			<ol>
				{#each toc as t (t.id)}
					<li><a href="#{t.id}"><span class="n">{t.n}</span>{t.title}</a></li>
				{/each}
			</ol>
		</nav>
	{/if}

	<article class="body">
		<!-- eslint-disable-next-line svelte/no-at-html-tags -- built at prerender time from the document's Markdown -->
		{@html html}
	</article>
</main>

<style>
	.paper {
		--left: 200px;
		--text: 680px;
		--right: 200px;
		--gap: 40px;
		--serif: 'Source Serif 4 Variable', 'KaTeX_Main', Georgia, serif;
		max-width: calc(var(--left) + var(--text) + var(--right) + 2 * var(--gap) + 64px);
		margin: 0 auto;
		padding: 72px 32px 140px;
		color: var(--color-ink);
	}

	/* ---- front matter ------------------------------------------------------------------------------------------ */
	.front {
		max-width: calc(var(--text) + 2 * var(--gap));
		margin: 0 auto;
		text-align: center;
	}
	.front h1 {
		margin-left: auto;
		margin-right: auto;
	}
	.front .links {
		justify-content: center;
	}
	.kicker {
		font-size: var(--text-xs);
		letter-spacing: 0.02em;
		color: var(--color-ink-soft);
	}
	h1 {
		margin-top: 14px;
		font-family: var(--serif);
		font-weight: 400;
		letter-spacing: -0.012em;
		text-wrap: balance;
	}
	.t-main {
		display: block;
		font-size: 36px;
		line-height: 1.15;
	}
	.t-sub {
		display: block;
		margin-top: 10px;
		font-size: 24px;
		line-height: 1.25;
		color: var(--color-ink);
	}
	.body :global(.cite-name) {
		white-space: nowrap;
	}
	.byline {
		margin-top: 18px;
		font-size: var(--text-ui);
		color: var(--color-ink-soft);
	}
	.links {
		margin-top: 6px;
		display: flex;
		flex-wrap: wrap;
		gap: 4px 10px;
		font-size: var(--text-ui);
		color: var(--color-ink-soft);
	}
	.links :global(a) {
		color: var(--color-ink);
	}

	/* ---- the grid: every block sits in the text column unless it asks for more ---------------------------------- */
	.body {
		display: grid;
		grid-template-columns:
			[left-start] var(--left) [left-end] var(--gap) [text-start] minmax(0, var(--text))
			[text-end] var(--gap) [right-start] var(--right) [right-end];
		margin-top: 48px;
		font-family: var(--serif);
		font-size: 18px;
		line-height: 1.62;
		font-optical-sizing: auto;
		font-feature-settings:
			'kern',
			'liga',
			'onum' 0;
	}
	.body > :global(*) {
		grid-column: text;
	}
	.body :global(p) {
		margin: 0 0 1em;
		hyphens: auto;
	}
	.body :global(strong) {
		font-weight: 600;
	}
	.body :global(code) {
		font-family: var(--font-mono);
		font-size: 0.8em;
		color: var(--color-ink-soft);
	}
	.body :global(a) {
		text-decoration-color: var(--color-mist);
	}
	.body :global(ol),
	.body :global(ul) {
		margin: 0 0 1em;
		padding-left: 1.3em;
		list-style: decimal;
	}
	.body :global(ul) {
		list-style: disc;
	}
	.body :global(li) {
		margin-bottom: 0.45em;
	}
	.body :global(li::marker) {
		font-family: var(--font-sans);
		font-size: 0.8em;
		color: var(--color-ink-soft);
	}
	.body :global(.dmath) {
		margin: 0.4em 0 1.2em;
		overflow-x: auto;
		overflow-y: hidden;
	}

	/* abstract: its label in the margin */
	.body :global(.abstract) {
		display: grid;
		grid-column: left-start / text-end;
		grid-template-columns: var(--left) minmax(0, var(--text));
		column-gap: var(--gap);
		margin-bottom: 8px;
	}
	.body :global(.abs-h) {
		margin: 6px 0 0;
		text-align: right;
		font-family: var(--font-sans);
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--color-ink-soft);
	}
	.body :global(.abstract p) {
		grid-column: 2;
		font-size: 17px;
		line-height: 1.6;
	}

	/* section heads: the number in the left margin */
	.body :global(h2.sec) {
		display: grid;
		grid-column: left-start / text-end;
		grid-template-columns: var(--left) minmax(0, var(--text));
		column-gap: var(--gap);
		align-items: baseline;
		margin: 56px 0 14px;
		font-family: var(--serif);
		font-size: 25px;
		font-weight: 500;
		line-height: 1.25;
		letter-spacing: -0.005em;
	}
	.body :global(.sec-n) {
		text-align: right;
		font-family: var(--font-sans);
		font-size: 15px;
		font-weight: 500;
		color: var(--color-rose);
	}
	.body :global(h3) {
		margin: 30px 0 8px;
		font-family: var(--font-sans);
		font-size: 15px;
		font-weight: 600;
		letter-spacing: 0.005em;
	}

	/* citations and cross-references */
	.body :global(sup.cite) {
		white-space: nowrap;
		font-family: var(--font-sans);
		font-size: 0.62em;
		line-height: 0;
		margin-left: 1px;
	}
	.body :global(sup.cite a),
	.body :global(a.xref) {
		color: var(--color-cobalt);
		text-decoration: none;
	}
	.body :global(sup.cite a:hover),
	.body :global(a.xref:hover) {
		color: var(--color-rose);
	}

	/* theorems: the label in the margin, the statement in italic, the Lean names quiet beneath */
	.body :global(.thm) {
		display: grid;
		grid-column: left-start / text-end;
		grid-template-columns: var(--left) minmax(0, var(--text));
		column-gap: var(--gap);
		margin: 10px 0 18px;
	}
	.body :global(.thm > *) {
		grid-column: 2;
	}
	.body :global(.thm-head) {
		grid-column: 1;
		grid-row: 1 / span 3;
		margin: 3px 0 0;
		text-align: right;
		font-family: var(--font-sans);
		font-size: var(--text-sm);
		line-height: 1.4;
	}
	.body :global(.thm-label) {
		display: block;
		font-weight: 600;
		color: var(--color-rose);
	}
	.body :global(.thm-name) {
		display: block;
		color: var(--color-ink-soft);
	}
	.body :global(.thm p:not(.thm-head):not(.lean)) {
		font-style: italic;
		margin-bottom: 0.5em;
	}
	.body :global(.lean) {
		font-family: var(--font-sans);
		font-size: var(--text-xs);
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.body :global(.lean code) {
		font-size: 11.5px;
	}
	.body :global(.lean-k) {
		font-weight: 600;
		margin-right: 4px;
	}
	.body :global(.proof) {
		margin: 0 0 22px;
		font-size: 16px;
		line-height: 1.6;
		color: var(--color-ink-soft);
	}
	.body :global(.proof-head) {
		display: inline;
		font-style: italic;
		margin-right: 0.4em;
	}
	.body :global(.proof-head + p) {
		display: inline;
	}

	/* what is still waiting: a quiet note with its label in the margin */
	.body :global(.status) {
		display: grid;
		grid-column: left-start / text-end;
		grid-template-columns: var(--left) minmax(0, var(--text));
		column-gap: var(--gap);
		margin: 4px 0 22px;
	}
	.body :global(.status-k) {
		padding-top: 4px;
		text-align: right;
		font-family: var(--font-sans);
		font-size: var(--text-xs);
		font-weight: 600;
		letter-spacing: 0.04em;
		text-transform: uppercase;
		color: var(--color-violet);
	}
	.body :global(.status p) {
		grid-column: 2;
		padding-left: 16px;
		background: linear-gradient(var(--color-violet), var(--color-violet)) left / 2px 100% no-repeat;
		font-size: 16.5px;
		color: var(--color-ink-soft);
	}

	/* figures: centred across both margins; captions centred under them at the text width */
	.body :global(figure.fig) {
		grid-column: left-start / right-end;
		margin: 24px 0 36px;
	}
	.body :global(figure.fig img) {
		max-width: 960px;
		margin-left: auto;
		margin-right: auto;
	}
	.body :global(figure.fig figcaption) {
		max-width: var(--text);
		margin-left: auto;
		margin-right: auto;
	}
	.body :global(figure.fig img) {
		display: block;
		width: 100%;
		height: auto;
	}
	.body :global(.fig-open) {
		display: block;
		cursor: zoom-in;
	}
	/* a figure still being drawn: one quiet line where it will go, never a broken image */
	.body :global(.fig-missing) {
		display: grid;
		place-items: center;
		height: 120px;
		background: var(--color-tint);
		font-family: var(--font-sans);
		font-size: var(--text-sm);
		color: var(--color-ink-soft);
	}
	.body :global(figcaption),
	.body :global(.tbl-cap) {
		max-width: var(--text);
		margin-top: 14px;
		font-family: var(--font-sans);
		font-size: 13.5px;
		line-height: 1.55;
		color: var(--color-ink-soft);
	}
	.body :global(figcaption strong),
	.body :global(.tbl-cap strong) {
		color: var(--color-ink);
	}
	.body :global(.fig-label) {
		font-weight: 600;
		color: var(--color-ink);
	}
	.body :global(figcaption .katex),
	.body :global(.tbl-cap .katex) {
		font-size: 1.02em;
	}

	/* tables: three rules, tabular figures */
	.body :global(.tbl) {
		grid-column: left-start / right-end;
		justify-self: center;
		max-width: 100%;
		margin: 18px 0 32px;
	}
	.body :global(.tbl-cap) {
		margin: 0 0 10px;
	}
	.body :global(.tbl-wrap) {
		overflow-x: auto;
	}
	.body :global(table) {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--font-sans);
		font-size: 13px;
		line-height: 1.4;
		font-variant-numeric: tabular-nums;
		border-top: 1px solid var(--color-ink);
		border-bottom: 1px solid var(--color-ink);
	}
	.body :global(th) {
		padding: 8px 10px 6px 0;
		text-align: left;
		font-weight: 600;
		border-bottom: 1px solid var(--color-mist);
		vertical-align: bottom;
	}
	.body :global(td) {
		padding: 6px 10px 6px 0;
		vertical-align: top;
		border-bottom: 1px solid var(--color-tint);
	}
	.body :global(th.r),
	.body :global(td.r) {
		text-align: right;
	}
	.body :global(th.c),
	.body :global(td.c) {
		text-align: center;
	}
	.body :global(table code) {
		font-size: 12px;
	}

	/* references */
	.body :global(ol.refs) {
		grid-column: text;
		padding-left: 2.2em;
		font-family: var(--font-sans);
		font-size: 13px;
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.body :global(ol.refs li) {
		margin-bottom: 0.5em;
		scroll-margin-top: 80px;
	}
	.body :global(ol.refs li:target) {
		color: var(--color-ink);
		background: var(--color-tint);
	}
	.body :global(.ref-link) {
		word-break: break-all;
		color: var(--color-blue-grey);
	}
	.body :global(h2[id]),
	.body :global(figure[id]),
	.body :global(.tbl[id]),
	.body :global(.thm[id]) {
		scroll-margin-top: 72px;
	}

	/* ---- a document's lede and contents (the tutorial) ------------------------------------------------------------- */
	.lede {
		max-width: 62ch;
		margin: 22px auto 0;
		text-align: left;
		font-size: 16px;
		line-height: 1.6;
		color: var(--color-ink-soft);
	}
	.toc {
		max-width: calc(var(--text) + 2 * var(--gap));
		margin: 40px auto 0;
		font-size: var(--text-sm);
	}
	.toc h2 {
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--color-ink-soft);
	}
	.toc ol {
		columns: 2;
		column-gap: 40px;
		margin-top: 8px;
		padding: 0;
		list-style: none;
	}
	.toc li {
		break-inside: avoid;
		padding: 3px 0;
	}
	.toc .n {
		display: inline-block;
		min-width: 1.6em;
		color: var(--color-rose);
		font-variant-numeric: tabular-nums;
	}
	.toc a {
		color: var(--color-ink);
		text-decoration: none;
	}
	.toc a:hover {
		text-decoration: underline;
	}

	/* asides: an analogy or a worked example beside the argument, indented under a quiet rule */
	.body :global(.aside) {
		margin: 6px 0 24px;
		padding: 2px 0 2px 18px;
		border-left: 2px solid var(--color-mist);
		font-size: 16px;
		line-height: 1.6;
		color: var(--color-ink-soft);
	}
	.body :global(.aside p:last-child) {
		margin-bottom: 0;
	}
	.body :global(.aside-t) {
		margin-bottom: 4px;
		font-family: var(--font-sans);
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--color-rose);
	}

	/* sidenotes: a numbered mark in the text; the note in the right margin on wide screens, under its line otherwise */
	.body :global(.sn-mark) {
		font-family: var(--font-sans);
		font-size: 0.62em;
		vertical-align: super;
		line-height: 0;
		color: var(--color-cobalt);
		cursor: pointer;
	}
	.body :global(.sn-toggle) {
		display: none;
	}
	.body :global(.sidenote) {
		float: right;
		clear: right;
		width: var(--right);
		margin-right: calc(-1 * (var(--right) + var(--gap)));
		font-family: var(--font-sans);
		font-size: var(--text-xs);
		line-height: 1.5;
		color: var(--color-ink-soft);
	}
	.body :global(.sn-n) {
		color: var(--color-cobalt);
	}

	/* ---- narrow screens: one column, labels above what they label ------------------------------------------- */
	@media (max-width: 1080px) {
		.paper {
			--left: 0px;
			--right: 0px;
			--gap: 0px;
			max-width: 720px;
			padding: 40px 16px 96px;
		}
		.front {
			margin: 0 auto;
		}
		.t-main {
			font-size: 29px;
		}
		.t-sub {
			font-size: 20px;
		}
		.body {
			display: block;
			font-size: 17px;
		}
		.body :global(.abstract),
		.body :global(h2.sec),
		.body :global(.thm),
		.body :global(.status) {
			display: block;
		}
		.body :global(.abs-h),
		.body :global(.thm-head),
		.body :global(.status-k) {
			text-align: left;
			margin-bottom: 6px;
		}
		.body :global(.sec-n) {
			margin-right: 10px;
		}
		/* an unnumbered section (Methods, AI usage) starts at the margin, not after an empty number */
		.body :global(.sec-n:empty) {
			display: none;
		}
		.body :global(.thm-label),
		.body :global(.thm-name) {
			display: inline;
		}
		.body :global(.thm-name) {
			margin-left: 4px;
		}
		.body :global(.status-k) {
			display: block;
		}
		.body :global(h2.sec) {
			font-size: 22px;
			margin-top: 44px;
		}
		.body :global(ol.refs) {
			padding-left: 1.8em;
		}
		.toc ol {
			columns: 1;
		}
		.body :global(.sidenote) {
			display: none;
			float: none;
			width: auto;
			margin: 6px 0;
		}
		.body :global(.sn-toggle:checked + .sidenote) {
			display: block;
		}
	}
	/* a phone: display maths a little smaller, so most equations fit the column; a wider one scrolls beside its number */
	@media (max-width: 480px) {
		.body :global(.dmath) {
			font-size: 0.8em;
		}
	}
</style>
