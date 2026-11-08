<script lang="ts">
	import '@fontsource-variable/source-serif-4/opsz.css';
	import '@fontsource-variable/source-serif-4/opsz-italic.css';
	import { asset, resolve } from '$app/paths';
	import Icon from '#lib/ui/Icon.svelte';
	import Meta from '#lib/ui/Meta.svelte';
	import { Book, Film, Github, Pdf, Play, School } from '#lib/ui/icons.ts';
	import { CODE_URL, DESCRIPTION, FILM_URL, PDF, splitTitle } from '#lib/site.ts';
	import Hero from '#lib/search/Hero.svelte';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
	const byline = $derived(data.authors.replace(/'([^']*)'/g, '‘$1’'));
	const title = $derived(splitTitle(data.title));
</script>

<Meta title={data.title} description={DESCRIPTION} />

<main class="landing">
	<section class="top">
		<header class="intro">
			<h1>
				<span class="t-main">{title.main}</span>
				{#if title.sub}<span class="t-sub">{title.sub}</span>{/if}
			</h1>
			<p class="idea">
				Each layer we study computes the partition function of a family of witnesses, the objects a
				task’s answer sums over (tokens, pairs, walks, subsets), and reads it at a temperature.
			</p>
			<p class="byline">{byline}{data.affiliation ? ` · ${data.affiliation}` : ''}</p>
			<nav class="links" aria-label="The paper and the demo">
				<a class="ic" href={resolve('/paper')}><Icon icon={Book} size={17} />Paper</a>
				<a class="ic" href={asset(PDF)}><Icon icon={Pdf} size={17} />PDF</a>
				<a class="ic demo" href={resolve('/demo')}><Icon icon={Play} size={17} />Demo</a>
				<a class="ic" href={resolve('/tutorial')}><Icon icon={School} size={17} />Tutorial</a>
				{#if FILM_URL}
					<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- an external URL, set in site.ts -->
					<a class="ic" href={FILM_URL}><Icon icon={Film} size={17} />Film</a>
				{/if}
				{#if CODE_URL}
					<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- an external URL, set in site.ts -->
					<a class="ic" href={CODE_URL}><Icon icon={Github} size={17} />Code</a>
				{/if}
			</nav>
		</header>
		<figure class="hero">
			<Hero layout={data.hero.layout} question={data.hero.question} frames={data.hero.frames} />
			<figcaption>
				A 250-vertex test question of the search benchmark (lookahead {data.hero.lookahead}), lit by
				the walks to the goal of one propagator layer trained on graphs of 49 vertices: the walks
				are added one length at a time, then all at once by the solve, and the layer names the road
				from the start. <a href={resolve('/demo')}>Run it yourself</a>.
			</figcaption>
		</figure>
	</section>

	<section class="abstract" aria-labelledby="abstract">
		<h2 id="abstract">Abstract</h2>
		<!-- eslint-disable-next-line svelte/no-at-html-tags -- built at prerender time from paper/paper.md -->
		<div class="text">{@html data.abstract}</div>
	</section>

	{#if data.figure}
		<figure class="key">
			<!-- eslint-disable-next-line svelte/no-at-html-tags -- built at prerender time from paper/paper.md -->
			{@html data.figure}
		</figure>
	{/if}

	<section class="found" aria-labelledby="found">
		<h2 id="found">What we found</h2>
		<div class="items">
			{#each data.found as f (f.key)}
				<article>
					<a class="shot" href="{resolve('/paper')}#{f.fig}" aria-label="{f.label} in the paper">
						<img
							src={f.src}
							alt={f.alt}
							width={f.width}
							height={f.height}
							loading="lazy"
							decoding="async"
						/>
					</a>
					<h3>{f.key}</h3>
					<!-- eslint-disable-next-line svelte/no-at-html-tags -- the paper's own sentence, rendered at build time -->
					<p>{@html f.html}</p>
					<a class="more" href="{resolve('/paper')}#{f.fig}">{f.label} in the paper</a>
				</article>
			{/each}
		</div>
	</section>
</main>

<style>
	.landing {
		--serif: 'Source Serif 4 Variable', 'KaTeX_Main', Georgia, serif;
		max-width: 1160px;
		margin: 0 auto;
		padding: 48px 24px 112px;
		color: var(--color-ink);
	}
	/* the title and the map side by side; on a phone, one above the other */
	.top {
		display: grid;
		grid-template-columns: minmax(0, 1fr) minmax(0, 640px);
		align-items: center;
		gap: 28px 56px;
	}
	h1 {
		font-family: var(--serif);
		font-weight: 500;
		letter-spacing: -0.012em;
		color: var(--color-ink);
	}
	.t-main {
		display: block;
		font-size: 38px;
		line-height: 1.12;
		text-wrap: balance;
	}
	.t-sub {
		display: block;
		margin-top: 12px;
		font-size: 23px;
		font-weight: 400;
		line-height: 1.3;
		text-wrap: balance;
	}
	.idea {
		max-width: 44ch;
		margin: 22px 0 0;
		font-family: var(--serif);
		font-size: 18px;
		line-height: 1.5;
		color: var(--color-ink);
		text-wrap: pretty;
	}
	.byline {
		margin-top: 16px;
		font-size: var(--text-sm);
		color: var(--color-ink-soft);
	}
	.links {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 24px;
		margin-top: 26px;
		font-size: var(--text-ui);
	}
	/* on a phone, rows of three rather than a lone link on the last row */
	@media (max-width: 560px) {
		.links {
			display: grid;
			grid-template-columns: repeat(3, max-content);
			gap: 12px 28px;
		}
	}
	.links a {
		color: var(--color-ink);
		text-decoration: none;
		border-bottom: 1.5px solid var(--color-mist);
		padding-bottom: 2px;
		transition: border-color var(--hover) ease;
	}
	.links a:hover {
		border-bottom-color: var(--color-ink);
	}
	.links a.demo {
		border-bottom-color: var(--color-rose);
	}
	.hero {
		margin: 0;
		min-width: 0;
	}
	.hero figcaption {
		margin-top: 6px;
		font-size: var(--text-xs);
		line-height: 1.55;
		color: var(--color-ink-soft);
	}
	.abstract {
		display: grid;
		grid-template-columns: 120px minmax(0, 680px);
		gap: 0 40px;
		justify-content: center;
		margin-top: 80px;
	}
	.abstract h2,
	.found h2 {
		padding-top: 6px;
		text-align: right;
		font-size: var(--text-xs);
		font-weight: 500;
		color: var(--color-ink-soft);
	}
	.text {
		font-family: var(--serif);
		font-size: 17px;
		line-height: 1.62;
		hyphens: auto;
	}
	.text :global(p + p) {
		margin-top: 1em;
	}
	.text :global(sup.cite) {
		font-family: var(--font-sans);
		font-size: 0.62em;
		line-height: 0;
	}
	.text :global(sup.cite a) {
		color: var(--color-cobalt);
		text-decoration: none;
	}
	.key {
		max-width: 960px;
		margin: 56px auto 0;
	}
	.key :global(.fig-open) {
		display: block;
		cursor: zoom-in;
	}
	.key :global(img) {
		display: block;
		width: 100%;
		height: auto;
	}
	.key :global(figcaption) {
		max-width: 680px;
		margin: 14px auto 0;
		font-size: 13.5px;
		line-height: 1.55;
		color: var(--color-ink-soft);
	}
	.key :global(.fig-label) {
		font-weight: 600;
		color: var(--color-ink);
	}
	.found {
		margin-top: 96px;
	}
	.found h2 {
		text-align: left;
		margin-bottom: 22px;
		font-size: var(--text-sm);
		color: var(--color-ink);
	}
	/* two findings a row; each one's figure, name, sentence and link share rows with its neighbour's (subgrid) */
	.items {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		column-gap: 56px;
	}
	article {
		display: grid;
		grid-row: span 4;
		grid-template-rows: subgrid;
		row-gap: 0;
		align-content: start;
		min-width: 0;
		margin-bottom: 64px;
	}
	.shot {
		display: block;
		align-self: start;
	}
	.shot img {
		display: block;
		width: 100%;
		height: auto;
		transition: opacity var(--hover) ease;
	}
	.shot:hover img {
		opacity: 0.88;
	}
	h3 {
		margin-top: 18px;
		font-size: var(--text-sm);
		font-weight: 600;
		color: var(--color-rose);
	}
	article p {
		margin-top: 6px;
		font-family: var(--serif);
		font-size: 17px;
		line-height: 1.55;
		color: var(--color-ink);
	}
	article p :global(.katex) {
		font-size: 1.02em;
	}
	article p :global(sup.cite) {
		font-family: var(--font-sans);
		font-size: 0.62em;
		line-height: 0;
	}
	.more {
		justify-self: start;
		align-self: start;
		margin-top: 10px;
		font-size: var(--text-sm);
		color: var(--color-ink-soft);
		text-decoration: underline;
		text-underline-offset: 4px;
	}
	.more:hover {
		color: var(--color-ink);
	}
	@media (max-width: 959px) {
		.top {
			grid-template-columns: minmax(0, 1fr);
		}
		.hero {
			max-width: 640px;
			width: 100%;
			justify-self: center;
		}
		.items {
			grid-template-columns: minmax(0, 1fr);
		}
	}
	@media (max-width: 719px) {
		.landing {
			padding: 32px 16px 72px;
		}
		.t-main {
			font-size: 31px;
		}
		.t-sub {
			font-size: 20px;
		}
		.idea {
			font-size: 17px;
		}
		.abstract {
			grid-template-columns: minmax(0, 1fr);
			margin-top: 52px;
		}
		.abstract h2 {
			text-align: left;
			margin-bottom: 6px;
		}
		.found {
			margin-top: 64px;
		}
		article {
			margin-bottom: 48px;
		}
	}
</style>
