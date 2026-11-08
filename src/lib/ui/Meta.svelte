<script lang="ts">
	/** A page's title, description and social card (a white-ground still of the project page's map). */
	import { CARD, SITE_URL } from '#lib/site.ts';

	let {
		title,
		description,
		path = ''
	}: { title: string; description: string; path?: string } = $props();
	const image = `${SITE_URL}/${CARD}`;
</script>

<svelte:head>
	<title>{title}</title>
	<meta name="description" content={description} />
	<meta property="og:type" content="article" />
	<meta property="og:title" content={title} />
	<meta property="og:description" content={description} />
	<meta property="og:url" content="{SITE_URL}{path}" />
	<meta property="og:image" content={image} />
	<meta property="og:image:width" content="1200" />
	<meta property="og:image:height" content="630" />
	<meta
		property="og:image:alt"
		content="A 250-town test map: the walks to the goal lit rose along the one route that reaches it"
	/>
	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:title" content={title} />
	<meta name="twitter:description" content={description} />
	<meta name="twitter:image" content={image} />
</svelte:head>
