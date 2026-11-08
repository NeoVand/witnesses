<script lang="ts">
	import './layout.css';
	import favicon from '#lib/assets/favicon.svg';
	import TopBar from '#lib/ui/TopBar.svelte';
	// the two faces every page sets first, fetched with the page so text never reflows when they arrive
	import geist from '@fontsource-variable/geist/files/geist-latin-wght-normal.woff2?url';
	import serif from '@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-normal.woff2?url';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<link rel="preload" href={geist} as="font" type="font/woff2" crossorigin="anonymous" />
	<link rel="preload" href={serif} as="font" type="font/woff2" crossorigin="anonymous" />
	<meta name="theme-color" content="#ffffff" />
</svelte:head>

<TopBar />
{@render children()}
