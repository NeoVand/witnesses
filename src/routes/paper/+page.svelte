<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import Icon from '#lib/ui/Icon.svelte';
	import Meta from '#lib/ui/Meta.svelte';
	import Doc from '#lib/paper/Doc.svelte';
	import { Github, Pdf, Play, School } from '#lib/ui/icons.ts';
	import { CODE_URL, DESCRIPTION, PDF } from '#lib/site.ts';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<Meta title={data.meta.title} description={DESCRIPTION} path="/paper" />

<Doc meta={data.meta} html={data.html}>
	{#snippet links()}
		<a class="ic" href={resolve('/demo')}><Icon icon={Play} size={16} />Demo</a>
		<span aria-hidden="true">·</span>
		<a class="ic" href={resolve('/tutorial')}><Icon icon={School} size={16} />Tutorial</a>
		<span aria-hidden="true">·</span>
		<a class="ic" href={asset(PDF)}><Icon icon={Pdf} size={16} />PDF</a>
		{#if CODE_URL}
			<span aria-hidden="true">·</span>
			<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- an external URL, set in site.ts -->
			<a class="ic" href={CODE_URL}><Icon icon={Github} size={16} />Code</a>
		{/if}
	{/snippet}
</Doc>
