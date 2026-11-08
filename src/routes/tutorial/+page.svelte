<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import Icon from '#lib/ui/Icon.svelte';
	import Meta from '#lib/ui/Meta.svelte';
	import Doc from '#lib/paper/Doc.svelte';
	import { Book, Pdf, Play } from '#lib/ui/icons.ts';
	import { TUTORIAL_PDF } from '#lib/site.ts';
	import type { PageProps } from './$types';

	let { data }: PageProps = $props();
</script>

<Meta
	title={data.meta.title}
	description="A tutorial for students on the paper “Attention Is Only as Good as Its Witnesses”: partition functions, walks and matrix powers, the depth law, Newton’s identities, characters, the size law, NC and P, with questions and answers."
	path="/tutorial"
/>

<Doc meta={data.meta} html={data.html} lede={data.lede} toc={data.toc}>
	{#snippet links()}
		<a class="ic" href={resolve('/paper')}><Icon icon={Book} size={16} />The paper</a>
		<span aria-hidden="true">·</span>
		<a class="ic" href={resolve('/demo')}><Icon icon={Play} size={16} />Demo</a>
		<span aria-hidden="true">·</span>
		<a class="ic" href={asset(TUTORIAL_PDF)}><Icon icon={Pdf} size={16} />PDF</a>
	{/snippet}
</Doc>
