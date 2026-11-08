<script lang="ts">
	/** Renders a Hugeicons outline icon from its element data. Decorative unless `label` is set. */
	import type { IconData } from './icons.ts';

	let {
		icon,
		size = 18,
		label,
		class: className = ''
	}: { icon: IconData; size?: number; label?: string; class?: string } = $props();

	const kebab = (k: string) => k.replace(/[A-Z]/g, (m) => '-' + m.toLowerCase());
	const shapes = $derived(
		(icon as readonly (readonly [string, Record<string, string | number>])[]).map(
			([tag, attrs]) => ({
				tag,
				attrs: Object.fromEntries(
					Object.entries(attrs)
						.filter(([k]) => k !== 'key')
						.map(([k, v]) => [kebab(k), String(v)])
				)
			})
		)
	);
</script>

<svg
	class={['icon', className]}
	width={size}
	height={size}
	viewBox="0 0 24 24"
	fill="none"
	role={label ? 'img' : undefined}
	aria-label={label}
	aria-hidden={label ? undefined : 'true'}
>
	{#each shapes as s, i (i)}
		<svelte:element this={s.tag} {...s.attrs} />
	{/each}
</svg>

<style>
	.icon {
		flex: none;
		display: block;
	}
</style>
