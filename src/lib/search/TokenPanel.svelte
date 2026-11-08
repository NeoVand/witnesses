<script lang="ts">
	/**
	 * What the model reads, raw: the question's token sequence exactly as the benchmark writes it and the model receives
	 * it, left-padded to 3V + 5 tokens; every road one `E u v`, in the input's own (shuffled) order, then `Q start goal`,
	 * then `P` and the path so far. Hovering a road's tokens lights that road on the map, and a road hovered on the map
	 * lights its tokens; hovering the query rings the start and the goal. Names or raw token ids, at a toggle.
	 */
	import { untrack } from 'svelte';
	import { spec as makeSpec } from './tokens.ts';

	let {
		tokens,
		V,
		hoverRoad = $bindable(null),
		onquery,
		walked = 0,
		edited = false
	}: {
		tokens: readonly number[];
		V: number;
		hoverRoad?: { u: number; v: number } | null;
		/** the query's tokens are hovered (true) or left (false) */
		onquery?: (on: boolean) => void;
		/** how many path tokens the walk has appended (the newest are marked) */
		walked?: number;
		/** the question was edited: its query tokens are marked */
		edited?: boolean;
	} = $props();

	const S = $derived(makeSpec(V));
	let ids = $state(false);
	const parts = $derived.by(() => {
		let pads = 0;
		while (pads < tokens.length && tokens[pads] === S.pad) pads++;
		const edges: { u: number; v: number }[] = [];
		let i = pads;
		while (i + 2 < tokens.length && tokens[i] === S.edge) {
			edges.push({ u: tokens[i + 1], v: tokens[i + 2] });
			i += 3;
		}
		const q = tokens[i] === S.query ? { s: tokens[i + 1], g: tokens[i + 2] } : null;
		const path = tokens[i + 3] === S.path ? tokens.slice(i + 4) : [];
		return { pads, edges, q, path };
	});
	/** a marker as the model sees it: its letter, or its token id */
	const mk = (letter: string, id: number) => (ids ? String(id) : letter);

	// a road hovered on the map: its tokens scroll into view inside the panel (never the page)
	let box: HTMLElement | undefined = $state();
	$effect(() => {
		const h = hoverRoad;
		if (!h || !box) return;
		untrack(() => {
			const el = box!.querySelector<HTMLElement>(`[data-k="${h.u}>${h.v}"]`);
			if (!el) return;
			const top = el.offsetTop - box!.offsetTop;
			if (top < box!.scrollTop || top + el.offsetHeight > box!.scrollTop + box!.clientHeight)
				box!.scrollTo({ top: top - box!.clientHeight / 2, behavior: 'smooth' });
		});
	});
</script>

<div class="tokens">
	<p class="meta num">
		<span>{tokens.length} tokens</span><span>{parts.pads} padding</span><span
			>{parts.edges.length} roads × 3</span
		><span>query and path {3 + 1 + parts.path.length}</span>
		<button class="plain toggle" aria-pressed={ids} onclick={() => (ids = !ids)}
			>{ids ? 'show markers as letters' : 'show raw token ids'}</button
		>
	</p>
	<div class="seq" bind:this={box} aria-label="The input tokens">
		{#if parts.pads}<span class="pad">{ids ? `${S.pad} ×${parts.pads}` : `pad ×${parts.pads}`}</span
			>{/if}
		{#each parts.edges as e, i (i)}
			{@const on = hoverRoad?.u === e.u && hoverRoad?.v === e.v}
			<span
				class={['e', { on }]}
				data-k="{e.u}>{e.v}"
				role="presentation"
				onpointerenter={() => (hoverRoad = { u: e.u, v: e.v })}
				onpointerleave={() => (hoverRoad = null)}
				><span class="m">{mk('E', S.edge)}</span> {e.u} {e.v}</span
			>
		{/each}
		{#if parts.q}
			<span
				class={['q', { edited }]}
				role="presentation"
				onpointerenter={() => onquery?.(true)}
				onpointerleave={() => onquery?.(false)}
				><span class="m">{mk('Q', S.query)}</span> <b>{parts.q.s}</b>
				<b class="goal">{parts.q.g}</b></span
			>
			<span class="p"
				><span class="m">{mk('P', S.path)}</span>{#each parts.path as v, k (k)}
					<span class={['v', { fresh: walked > 0 && k >= parts.path.length - walked }]}>{v}</span>
				{/each}</span
			>
		{/if}
	</div>
</div>

<style>
	.tokens {
		min-width: 0;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		gap: 2px 0;
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.meta > span + span::before {
		content: ' · ';
		white-space: pre;
	}
	.toggle {
		margin-left: auto;
		font-size: var(--text-xs);
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.seq {
		position: relative;
		max-height: 168px;
		margin-top: 8px;
		padding: 8px 0;
		overflow-y: auto;
		border-top: 1px solid var(--color-edge);
		border-bottom: 1px solid var(--color-edge);
		font-family: var(--font-mono);
		font-size: 11.5px;
		line-height: 1.75;
		color: var(--color-ink);
		overscroll-behavior: contain;
	}
	.m {
		color: var(--color-blue-grey);
	}
	.pad {
		color: var(--color-mist);
	}
	/* every road's three tokens stay together; the groups wrap */
	.seq {
		display: flex;
		flex-wrap: wrap;
		align-content: start;
		gap: 0 0.65em;
	}
	.p {
		display: inline-flex;
		flex-wrap: wrap;
		gap: 0 0.65em;
	}
	.e {
		white-space: nowrap;
		border-radius: 2px;
		cursor: default;
		transition:
			color var(--hover) ease,
			background-color var(--hover) ease;
	}
	.e:hover,
	.e.on {
		color: var(--color-cobalt);
		background: rgb(37 87 167 / 0.1);
	}
	.e.on .m,
	.e:hover .m {
		color: var(--color-cobalt);
	}
	.q {
		white-space: nowrap;
		cursor: default;
	}
	.q b {
		font-weight: 600;
	}
	.q .goal {
		color: var(--color-rose);
	}
	.q:hover {
		background: rgb(37 87 167 / 0.1);
	}
	.q.edited .goal {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.fresh {
		color: var(--color-rose);
		font-weight: 600;
	}
</style>
