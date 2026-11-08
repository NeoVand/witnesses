<script lang="ts">
	/**
	 * What the model writes, raw: one score (logit) for every town name in the input, turned into probabilities by a
	 * softmax; the answer is the most probable name. A strip over every name in order, the leaders labelled, and the
	 * top five as numbers. From a recording (before the model loads) only the probabilities are kept.
	 */
	let {
		names,
		logits = null,
		probs = null,
		current,
		pending = false
	}: {
		names: readonly number[];
		/** one logit per name (aligned with `names`) */
		logits?: readonly number[] | null;
		/** probabilities, when no logits were kept */
		probs?: ArrayLike<number> | null;
		current: number;
		/** the model is still running: the last answer is shown dimmed */
		pending?: boolean;
	} = $props();

	const p = $derived.by(() => {
		if (logits && logits.length === names.length) {
			let mx = -Infinity;
			for (const l of logits) mx = Math.max(mx, l);
			const e = logits.map((l) => Math.exp(l - mx));
			const z = e.reduce((a, b) => a + b, 0);
			return e.map((x) => x / z);
		}
		return probs ? Array.from(probs) : [];
	});
	const ranked = $derived(
		names.map((v, i) => ({ v, i, p: p[i] ?? 0, l: logits?.[i] ?? null })).sort((a, b) => b.p - a.p)
	);
	const top = $derived(ranked.slice(0, 5));
	const answer = $derived(ranked[0]?.v ?? null);

	let width = $state(0);
	const H = 52;
	const PAD = 14;
	const bw = $derived(Math.max(1, (width - 2) / Math.max(1, names.length)));
	const x = (i: number) => 1 + i * bw;
	const y = (q: number) => H - 4 - q * (H - PAD - 6);
	/**
	 * The strip: every name's logit, scaled from the lowest to the highest (the softmax keeps only the order and the
	 * gaps); from a recording, the probability itself
	 */
	const h = $derived.by(() => {
		if (logits && logits.length === names.length) {
			const lo = Math.min(...logits);
			const hi = Math.max(...logits);
			return logits.map((l) => (hi > lo ? (l - lo) / (hi - lo) : 0));
		}
		return p;
	});
	const lab = $derived(ranked.slice(0, 3));
	const pf = (q: number) =>
		q >= 0.995 ? '1.00' : q === 0 ? '0' : q < 0.005 ? q.toExponential(0) : q.toFixed(2);
</script>

<div class={['readout', { pending }]}>
	<p class="meta num">
		{names.length} names, one {logits ? 'logit' : 'score'} each, then a softmax: the answer is
		<b>{answer ?? '–'}</b>, the next town from {current}
	</p>
	<div class="strip" bind:clientWidth={width}>
		{#if width > 0}
			<svg
				{width}
				height={H}
				role="img"
				aria-label="The probability the model gives each town name"
			>
				<line class="base" x1="0" x2={width} y1={H - 4} y2={H - 4} />
				{#each names as v, i (v)}
					{@const q = h[i] ?? 0}
					{#if q > 0.004}
						<rect
							class={['bar', { ans: v === answer }]}
							x={x(i)}
							y={y(q)}
							width={Math.max(1.5, bw - 1)}
							height={H - 4 - y(q)}
						/>
					{/if}
				{/each}
				{#each lab as r (r.v)}
					<text
						class={['lab', { ans: r.v === answer }]}
						x={Math.min(width - 14, Math.max(14, x(r.i) + bw / 2))}
						y={Math.max(10, y(h[r.i] ?? 0) - 4)}>{r.v}</text
					>
				{/each}
			</svg>
		{/if}
	</div>
	<p class="axis num" aria-hidden="true">
		<span>{names[0]}</span><span>{names[names.length - 1]}</span>
	</p>
	<p class="caption">
		{logits ? 'Each bar a name’s logit, lowest to highest' : 'Each bar a name’s probability'}; names
		in order.
	</p>
	<table class="num">
		<thead>
			<tr><th>name</th><th class="r">logit</th><th class="r">p</th></tr>
		</thead>
		<tbody>
			{#each top as r, k (r.v)}
				<tr class={{ ans: k === 0 }}>
					<td>{r.v}</td>
					<td class="r">{r.l === null ? '–' : r.l.toFixed(2)}</td>
					<td class="r">{pf(r.p)}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.readout {
		min-width: 0;
		transition: opacity var(--hover) ease;
	}
	.pending {
		opacity: 0.45;
	}
	.meta {
		font-size: var(--text-xs);
		color: var(--color-ink-soft);
	}
	.meta b {
		color: var(--color-rose);
		font-weight: 600;
	}
	/* the strip keeps its height before it is measured, so nothing under it moves */
	.strip {
		height: 52px;
		margin-top: 8px;
	}
	svg {
		display: block;
		overflow: visible;
	}
	.base {
		stroke: var(--color-mist);
	}
	.bar {
		fill: var(--color-blue-grey);
	}
	.bar.ans {
		fill: var(--color-rose);
	}
	.lab {
		font-family: var(--font-mono);
		font-size: 10.5px;
		text-anchor: middle;
		fill: var(--color-ink-soft);
	}
	.lab.ans {
		fill: var(--color-rose);
		font-weight: 600;
	}
	.axis {
		display: flex;
		justify-content: space-between;
		margin-top: 3px;
		font-family: var(--font-mono);
		font-size: 10.5px;
		color: var(--color-ink-soft);
	}
	.caption {
		margin-top: 4px;
		font-size: 11px;
		color: var(--color-ink-soft);
	}
	table {
		width: 100%;
		margin-top: 12px;
		border-collapse: collapse;
		font-family: var(--font-mono);
		font-size: 11.5px;
	}
	th {
		padding: 0 0 3px;
		text-align: left;
		font-family: var(--font-sans);
		font-size: var(--text-xs);
		font-weight: 400;
		color: var(--color-ink-soft);
		border-bottom: 1px solid var(--color-edge);
	}
	td {
		padding: 2px 0;
		color: var(--color-ink);
	}
	.r {
		text-align: right;
	}
	tr.ans td {
		color: var(--color-rose);
		font-weight: 600;
	}
</style>
