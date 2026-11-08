<script lang="ts">
	/**
	 * The mark: two vertices u and v joined through a middle vertex k. The two steps u → k and k → v are ink; the pair
	 * they compose, u → v, is the rose arc beneath, as one layer of the model writes it.
	 */
	let { size = 20 }: { size?: number } = $props();
</script>

<svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
	<path class="join" d="M6 21 Q16 33 26 21" />
	<path class="step" d="M6 21 L16 8 L26 21" />
	<circle class="ink" cx="6" cy="21" r="3.6" />
	<circle class="mid" cx="16" cy="8" r="3.2" />
	<circle class="ink" cx="26" cy="21" r="3.6" />
</svg>

<style>
	svg {
		display: block;
		flex: none;
	}
	.step {
		fill: none;
		stroke: var(--color-ink);
		stroke-width: 2;
		stroke-linejoin: round;
	}
	.join {
		fill: none;
		stroke: var(--color-rose);
		stroke-width: 2.6;
		stroke-linecap: round;
	}
	.ink {
		fill: var(--color-ink);
	}
	.mid {
		fill: var(--color-paper);
		stroke: var(--color-ink);
		stroke-width: 2;
	}
</style>
