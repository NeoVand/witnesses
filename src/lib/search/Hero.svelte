<script lang="ts">
	/**
	 * The project page's map: the trained propagator on a 250-town test question. The walks to the goal are added one
	 * length at a time (the partial sums Σ_{m ≤ M} S Gᵐ, M = 0, 1, 2, …), so their weight spreads back from the goal
	 * along the roads; then the map holds on the solve, every walk at once, and the road the layer chooses lights. It
	 * plays on load and loops while in view; with reduced motion, and before the page's script runs, it shows the last
	 * frame. Every light is the model's own number (hero.server.ts).
	 */
	import { onMount } from 'svelte';
	import MapView from './MapView.svelte';
	import { lightAt, type WalkFrames } from './walks.ts';
	import type { MapLayout } from './map.ts';
	import type { Problem } from './tokens.ts';
	import { ease } from './theme.ts';

	let {
		layout,
		question,
		frames,
		width = 640
	}: { layout: MapLayout; question: Problem; frames: WalkFrames; width?: number } = $props();

	// the timeline, in ms: the goal alone, the walks spreading, the solve, the chosen road, a hold, a fade
	const T = { still: 700, spread: 4600, solve: 1100, choose: 800, hold: 3600, fade: 900 };
	const LOOP = Object.values(T).reduce((a, b) => a + b, 0);

	// the last frame until the script runs (and for reduced motion)
	let M = $state(0);
	let solved = $state(1);
	let chose = $state(1);
	let fade = $state(0);
	let mounted = $state(false);
	$effect.pre(() => {
		if (!mounted) M = frames.reach;
	});

	let box: HTMLElement | undefined = $state();
	onMount(() => {
		mounted = true;
		const still = matchMedia('(prefers-reduced-motion: reduce)');
		let raf = 0;
		let t0 = performance.now();
		let visible = true;
		const frame = (now: number) => {
			let t = (now - t0) % LOOP;
			const step = (d: number) => {
				const f = Math.min(1, t / d);
				t -= d;
				return f;
			};
			const a = step(T.still);
			const b = step(T.spread);
			const c = step(T.solve);
			const d = step(T.choose);
			step(T.hold);
			const e = Math.max(0, step(T.fade));
			M = a < 1 ? 0 : frames.reach * ease(b);
			solved = ease(Math.max(0, c));
			chose = ease(Math.max(0, d));
			fade = ease(e);
			raf = requestAnimationFrame(frame);
		};
		const play = () => {
			cancelAnimationFrame(raf);
			if (still.matches || !visible || document.hidden) {
				M = frames.reach;
				solved = 1;
				chose = 1;
				fade = 0;
				return;
			}
			t0 = performance.now();
			raf = requestAnimationFrame(frame);
		};
		const io = new IntersectionObserver(([en]) => {
			const was = visible;
			visible = en.isIntersecting;
			if (visible !== was) play();
		});
		if (box) io.observe(box);
		still.addEventListener('change', play);
		document.addEventListener('visibilitychange', play);
		play();
		return () => {
			cancelAnimationFrame(raf);
			io.disconnect();
			still.removeEventListener('change', play);
			document.removeEventListener('visibilitychange', play);
		};
	});

	const light = $derived.by(() => {
		const l = lightAt(frames, M, solved);
		if (fade > 0) for (const [k, v] of l) l.set(k, v * (1 - fade));
		return l;
	});
	const chosen = $derived(
		chose > 0 && fade === 0 ? { u: frames.current, v: frames.answer, t: chose } : null
	);
	const walkLen = $derived(Math.min(frames.reach, Math.floor(M + 0.0001)));
	const stage = $derived(
		chose > 0 && fade === 0 ? 'choose' : solved > 0 && fade === 0 ? 'solve' : 'walks'
	);
</script>

<div class="hero" bind:this={box}>
	<div class="frame">
		<MapView
			{layout}
			{question}
			problem={question}
			{light}
			front={solved > 0 ? null : M}
			{chosen}
			trees
			numbers="key"
			interactive={false}
			fixed={mounted ? undefined : width}
			label="A 250-town test question from the benchmark: the weight of the walks to the goal, computed by one trained propagator layer, lights the one route that reaches it"
		/>
	</div>
	<p class="line" aria-hidden="true">
		{#if stage === 'choose'}
			<span class="k">the next road</span> from {frames.current} to
			<b>{frames.answer}</b>, read from the walks
		{:else if stage === 'solve'}
			<span class="k">every walk</span> at once, one linear solve:
			<i>X</i> = <i>S</i> (<i>I</i> − <i>G</i>)<sup>−1</sup>
		{:else}
			<span class="k">walks of up to <span class="num">{walkLen}</span> roads</span>
			Σ<sub><i>m</i> ≤ <span class="num">{walkLen}</span></sub> <i>S G<sup>m</sup></i>
		{/if}
	</p>
</div>

<style>
	.hero {
		min-width: 0;
	}
	.frame {
		overflow: hidden;
		border: 1px solid var(--color-edge);
		border-radius: 6px;
		background: var(--color-paper);
	}
	.line {
		margin-top: 10px;
		min-height: 20px;
		font-size: var(--text-sm);
		color: var(--color-ink-soft);
	}
	.line .k {
		margin-right: 8px;
		color: var(--color-ink);
		font-weight: 500;
	}
	.line b {
		color: var(--color-rose);
		font-weight: 600;
	}
	.line i {
		font-family: var(--font-serif);
		font-size: 1.05em;
	}
	.line sub,
	.line sup {
		font-size: 0.72em;
	}
</style>
