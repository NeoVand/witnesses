/**
 * The demo's state and its one timeline: show a question, sweep the layers (every inside view follows the playhead),
 * light the answer, then step along the path, one forward pass per step, until the goal. Any click on the graph or a
 * control cancels the timeline and starts a new one from the new question.
 *
 * Two models answer: the 8 layers (pairn_v49.json, 49-town maps only: it was trained with positions for that input
 * length) and the OR join, one layer repeated K times (orjoin_v49.json), on maps of 49 or 125 towns. For the repeated
 * layer the playhead runs over its iterations, 0 (the start) to K.
 */
import { asset } from '$app/paths';
import { goodSteps, children, blocked } from './graph.ts';
import {
	current,
	encode,
	parse,
	renamed,
	room,
	spec as makeSpec,
	type Problem,
	type Spec
} from './tokens.ts';
import { unpack, type Packed, type Trace } from './trace.ts';
import { ease } from './theme.ts';
import type { FromWorker, ModelKind, ToWorker } from './worker.ts';
import opening from './opening.json';

export const SPEC = makeSpec(49);
export const LAYERS = 8;
export type { ModelKind };
/** the map sizes: the 49-town questions both models answer, and 125-town ones for the repeated layer */
export const SIZES = [49, 125] as const;
export type Size = (typeof SIZES)[number];
/** the iterations the repeated layer was scored with: ceil(log2 n) + 2, as registered (8 at 49, 9 at 125) */
export const DEFAULT_K: Record<Size, number> = { 49: 8, 125: 9 };
export const MAX_K = 10;
const FILES: Record<Size, { questions: string }> = {
	49: { questions: 'model/graphs_v49.json' },
	125: { questions: 'model/graphs_v125.json' }
};

export interface Question {
	lookahead: number | null;
	tokens: number[];
	answer: number;
}

export type Pace = 'slow' | 'fast';
const PACE = {
	slow: { layer: 420, hold: 650, walkLayer: 95, walkHold: 380 },
	fast: { layer: 0, hold: 220, walkLayer: 0, walkHold: 180 }
} as const;
/** how long the current-town marker takes to drive one road (the map view eases it along the road) */
export const MOVE = { slow: 800, fast: 450 } as const;

const clone = (p: Problem): Problem => ({
	edges: p.edges.map(([u, v]) => [u, v] as [number, number]),
	start: p.start,
	goal: p.goal,
	path: [...p.path]
});

const key = (tokens: number[]) => tokens.join(',');

export class Search {
	/** which model answers, on which size of map, and (for the repeated layer) how many times it runs */
	model = $state<ModelKind>('stack');
	size = $state<Size>(49);
	K = $state(DEFAULT_K[49]);
	spec = $derived<Spec>(makeSpec(this.size));
	/** the layers the playhead runs over: 8, or the repeated layer's K iterations */
	depth = $derived(this.model === 'stack' ? LAYERS : this.K);
	/** every exported test question of this size (fetched); until then, the opening one */
	questions = $state.raw<Question[]>([]);
	/** which sizes' questions have arrived */
	have = $state<Record<Size, boolean>>({ 49: false, 125: false });
	index = $state((opening as { question: number }).question);
	lookahead = $state<number | null>((opening as { lookahead: number | null }).lookahead);
	/** the question as the benchmark wrote it; as the visitor edited it; and as it stands after the steps walked */
	original = $state.raw<Problem>(parse((opening as { steps: Packed[] }).steps[0].tokens, SPEC));
	question = $state.raw<Problem>(parse((opening as { steps: Packed[] }).steps[0].tokens, SPEC));
	problem = $state.raw<Problem>(parse((opening as { steps: Packed[] }).steps[0].tokens, SPEC));
	edited = $state(false);
	/** the benchmark's names → the names on screen, once the visitor renames the towns (null: the benchmark's own) */
	names = $state.raw<Map<number, number> | null>(null);
	/** the last renaming: the first answer before it (old names), the renaming, and the first answer after it */
	renaming = $state.raw<{ before: number; step: Map<number, number>; after?: number } | null>(null);
	trace = $state.raw<Trace | null>(null);
	/** the layer the views show: a float while the playhead moves between layers */
	layer = $state(0);
	phase = $state<'sweep' | 'answer' | 'walk' | 'done' | 'paused'>('sweep');
	/** the steps the model has taken on this question */
	taken = $state.raw<number[]>([]);
	pace = $state<Pace>('slow');
	/** each model in this browser: null while it loads */
	ready = $state<Record<ModelKind, boolean | null>>({ stack: null, loop: null });
	errors = $state<Record<ModelKind, string | null>>({ stack: null, loop: null });
	/** parameters in each model's loaded weights */
	paramsBy = $state<Record<ModelKind, number | null>>({ stack: null, loop: null });
	note = $state<string | null>(null);
	/** the model on screen: loaded, failed (false) or loading (null), its error and its parameters */
	get live() {
		return this.ready[this.model];
	}
	get loadError() {
		return this.errors[this.model];
	}
	set loadError(m: string | null) {
		this.errors[this.model] = m;
	}
	get params() {
		return this.paramsBy[this.model];
	}
	/** the last live forward pass, in milliseconds */
	ms = $state<number | null>(null);
	/** a new question: the layout may let go once (ratchet) */
	runs = $state(0);

	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches and a worker URL, not state
	#recorded = new Map<string, Trace>();
	#worker: Worker | null = null;
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches and a worker URL, not state
	#pending = new Map<number, { resolve: (t: Trace) => void; reject: (e: Error) => void }>();
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches and a worker URL, not state
	#sets = new Map<Size, Question[]>();
	#id = 0;
	#token = 0;
	/** the model's first answer on the question as asked (before any step is walked) */
	#firstAnswer: number | null = null;
	#raf = 0;

	constructor() {
		for (const s of (opening as { steps: Packed[] }).steps)
			this.#recorded.set(key(s.tokens), unpack(s));
		this.trace = this.#recorded.get(key((opening as { steps: Packed[] }).steps[0].tokens)) ?? null;
	}

	/**
	 * Start both models and fetch the questions; the recording plays meanwhile (or, given `first`, that question, on the
	 * map size and model given)
	 */
	start(first?: number, opts: { size?: Size; model?: ModelKind; K?: number } = {}) {
		const size = opts.size ?? 49;
		const model = size === 49 ? (opts.model ?? 'stack') : 'loop';
		const K = opts.K && opts.K >= 1 && opts.K <= MAX_K ? opts.K : DEFAULT_K[size];
		const opening = first === undefined && size === 49 && model === 'stack' && !opts.K;
		void this.play();
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches and a worker URL, not state
		this.#worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.#worker.onmessage = (e: MessageEvent<FromWorker>) => {
			const m = e.data;
			if (m.kind === 'loaded') {
				if (m.model === 'prop') return;
				this.paramsBy[m.model] = m.params;
				this.ready[m.model] = true;
			} else if (m.kind === 'failed') {
				if (m.model && m.model !== 'prop') {
					if (this.ready[m.model] === null) this.ready[m.model] = false;
					this.errors[m.model] = m.message;
				}
				if (m.id !== undefined) {
					this.#pending.get(m.id)?.reject(new Error(m.message));
					this.#pending.delete(m.id);
				}
			} else if (m.kind === 'ran') {
				this.ms = m.ms;
				const t: Trace = {
					tokens: [],
					names: m.names,
					current: m.current,
					answer: m.answer,
					maps: m.maps,
					logits: m.logits,
					ms: m.ms
				};
				this.#pending.get(m.id)?.resolve(t);
				this.#pending.delete(m.id);
			}
		};
		this.#worker.postMessage({
			kind: 'load',
			model: 'stack',
			url: asset('model/pairn_v49.json'),
			V: SPEC.V
		} satisfies ToWorker);
		this.#worker.postMessage({
			kind: 'load',
			model: 'loop',
			url: asset('model/orjoin_v49.json'),
			V: SPEC.V
		} satisfies ToWorker);
		for (const V of SIZES)
			fetch(asset(FILES[V].questions as Parameters<typeof asset>[0]))
				.then((r) => r.json())
				.then((x: { questions: Question[] }) => {
					this.#sets.set(V, x.questions);
					this.have[V] = true;
					if (V !== size) return;
					if (!opening) {
						this.size = size;
						this.model = model;
						this.K = K;
					}
					this.questions = x.questions;
					if (!opening) this.show(first !== undefined && first < x.questions.length ? first : 0);
				})
				.catch(() => (this.note = `The ${V}-town questions could not be read.`));
	}

	stop() {
		this.#token++;
		cancelAnimationFrame(this.#raf);
		this.#worker?.terminate();
	}

	/** the model's run on tokens: recorded if this exact input was recorded (8 layers, 49 towns), else the live model */
	async #ask(tokens: number[], ctx: { model: ModelKind; V: number; K: number }): Promise<Trace> {
		// the recording stands in only until the model has loaded; from then on every number is the live pass's
		const rec =
			ctx.model === 'stack' && !this.ready.stack ? this.#recorded.get(key(tokens)) : undefined;
		if (rec) return rec;
		while (!this.ready[ctx.model]) {
			if (this.ready[ctx.model] === false)
				throw new Error(this.errors[ctx.model] ?? 'the model could not be loaded');
			await new Promise((r) => setTimeout(r, 60));
		}
		const id = ++this.#id;
		return new Promise<Trace>((resolve, reject) => {
			this.#pending.set(id, { resolve: (t) => resolve({ ...t, tokens }), reject });
			this.#worker!.postMessage({
				kind: 'run',
				id,
				model: ctx.model,
				V: ctx.V,
				iters: ctx.K,
				tokens
			} satisfies ToWorker);
		});
	}

	/** move the playhead to layer `to` over `ms`, eased; resolves false if a newer timeline took over */
	#glide(to: number, ms: number, token: number): Promise<boolean> {
		return new Promise((resolve) => {
			const from = this.layer;
			if (ms <= 0 || from === to) {
				this.layer = to;
				return resolve(token === this.#token);
			}
			const t0 = performance.now();
			const tick = (now: number) => {
				if (token !== this.#token) return resolve(false);
				const t = Math.max(0, Math.min(1, (now - t0) / ms));
				this.layer = from + (to - from) * ease(t);
				if (t < 1) this.#raf = requestAnimationFrame(tick);
				else resolve(true);
			};
			this.#raf = requestAnimationFrame(tick);
		});
	}

	#wait(ms: number, token: number) {
		return new Promise<boolean>((r) => setTimeout(() => r(token === this.#token), ms));
	}

	/**
	 * The timeline: sweep the layers, light the answer, then walk to the goal. Each step's question and its run are
	 * shown together, so the words and the drawing never describe different steps.
	 */
	async play(opts: { walk?: boolean } = {}) {
		const walk = opts.walk ?? true;
		const token = ++this.#token;
		cancelAnimationFrame(this.#raf);
		// the model, the map size and the depth stay fixed for a timeline; changing any of them starts a new one
		const S = this.spec;
		const depth = this.depth;
		const ctx = { model: this.model, V: S.V, K: depth };
		let first = true;
		let next = this.problem;
		/** the next step's run, asked for while this step is drawn */
		let ahead: Promise<Trace> | null = null;
		for (;;) {
			const tokens = encode(next, S);
			if (!tokens) {
				this.note = `The input is full: the benchmark gives every question ${S.n} tokens.`;
				this.phase = 'done';
				return;
			}
			let t: Trace;
			try {
				t = await (ahead ?? this.#ask(tokens, ctx));
			} catch (e) {
				if (token !== this.#token) return;
				this.loadError = (e as Error).message;
				this.phase = 'done';
				return;
			}
			ahead = null;
			if (token !== this.#token) return;
			// the step this answer takes; when the walk goes on, its run starts now and is ready when the drawing is
			const cur = current(next);
			const isStep = next.edges.some(([u, v]) => u === cur && v === t.answer);
			const step = { ...clone(next), path: [...next.path, t.answer] };
			const goesOn = walk && isStep && t.answer !== next.goal && room(next, S) >= 1;
			if (goesOn) {
				const nt = encode(step, S);
				if (nt) {
					ahead = this.#ask(nt, ctx);
					ahead.catch(() => {});
				}
			}
			const p = PACE[this.pace];
			const per = first ? p.layer : p.walkLayer;
			if (next !== this.problem) {
				// the step: the marker drives the road to the next town, then the model reads from there
				this.taken = [...this.taken, current(next)];
				this.problem = next;
				this.phase = 'walk';
				if (!(await this.#wait(MOVE[this.pace], token))) return;
			}
			this.trace = t;
			if (first && t.current === current(this.question)) this.#firstAnswer = t.answer;
			if (first && this.renaming && this.renaming.after === undefined)
				this.renaming = { ...this.renaming, after: t.answer };
			this.phase = 'sweep';
			if (per > 0) {
				this.layer = 0;
				if (!(await this.#wait(first ? 300 : 60, token))) return;
				for (let l = 1; l <= depth; l++) if (!(await this.#glide(l, per, token))) return;
			} else this.layer = depth;
			this.phase = 'answer';
			if (!(await this.#wait(first ? p.hold : p.walkHold, token))) return;
			first = false;
			if (isStep && t.answer === this.problem.goal) {
				// the last step lands on the goal: no question is left to ask there
				this.taken = [...this.taken, t.answer];
				this.problem = step;
				this.phase = 'walk';
				if (!(await this.#wait(MOVE[this.pace], token))) return;
				this.phase = 'done';
				return;
			}
			if (!goesOn) {
				// the path so far is part of the input: a graph with many roads leaves room for only a few steps
				if (walk && isStep && room(next, S) < 1)
					this.note = `The input is full: the benchmark gives every question ${S.n} tokens, and the roads and the path so far fill them.`;
				this.phase = 'done';
				return;
			}
			this.phase = 'walk';
			next = step;
		}
	}

	/** show the question at `i` of the exported list and play it */
	show(i: number) {
		const q = this.questions[i];
		if (!q) return;
		this.index = i;
		this.lookahead = q.lookahead;
		this.original = parse(q.tokens, this.spec);
		this.names = null;
		this.renaming = null;
		this.restart(clone(this.original), false);
	}

	/**
	 * New random names for every town and a new order for the roads: the same question, written differently. The model
	 * runs again on the new tokens; the page checks that its first answer is the same road under its new name.
	 */
	rename() {
		const before = this.#firstAnswer;
		const { problem, to: step } = renamed(this.question, this.spec.V);
		const prev = this.names;
		// a map built once and stored whole as raw state: it never changes after
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		const all = new Map<number, number>();
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		for (const v of new Set([
			...this.original.edges.flat(),
			this.original.start,
			this.original.goal
		]))
			all.set(v, step.get(prev?.get(v) ?? v) ?? v);
		this.names = all;
		this.renaming = before === null ? null : { before, step };
		this.restart(problem, true);
	}

	/** another question with this lookahead (or the first with it) */
	pick(lookahead: number | null, another = false) {
		const all = this.questions
			.map((q, i) => [q, i] as const)
			.filter(([q]) => q.lookahead === lookahead);
		if (!all.length) return;
		const at = all.findIndex(([, i]) => i === this.index);
		const next = another || at < 0 ? all[(at + 1) % all.length] : all[at];
		this.show(next[1]);
	}

	restart(p: Problem, edited: boolean) {
		this.#firstAnswer = null;
		this.question = p;
		this.problem = clone(p);
		this.edited = edited;
		this.taken = [];
		this.note = null;
		this.runs++;
		this.layer = 0;
		void this.play();
	}

	reset() {
		this.names = null;
		this.renaming = null;
		this.restart(clone(this.original), false);
	}

	/** the other model, on the question as it stands (edits kept); the 8 layers read 49-town maps only */
	setModel(m: ModelKind) {
		if (m === this.model || (m === 'stack' && this.size !== 49)) return;
		this.model = m;
		this.restart(this.question, this.edited);
	}

	/** how many times the repeated layer runs: the question as it stands, run again */
	setK(k: number) {
		const K = Math.max(1, Math.min(MAX_K, Math.round(k)));
		if (K === this.K) return;
		this.K = K;
		this.restart(this.question, this.edited);
	}

	/** maps of another size: its questions (this lookahead if it has one), the repeated layer at its registered K */
	setSize(V: Size) {
		if (V === this.size) return;
		const set = this.#sets.get(V);
		if (!set) return;
		this.size = V;
		this.K = DEFAULT_K[V];
		if (V !== 49) this.model = 'loop';
		this.questions = set;
		const looks = set.map((q) => q.lookahead);
		const want = this.lookahead;
		const near = want === null ? null : (looks.find((L) => L !== null && L >= want) ?? null);
		const i = set.findIndex((q) => q.lookahead === (looks.includes(want) ? want : near));
		this.show(i >= 0 ? i : 0);
	}

	/** edits change the question, and the model searches again from its start */
	#base(): Problem {
		return clone(this.question);
	}

	setGoal(v: number) {
		const p = this.#base();
		this.renaming = null;
		if (v === p.goal) return;
		if (p.path.includes(v)) {
			this.note = `${v} is already on the path; pick a vertex ahead of it.`;
			return;
		}
		p.goal = v;
		this.restart(p, true);
		if (!goodSteps(p).length)
			this.note = `No path leads from ${current(p)} to ${v}; the model answers anyway.`;
	}

	addEdge(u: number, v: number) {
		const p = this.#base();
		this.renaming = null;
		const why = blocked(p, u, v, children(p));
		if (why) {
			this.note = why + '.';
			return;
		}
		if (room(p, this.spec) < 3) {
			this.note = `The input is full: the benchmark gives every question ${this.spec.n} tokens.`;
			return;
		}
		p.edges.push([u, v]);
		this.restart(p, true);
		this.note = `Added ${u} → ${v}.` + this.#reach(p);
	}

	removeEdge(u: number, v: number) {
		const p = this.#base();
		this.renaming = null;
		for (let i = 0; i + 1 < p.path.length; i++)
			if (p.path[i] === u && p.path[i + 1] === v) {
				this.note = `${u} → ${v} is on the question's path so far; it stays.`;
				return;
			}
		p.edges = p.edges.filter(([a, b]) => !(a === u && b === v));
		this.restart(p, true);
		this.note = `Removed ${u} → ${v}.` + this.#reach(p);
	}

	/** a warning when no path is left from the current vertex to the goal */
	#reach(p: Problem): string {
		return goodSteps(p).length
			? ''
			: ` No path now leads from ${current(p)} to ${p.goal}; the model answers anyway.`;
	}

	/** the visitor moves the playhead: the timeline stops there */
	seek(l: number) {
		this.#token++;
		cancelAnimationFrame(this.#raf);
		this.layer = l;
		this.phase = 'paused';
	}
}
