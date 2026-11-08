/**
 * The propagator demo's state and its one timeline: show a question, add the walks back from the goal one length at
 * a time (the partial sums), ease to the solve, then light the road the layer chooses. A click on a town or a control
 * starts a new timeline. The page opens on the 250-town question, recorded at build time by the same forward pass
 * (hero.server.ts), and goes live once the weights (300 KB) arrive.
 */
import { asset } from '$app/paths';
import { goodSteps } from './graph.ts';
import { current, encode, parse, renamed, spec as makeSpec, type Problem } from './tokens.ts';
import { ease } from './theme.ts';
import type { FromWorker, ToWorker, WalkRun } from './worker.ts';
import type { WalkFrames } from './walks.ts';

export const TOWNS = [49, 125, 250] as const;
export type Towns = (typeof TOWNS)[number];
export const LOOKS: Record<Towns, number[]> = {
	49: [1, 2, 3, 4, 6, 8, 12, 16, 24],
	125: [1, 2, 4, 8, 16, 24, 32, 48, 62],
	250: [31]
};

export interface Question {
	lookahead: number | null;
	tokens: number[];
	answer: number;
}

const clone = (p: Problem): Problem => ({
	edges: p.edges.map(([u, v]) => [u, v] as [number, number]),
	start: p.start,
	goal: p.goal,
	path: [...p.path]
});

/** the recorded opening: the 250-town question's run, computed at build time by the same pass */
export interface Opening {
	run: WalkRun;
	lookahead: number;
}

export class Walks {
	towns = $state<Towns>(250);
	spec = $derived(makeSpec(this.towns));
	questions = $state.raw<Question[]>([]);
	have = $state<Record<Towns, boolean>>({ 49: false, 125: false, 250: false });
	index = $state(0);
	lookahead = $state<number | null>(31);
	/** the question as the benchmark wrote it, and as it stands (the visitor may move the goal) */
	original = $state.raw<Problem>({ edges: [], start: 0, goal: 0, path: [0] });
	/** the benchmark's names → the names on screen, once the visitor renames the towns (null: the benchmark's own) */
	names = $state.raw<Map<number, number> | null>(null);
	/** the last renaming: the answer before it (in the names before it) and the renaming itself */
	renaming = $state.raw<{ before: number; step: Map<number, number> } | null>(null);
	problem = $state.raw<Problem>({ edges: [], start: 0, goal: 0, path: [0] });
	edited = $state(false);
	/** the run on screen: live, or the recorded opening (frames only) */
	run = $state.raw<WalkRun | null>(null);
	frames = $state.raw<WalkFrames | null>(null);
	recorded = $state(true);
	/** the playhead: walk length M (a float, 0 to frames.reach), then the solve (0 to 1), then the chosen road */
	M = $state(0);
	solved = $state(0);
	chose = $state(0);
	phase = $state<'walks' | 'solve' | 'choose' | 'done' | 'paused' | 'waiting'>('waiting');
	ready = $state<boolean | null>(null);
	error = $state<string | null>(null);
	params = $state<number | null>(null);
	ms = $state<number | null>(null);
	note = $state<string | null>(null);
	/** a new question: the page may let go of its heights once */
	runs = $state(0);

	#worker: Worker | null = null;
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches, not state
	#pending = new Map<number, { resolve: (r: WalkRun) => void; reject: (e: Error) => void }>();
	// eslint-disable-next-line svelte/prefer-svelte-reactivity -- private caches, not state
	#sets = new Map<Towns, Question[]>();
	#id = 0;
	#token = 0;
	#raf = 0;
	#still = false;

	constructor(opening: Opening) {
		const p = parse(opening.run.tokens, makeSpec(250));
		this.original = p;
		this.problem = clone(p);
		this.run = opening.run;
		this.frames = opening.run.frames;
		this.lookahead = opening.lookahead;
	}

	/** start the model and fetch the questions; the recording plays meanwhile (or, given `first`, that question) */
	start(first?: { towns: Towns; index: number }) {
		this.#still = matchMedia('(prefers-reduced-motion: reduce)').matches;
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- a worker URL, not state
		this.#worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
		this.#worker.onmessage = (e: MessageEvent<FromWorker>) => {
			const m = e.data;
			if (m.kind === 'loaded' && m.model === 'prop') {
				this.params = m.params;
				this.ready = true;
			} else if (m.kind === 'failed') {
				if (m.model === 'prop' || m.model === undefined) {
					if (this.ready === null) this.ready = false;
					this.error = m.message;
				}
				if (m.id !== undefined) {
					this.#pending.get(m.id)?.reject(new Error(m.message));
					this.#pending.delete(m.id);
				}
			} else if (m.kind === 'walked') {
				this.ms = m.ms;
				this.#pending.get(m.id)?.resolve(m);
				this.#pending.delete(m.id);
			}
		};
		this.#worker.postMessage({
			kind: 'load',
			model: 'prop',
			url: asset('model/prop_v49.json'),
			V: 250
		} satisfies ToWorker);
		for (const V of TOWNS)
			fetch(asset(`model/graphs_v${V}.json` as Parameters<typeof asset>[0]))
				.then((r) => r.json())
				.then((x: { questions: Question[] }) => {
					this.#sets.set(V, x.questions);
					this.have[V] = true;
					if (V === (first?.towns ?? 250)) {
						this.questions = x.questions;
						if (first && (first.towns !== 250 || first.index !== 0)) {
							this.towns = first.towns;
							this.show(Math.min(first.index, x.questions.length - 1));
						}
					}
				})
				.catch(() => (this.note = `The ${V}-town questions could not be read.`));
		// the recording plays at once; the live run takes over when it arrives
		void this.play(this.frames, null);
		void this.#live();
	}

	stop() {
		this.#token++;
		cancelAnimationFrame(this.#raf);
		this.#worker?.terminate();
	}

	/** the live model on the question as it stands; the recording keeps playing until it answers */
	async #live() {
		const token = this.#token;
		const tokens = encode(this.problem, this.spec);
		if (!tokens) return;
		try {
			const r = await this.#ask(tokens, this.towns);
			if (token !== this.#token) return;
			this.run = r;
			this.recorded = false;
			this.frames = r.frames;
		} catch (e) {
			if (token !== this.#token) return;
			this.error = (e as Error).message;
		}
	}

	async #ask(tokens: number[], V: number): Promise<WalkRun> {
		while (!this.ready) {
			if (this.ready === false) throw new Error(this.error ?? 'the model could not be loaded');
			await new Promise((r) => setTimeout(r, 60));
		}
		const id = ++this.#id;
		return new Promise<WalkRun>((resolve, reject) => {
			this.#pending.set(id, { resolve, reject });
			this.#worker!.postMessage({ kind: 'walks', id, V, tokens } satisfies ToWorker);
		});
	}

	#frame(ms: number, token: number, set: (t: number) => void): Promise<boolean> {
		return new Promise((resolve) => {
			if (ms <= 0) {
				set(1);
				return resolve(token === this.#token);
			}
			const t0 = performance.now();
			const tick = (now: number) => {
				if (token !== this.#token) return resolve(false);
				const t = Math.min(1, (now - t0) / ms);
				set(t);
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
	 * The timeline on frames f: the walks spread back from the goal (about 130 ms a road, 1.6 to 6 s in all), the solve,
	 * the chosen road. With `pending`, the live run is awaited first (the recording stays on screen meanwhile).
	 */
	async play(f: WalkFrames | null, pending: Promise<WalkRun> | null) {
		const token = ++this.#token;
		cancelAnimationFrame(this.#raf);
		if (pending) {
			// the last question's light goes out while the layer runs on the new one
			this.M = 0;
			this.solved = 0;
			this.chose = 0;
			this.phase = 'waiting';
			try {
				const r = await pending;
				if (token !== this.#token) return;
				this.run = r;
				this.recorded = false;
				f = this.frames = r.frames;
			} catch (e) {
				if (token !== this.#token) return;
				this.error = (e as Error).message;
				return;
			}
		}
		if (!f) return;
		if (this.#still) {
			this.M = f.reach;
			this.solved = 1;
			this.chose = 1;
			this.phase = 'done';
			return;
		}
		this.M = 0;
		this.solved = 0;
		this.chose = 0;
		this.phase = 'walks';
		if (!(await this.#wait(500, token))) return;
		const spread = Math.max(1600, Math.min(6000, 130 * f.reach + 600));
		// the frames may be replaced by the live run mid-way; the playhead keeps its place
		if (!(await this.#frame(spread, token, (t) => (this.M = (this.frames?.reach ?? 0) * ease(t)))))
			return;
		this.phase = 'solve';
		if (!(await this.#frame(1000, token, (t) => (this.solved = ease(t))))) return;
		this.phase = 'choose';
		if (!(await this.#frame(750, token, (t) => (this.chose = ease(t))))) return;
		this.phase = 'done';
	}

	/** replay the question on screen */
	replay() {
		void this.play(this.frames, null);
	}

	/** the visitor drags the walk length: the timeline stops there (past the end: the solve) */
	seek(M: number) {
		this.#token++;
		cancelAnimationFrame(this.#raf);
		const R = this.frames?.reach ?? 0;
		this.M = Math.max(0, Math.min(R, M));
		this.solved = M > R ? Math.min(1, M - R) : 0;
		this.chose = M >= R + 1 ? 1 : 0;
		this.phase = 'paused';
	}

	#restart(p: Problem, edited: boolean) {
		this.problem = p;
		this.edited = edited;
		this.runs++;
		this.note = null;
		const tokens = encode(p, this.spec);
		if (!tokens) {
			this.note = `The input is full: the benchmark gives every question ${this.spec.n} tokens.`;
			return;
		}
		void this.play(null, this.#ask(tokens, this.towns));
	}

	/** show the question at `i` of this size's exported list */
	show(i: number) {
		const q = this.questions[i];
		if (!q) return;
		this.index = i;
		this.lookahead = q.lookahead;
		this.original = parse(q.tokens, this.spec);
		this.names = null;
		this.renaming = null;
		this.#restart(clone(this.original), false);
	}

	/** the question as asked, under the names on screen */
	#asked(): Problem {
		const to = this.names;
		if (!to) return clone(this.original);
		const r = (v: number) => to.get(v) ?? v;
		const p = this.problem;
		return {
			edges: p.edges.map(([u, v]) => [u, v]),
			start: p.start,
			goal: r(this.original.goal),
			path: this.original.path.map(r)
		};
	}

	/**
	 * New random names for every town and a new order for the roads: the same question, written differently. The layer
	 * runs again on the new tokens; the page checks that it chooses the same road under its new name.
	 */
	rename() {
		const before = this.run?.answer ?? null;
		const { problem, to: step } = renamed(this.problem, this.towns);
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
		this.#restart(problem, this.edited);
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

	/** maps of another size: its question nearest this lookahead */
	setTowns(V: Towns) {
		if (V === this.towns) return;
		const set = this.#sets.get(V);
		if (!set) return;
		this.towns = V;
		this.questions = set;
		const want = this.lookahead;
		const looks = set.map((q) => q.lookahead);
		const near = want === null ? null : (looks.find((L) => L !== null && L >= want) ?? looks[0]);
		const i = set.findIndex((q) => q.lookahead === (looks.includes(want) ? want : near));
		this.show(i >= 0 ? i : 0);
	}

	/** a town becomes the goal; the walks start from it */
	setGoal(v: number) {
		const p = this.#asked();
		if (v === this.problem.goal) return;
		if (v === current(p)) {
			this.note = `${v} is where the question stands; pick another town as the goal.`;
			return;
		}
		const asked = p.goal;
		p.goal = v;
		this.renaming = null;
		this.#restart(p, v !== asked);
		if (!goodSteps(p).length)
			this.note = `No road leads from ${current(p)} to ${v}; the layer answers anyway.`;
	}

	reset() {
		this.renaming = null;
		this.#restart(this.#asked(), false);
	}
}
