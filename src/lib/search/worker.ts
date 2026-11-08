/**
 * The models run here, off the page's thread, so the drawing never waits for a forward pass (about 50 ms on a laptop
 * for the 8 layers on the largest 49-town graphs; the OR join repeated 9 times on 125 towns, about half a second; the
 * propagator on 250 towns, under 100 ms). Messages: `load` (a model's weights) and `run` (a question's tokens, with the
 * model, the map's size and, for the repeated layer, how many times to run it). The propagator answers `walks`: its
 * walk channel's partial sums, the solve, its gates and the read-out over the current town's roads.
 */
import {
	loadLoopWeights,
	loadPropWeights,
	loadWeights,
	run,
	runLoop,
	type Exported,
	type LoopExported,
	type LoopWeights,
	type PropExported,
	type PropWeights,
	type Weights
} from './model.ts';
import { spec } from './tokens.ts';
import { fromRun } from './trace.ts';
import { walkRun, type WalkRun } from './walkrun.ts';

export type { WalkRun };

export type ModelKind = 'stack' | 'loop';
export type ToWorker =
	| { kind: 'load'; model: ModelKind | 'prop'; url: string; V: number }
	| { kind: 'run'; id: number; model: ModelKind; V: number; iters: number; tokens: number[] }
	| { kind: 'walks'; id: number; V: number; tokens: number[] };
export type FromWorker =
	| { kind: 'loaded'; model: ModelKind | 'prop'; params: number }
	| { kind: 'failed'; model?: ModelKind | 'prop'; id?: number; message: string }
	| ({ kind: 'walked'; id: number } & WalkRun)
	| {
			kind: 'ran';
			id: number;
			names: number[];
			current: number;
			answer: number;
			maps: Float32Array[];
			/** the current town's read-out after the last layer: one logit per name */
			logits: number[];
			ms: number;
	  };

let stack: Weights | null = null;
let loopX: LoopExported | null = null;
const loops = new Map<number, LoopWeights>();
let propX: PropExported | null = null;
const props = new Map<number, PropWeights>();
const post = (m: FromWorker, transfer: Transferable[] = []) =>
	(self as unknown as Worker).postMessage(m, transfer);

/** the repeated layer's weights for questions of V vertices (its names come from one pool, at any size) */
function loopFor(V: number): LoopWeights {
	if (!loopX) throw new Error('run before load');
	let W = loops.get(V);
	if (!W) loops.set(V, (W = loadLoopWeights(loopX, spec(V))));
	return W;
}

/** the propagator's weights for questions of V vertices (names from one pool of 256, at any size) */
function propFor(V: number): PropWeights {
	if (!propX) throw new Error('run before load');
	let W = props.get(V);
	if (!W) props.set(V, (W = loadPropWeights(propX, spec(V))));
	return W;
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
	const m = e.data;
	try {
		if (m.kind === 'load') {
			const res = await fetch(m.url);
			if (!res.ok) throw new Error(`the weights could not be read (${res.status})`);
			if (m.model === 'prop') {
				propX = (await res.json()) as PropExported;
				post({ kind: 'loaded', model: 'prop', params: propFor(m.V).params });
			} else if (m.model === 'stack') {
				const x = (await res.json()) as Exported;
				stack = loadWeights(x, spec(m.V));
				post({
					kind: 'loaded',
					model: 'stack',
					params: x.weights.reduce((a, w) => a + w.data.length, 0)
				});
			} else {
				loopX = (await res.json()) as LoopExported;
				post({ kind: 'loaded', model: 'loop', params: loopFor(m.V).params });
			}
		} else if (m.kind === 'walks') {
			const t0 = performance.now();
			const w = walkRun(propFor(m.V), m.V, m.tokens);
			w.ms = performance.now() - t0;
			post({ kind: 'walked', id: m.id, ...w });
		} else if (m.kind === 'run') {
			let t;
			let logits: number[];
			if (m.model === 'stack') {
				if (!stack) throw new Error('run before load');
				const r = run(stack, m.tokens);
				logits = Array.from(r.readouts[r.readouts.length - 1]);
				t = fromRun(r, m.tokens);
			} else {
				const r = runLoop(loopFor(m.V), m.tokens, m.iters);
				logits = Array.from(r.readouts[r.readouts.length - 1]);
				t = fromRun(r, m.tokens);
			}
			post(
				{
					kind: 'ran',
					id: m.id,
					names: t.names,
					current: t.current,
					answer: t.answer,
					maps: t.maps,
					logits,
					ms: t.ms ?? 0
				},
				t.maps.map((a) => a.buffer)
			);
		}
	} catch (err) {
		post({
			kind: 'failed',
			model: m.kind === 'load' ? m.model : undefined,
			id: m.kind === 'run' || m.kind === 'walks' ? m.id : undefined,
			message: (err as Error).message
		});
	}
};
