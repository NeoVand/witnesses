/**
 * What the demo draws for one question: the model's answer, and after every layer the read-out applied to every pair,
 * as probabilities (each row a softmax over the names: "if u were the current vertex, which next step?"). The current
 * vertex's row is the model's own answer at that layer.
 *
 * A trace comes from the live model (worker.ts) or from a recording (opening.json, made by scripts/record.ts with the
 * same forward pass), stored as bytes.
 */
import { softmax, type Run } from './model.ts';

export interface Trace {
	tokens: number[];
	/** the names in the input, ascending: the model's order for rows and columns */
	names: number[];
	current: number;
	answer: number;
	/** after the start and after each layer: m × m row-softmax probabilities */
	maps: Float32Array[];
	/** milliseconds the forward pass took, for a live run */
	ms?: number;
	/** the current town's read-out after the last layer, one logit per name (live runs; a recording keeps only `maps`) */
	logits?: number[];
}

/** a trace of a run of either model (the 8 layers' run, or the repeated layer's) */
export function fromRun(
	r: Pick<Run, 'names' | 'current' | 'answer' | 'maps' | 'ms'>,
	tokens: number[]
): Trace {
	const m = r.names.length;
	const maps = r.maps.map((mp) => {
		const out = new Float32Array(m * m);
		for (let i = 0; i < m; i++) out.set(softmax(mp.subarray(i * m, i * m + m)), i * m);
		return out;
	});
	return { tokens, names: r.names, current: r.current, answer: r.answer, maps, ms: r.ms };
}

/** the current vertex's row of each map: the answer the model would give if it stopped after that layer */
export function readouts(t: Trace): Float32Array[] {
	const m = t.names.length;
	const i = t.names.indexOf(t.current);
	return t.maps.map((mp) => mp.slice(i * m, i * m + m));
}

export interface Packed {
	tokens: number[];
	names: number[];
	current: number;
	answer: number;
	/** base64 of one byte per probability (× 255), layer after layer */
	maps: string;
}

export function pack(t: Trace): Packed {
	const m = t.names.length;
	const bytes = new Uint8Array(t.maps.length * m * m);
	t.maps.forEach((mp, l) => mp.forEach((p, k) => (bytes[l * m * m + k] = Math.round(p * 255))));
	let s = '';
	for (const b of bytes) s += String.fromCharCode(b);
	return { tokens: t.tokens, names: t.names, current: t.current, answer: t.answer, maps: btoa(s) };
}

export function unpack(p: Packed): Trace {
	const m = p.names.length;
	const s = atob(p.maps);
	const L1 = s.length / (m * m);
	const maps = Array.from({ length: L1 }, (_, l) => {
		const out = new Float32Array(m * m);
		for (let k = 0; k < m * m; k++) out[k] = s.charCodeAt(l * m * m + k) / 255;
		return out;
	});
	return { tokens: p.tokens, names: p.names, current: p.current, answer: p.answer, maps };
}
