/**
 * The project page's map, computed at build time: the trained propagator (static/model/prop_v49.json, the forward pass
 * prop.spec.ts checks against the training code) run on the 250-vertex test question (static/model/graphs_v250.json),
 * its walk channel's partial sums x_M = Σ_{m ≤ M} s Gᵐ until the walks from the goal reach the start, and the solve.
 * Nothing on the map is drawn by hand: every town's light is the model's own number.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { loadPropWeights, type PropExported } from './model.ts';
import { parse, spec } from './tokens.ts';
import type { MapLayout } from './map.ts';
import type { WalkFrames } from './walks.ts';
import type { Problem } from './tokens.ts';
import { walkRun, type WalkRun } from './walkrun.ts';

export interface HeroData {
	layout: MapLayout;
	question: Problem;
	tokens: number[];
	frames: WalkFrames;
	/** the whole run: frames, gates and the read-out over the start's roads */
	run: WalkRun;
	/** the question's lookahead and size, as the benchmark's test set gives them */
	lookahead: number;
	towns: number;
}

let cached: HeroData | null = null;

export function heroData(root = process.cwd()): HeroData {
	if (cached) return cached;
	const read = (f: string) => JSON.parse(readFileSync(resolve(root, f), 'utf8'));
	const S = spec(250);
	const W = loadPropWeights(read('static/model/prop_v49.json') as PropExported, S);
	const q = (
		read('static/model/graphs_v250.json') as {
			questions: { tokens: number[]; lookahead: number }[];
		}
	).questions[0];
	// the map's water names are not drawn (the maps label only towns), so they are not shipped either
	const layout = { ...(read('static/maps250/0.json') as MapLayout), names: [] };
	const question = parse(q.tokens, S);
	const run = walkRun(W, 250, q.tokens);
	cached = {
		layout,
		question,
		tokens: q.tokens,
		frames: run.frames,
		run,
		lookahead: q.lookahead,
		towns: run.names.length
	};
	return cached;
}
