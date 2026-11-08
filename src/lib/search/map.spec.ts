import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { length, lineD, newRoad, placeLabels, route, shapeD, type MapLayout } from './map.ts';
import { parse, spec, vertices } from './tokens.ts';

const finite = (p: readonly number[]) => p.every((x) => Number.isFinite(x));

// the 49-town questions' maps (static/maps/) and the 125-town ones (static/maps125/)
for (const [V, dir] of [
	[49, 'maps'],
	[125, 'maps125']
] as const) {
	const S = spec(V);
	const { questions } = JSON.parse(readFileSync(`static/model/graphs_v${V}.json`, 'utf8')) as {
		questions: { lookahead: number | null; tokens: number[]; answer: number }[];
	};
	const maps = questions.map(
		(_, i) => JSON.parse(readFileSync(`static/${dir}/${i}.json`, 'utf8')) as MapLayout
	);

	describe(`the ${V}-town questions’ maps`, () => {
		it('draw every town and every road of their question, each road from its first town to its second', () => {
			questions.forEach((q, i) => {
				const p = parse(q.tokens, S);
				const m = maps[i];
				expect(m.start).toBe(p.start);
				expect(m.goal).toBe(p.goal);
				expect(
					Object.keys(m.towns)
						.map(Number)
						.sort((a, b) => a - b)
				).toEqual(vertices(p).sort((a, b) => a - b));
				const roads = new Map(m.roads.map((r) => [`${r.edge[0]}>${r.edge[1]}`, r.path]));
				expect(roads.size).toBe(p.edges.length);
				for (const [u, v] of p.edges) {
					const path = roads.get(`${u}>${v}`)!;
					expect(path, `question ${i}: road ${u} > ${v}`).toBeDefined();
					const [a, b] = [m.towns[String(u)], m.towns[String(v)]];
					expect(Math.hypot(path[0][0] - a[0], path[0][1] - a[1])).toBeLessThan(0.3);
					const e = path[path.length - 1];
					expect(Math.hypot(e[0] - b[0], e[1] - b[1])).toBeLessThan(0.3);
					expect(lineD(path).length).toBeGreaterThan(4);
				}
			});
		});

		it('hold finite geometry on the sheet, and land that draws', () => {
			for (const m of maps) {
				const [W, H] = m.size;
				for (const p of Object.values(m.towns)) {
					expect(finite(p)).toBe(true);
					expect(p[0] > 0 && p[0] < W && p[1] > 0 && p[1] < H).toBe(true);
				}
				for (const r of m.roads) expect(r.path.every(finite)).toBe(true);
				for (const s of [...m.water.lakes, ...m.woods, ...m.islands])
					expect(shapeD(s).length).toBeGreaterThan(8);
				for (const r of m.water.rivers) expect(r.outline.every(finite)).toBe(true);
			}
		});

		it('cross roads only as overpasses, and only where the graph is not planar', () => {
			for (const m of maps) {
				if (m.planar) expect(m.overpasses).toHaveLength(0);
				const keys = new Set(m.roads.map((r) => `${r.edge[0]}>${r.edge[1]}`));
				for (const o of m.overpasses) {
					expect(keys.has(`${o.over[0]}>${o.over[1]}`)).toBe(true);
					expect(keys.has(`${o.under[0]}>${o.under[1]}`)).toBe(true);
				}
				for (const b of m.bridges) expect(keys.has(`${b.road[0]}>${b.road[1]}`)).toBe(true);
			}
		});

		it('keep every town’s number at the drawing size they were laid out for', () => {
			for (const m of maps) {
				const k = m.demo.px / m.size[0];
				const labels = placeLabels(m, k, 1, [m.goal, m.current], new Map());
				expect(labels.size).toBe(Object.keys(m.towns).length);
			}
		});
	});
}

describe('the map’s helpers', () => {
	it('finds a shortest directed route', () => {
		const edges: [number, number][] = [
			[1, 2],
			[2, 3],
			[1, 4],
			[4, 3],
			[3, 5]
		];
		expect(route(edges, 1, 5)).toHaveLength(4);
		expect(route(edges, 5, 1)).toBeNull();
	});

	it('draws a new road from its first town to its second, marking the stretches over water', () => {
		const m = JSON.parse(readFileSync('static/maps/64.json', 'utf8')) as MapLayout;
		const a = m.towns[String(m.start)];
		const b = m.towns[String(m.goal)];
		const r = newRoad(m, a, b);
		expect(r.path[0]).toEqual(a);
		expect(Math.hypot(r.path.at(-1)![0] - b[0], r.path.at(-1)![1] - b[1])).toBeLessThan(1e-9);
		expect(length(r.path)).toBeGreaterThanOrEqual(Math.hypot(b[0] - a[0], b[1] - a[1]) - 1e-9);
		for (const run of r.wet) expect(run.length).toBeGreaterThan(0);
	});
});
