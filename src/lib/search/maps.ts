/**
 * The questions' maps (static/maps/<question>.json for the 49-town questions, static/maps125/ for the 125-town ones,
 * laid out offline), fetched once each and kept.
 */
import { asset } from '$app/paths';
import type { MapLayout } from './map.ts';

const cache = new Map<string, Promise<MapLayout>>();

export function loadMap(index: number, size = 49): Promise<MapLayout> {
	const name = `${size === 49 ? 'maps' : `maps${size}`}/${index}.json`;
	let p = cache.get(name);
	if (!p) {
		// asset() knows every static file by name; the index is checked against the exported list by the caller
		p = fetch(asset(name as Parameters<typeof asset>[0])).then((r) => {
			if (!r.ok)
				throw new Error(`the map of question ${index + 1} could not be read (${r.status})`);
			return r.json() as Promise<MapLayout>;
		});
		p.catch(() => cache.delete(name));
		cache.set(name, p);
	}
	return p;
}
