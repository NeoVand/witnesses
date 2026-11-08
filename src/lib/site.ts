/**
 * The site's few fixed facts: where it is published, the paper's PDF, and the links that appear only once they exist.
 * The paper's own title, byline and abstract are read from paper/paper.md, never typed here.
 */

/** where GitHub Pages publishes the site (the deploy workflow sets BASE_PATH=/witnesses); used for the social card */
export const SITE_URL = 'https://neovand.github.io/witnesses';

/** the paper's PDF in static/, copied from the author's build by scripts/sync-paper.mjs */
export const PDF = 'attention-witnesses.pdf';

/** the tutorial's PDF in static/, copied from the author's build by scripts/sync-tutorial.mjs */
export const TUTORIAL_PDF = 'attention-witnesses-tutorial.pdf';

/** the film: set its URL and the Film link appears on the project page */
export const FILM_URL: string | null = null;

/** the code: the Code links appear while this is set */
export const CODE_URL: string | null = 'https://github.com/NeoVand/witnesses';

/** the social card: a white-ground still of the project page's map, its last frame (scripts/shots.mjs) */
export const CARD = 'media/card.png';

/**
 * A title with a colon breaks after it, the subtitle set smaller on its own line (as the PDF's title page does):
 * `main` keeps the colon.
 */
export function splitTitle(title: string): { main: string; sub: string | null } {
	const i = title.indexOf(': ');
	return i > 0
		? { main: title.slice(0, i + 1), sub: title.slice(i + 2) }
		: { main: title, sub: null };
}

/** the description every page's meta tags give, in the paper's own words */
export const DESCRIPTION =
	'Each layer we study computes the partition function of a family of witnesses and reads it at a temperature. ' +
	'One propagator layer trained on 49-vertex graphs answers every question at every search depth on 250-vertex graphs.';
