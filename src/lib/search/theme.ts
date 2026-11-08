/**
 * The site's palette, Cobalt and Rose, for the canvases and SVG the demo draws (layout.css holds the same colours as
 * CSS tokens). Rose is the pair state and its answer; cobalt the strongest alternative; everything else ink on white.
 */
export const PALETTE = {
	paper: '#FFFFFF',
	tint: '#F2F5FB',
	ink: '#1F2A44',
	inkSoft: '#4C5B78',
	blueGrey: '#7D8BA8',
	mist: '#BAC5D9',
	edge: '#DDE3EE',
	rose: '#D6336C',
	roseLight: '#F1A3BC',
	cobalt: '#2557A7',
	cobaltLight: '#9DB8E0',
	// the map: pale water with a darker shore, warm-grey woods, grey roads (the paper's map figures)
	water: '#E6ECF8',
	shore: '#C3CEE5',
	waterInk: '#7C90BC',
	wood: '#F2F1EE',
	road: '#BAC1CD',
	roadInk: '#8E97A8'
} as const;

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const ROSE = hex(PALETTE.rose);
const COBALT = hex(PALETTE.cobalt);

/** white to rose by p in [0, 1], with a gentle curve so small probabilities stay visible */
export function roseRamp(p: number): [number, number, number] {
	const t = Math.pow(Math.max(0, Math.min(1, p)), 0.7);
	return ROSE.map((c) => Math.round(255 + (c - 255) * t)) as [number, number, number];
}

export function cobaltRamp(p: number): [number, number, number] {
	const t = Math.pow(Math.max(0, Math.min(1, p)), 0.7);
	return COBALT.map((c) => Math.round(255 + (c - 255) * t)) as [number, number, number];
}

export const rgb = ([r, g, b]: [number, number, number]) => `rgb(${r} ${g} ${b})`;

/**
 * A town's tint by its read-out score c (about the row's mean): rose above, cobalt below, white near zero; full at
 * |c| = 3 (a factor of twenty in probability).
 */
export function diverging(c: number): string {
	const t = Math.pow(Math.min(1, Math.abs(c) / 3), 0.8);
	const to = c >= 0 ? ROSE : COBALT;
	const k = c >= 0 ? t : t * 0.85;
	return rgb(to.map((x) => Math.round(255 + (x - 255) * k)) as [number, number, number]);
}

/** the one easing the page moves with (layout.css --ease) */
export const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
