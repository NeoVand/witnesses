import tailwindcss from '@tailwindcss/vite';
import { defineConfig, type Plugin } from 'vitest/config';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { copyFileSync, existsSync, mkdirSync, readdirSync, unlinkSync, watch } from 'node:fs';
import { join } from 'node:path';

/**
 * The paper's and the tutorial's figures are drawn into paper/figures/out/ and tutorial/figures/out/ (SVG or PNG for
 * the page, PDF for LaTeX). The site serves the SVGs and PNGs of both from static/figures/, mirrored here when the dev
 * server or a build starts, so there is one copy in git and no figure that neither uses is served.
 */
function figures(): Plugin {
	const dirs = ['paper/figures/out', 'tutorial/figures/out'].map((d) =>
		join(import.meta.dirname, d)
	);
	const sync = () => {
		const to = join(import.meta.dirname, 'static/figures');
		mkdirSync(to, { recursive: true });
		const web: string[] = [];
		for (const from of dirs) {
			if (!existsSync(from)) continue;
			for (const f of readdirSync(from).filter((x) => /\.(svg|png)$/.test(x))) {
				copyFileSync(join(from, f), join(to, f));
				web.push(f);
			}
		}
		for (const f of readdirSync(to)) if (!web.includes(f)) unlinkSync(join(to, f));
	};
	return {
		name: 'paper-figures',
		buildStart: sync,
		configureServer() {
			sync();
			// figures drawn while the dev server runs are copied as they land
			for (const from of dirs) if (existsSync(from)) watch(from, () => sync()).unref();
		}
	};
}

export default defineConfig({
	server: {
		watch: { ignored: ['**/paper/tex/**', '**/paper/figures/**', '**/tutorial/figures/**'] }
	},
	plugins: [
		figures(),
		tailwindcss(),
		sveltekit({
			compilerOptions: {
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter({ fallback: '404.html' }),
			// GitHub Pages serves the site under /<repo> (/witnesses); the deploy workflow sets BASE_PATH
			paths: { base: (process.env.BASE_PATH ?? '') as '' | `/${string}` }
		})
	],
	test: {
		expect: { requireAssertions: true },
		environment: 'node',
		include: ['src/**/*.{test,spec}.{js,ts}']
	}
});
