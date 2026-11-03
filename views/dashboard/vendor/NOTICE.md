# Bundled frontend dependencies

- `vue.global.prod.js`: Vue 3.5.22 global production distribution, https://unpkg.com/vue@3.5.22/dist/vue.global.prod.js. Copyright Vue contributors; MIT license. https://github.com/vuejs/core/blob/v3.5.22/LICENSE
- `liquid-glass.js`: GooseHyperGlass, the exact script served at https://glass.goose.cc.cd/liquid-glass.js (SHA-256 `6578d9337b0d7262ff2a3a1bd307119231151b25f7fc8f3fe6fc4da4aa778e44`), which this page previously loaded from that CDN. Usage documentation: https://glass.goose.cc.cd/doc/. The distribution carries no license header; upstream projects credited by the documentation: https://github.com/Kyant0/AndroidLiquidGlass, https://github.com/martin65536/liquid-glass-webgl, and https://github.com/aaaa-zhen/siri-glsl (Siri animation). Loaded only when the liquid-glass switch is on.
- `cytoscape.min.js`: Cytoscape.js 3.33.1, MIT license (`cytoscape.LICENSE`). Loaded when the personal network detail card is opened.

The original distributions are preserved; the page loads them from this directory instead of a CDN.
