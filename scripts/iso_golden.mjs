// SPDX-License-Identifier: AGPL-3.0-only
// Writes tests/fixtures/iso_golden.json: SVG strings from the anatomy kit's own iso-kit.mjs for a
// fixed set of shapes, so tests/iso_golden.rs can check the Rust port (engine::iso) byte for
// byte. Usage: node scripts/iso_golden.mjs PATH/TO/anatomy/kit/iso-kit.mjs
// With --bench N, times N builds of the bench workload instead and prints milliseconds.
import { writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const kitPath = process.argv[2];
if (!kitPath) {
  console.error("usage: node scripts/iso_golden.mjs <iso-kit.mjs> [--bench N]");
  process.exit(2);
}
const K = await import(pathToFileURL(kitPath).href);

const plans = [
  { x: 0, y: 0, w: 40, d: 30, r: 0 },
  { x: -10, y: 5, w: 60, d: 24, r: 6 },
  { x: 3.3, y: -7.7, w: 12.5, d: 12.5, r: 6.25 },
];
const projections = [
  { origin: [200, 150], scale: 2, azimuth: 45, elevation: 30 },
  { origin: [320.5, 180.25], scale: 3.7, azimuth: 35, elevation: 30 },
  { origin: [100, 220], scale: 1.3, azimuth: 60, elevation: 44 },
];

function cases() {
  const out = [];
  projections.forEach((P, pi) => {
    plans.forEach((plan, i) => {
      out.push({ name: `slab p${pi} plan${i}`, svg: K.solidSvg(K.slabOf(plan, 2, 10 + i, P, 8, i * 0.6)) });
      out.push({ name: `slab-lit-halo p${pi} plan${i}`, svg: K.solidSvg(K.slabOf(plan, 0, 4, P), { tone: "lo", halo: true, lit: true }) });
      out.push({ name: `outline p${pi} plan${i}`, svg: K.planOutline(K.insetPlan(plan, 3), 12, P) });
    });
    out.push({ name: `cylinder p${pi}`, svg: K.solidSvg(K.cylinder(10, 12, 8.5, 1, 22, P, 32, 0.8), { crease: "mid" }) });
    out.push({ name: `thin-cylinder p${pi}`, svg: K.solidSvg(K.cylinder(-4, 2, 0.55, 0, 0.4, P, 12), { flat: true, tone: "mid" }) });
    out.push({ name: `segment p${pi}`, svg: K.lineSvg(K.segment([0, 0, 0], [12.3, -4.4, 9.9], P), { tone: "mid" }) });
    out.push({ name: `dots p${pi}`, svg: K.dotsSvg([K.iso([1, 2, 3], P), K.iso([-4.25, 7.5, 0], P)], { size: 0.5, tone: "hi" }) });
    out.push({ name: `wire p${pi}`, svg: K.wireSvg(K.iso([0, 0, 5], P), K.iso([30, 10, 5], P), { via: [K.iso([15, 0, 5], P)], tone: "lo" }) });
    out.push({ name: `extrude-concave p${pi}`, svg: K.solidSvg(K.extrude([[0, 0], [20, 0], [20, 6], [8, 6], [8, 18], [0, 18]], 0, 7, P, { convex: false })) });
  });
  const fit = K.fitProjection([[0, 0, 0], [100, 60, 0], [0, 60, 40], [100, 0, 40]], 600, 400, { pad: 24, azimuth: 40 });
  out.push({ name: "fit", svg: JSON.stringify(fit) });
  out.push({ name: "figure", svg: K.figureSvg({ width: 600, height: 400, label: "A <test> & figure", body: "<g/>" }) });
  out.push({ name: "plate", svg: K.plateHtml({ fig: "Fig 1", title: "Rig", hint: "Drag", readout: "0.42 mm", caption: "Caption & more", body: "<svg/>" }) });
  return out;
}

const bench = process.argv.indexOf("--bench");
if (bench > 0) {
  const n = Number(process.argv[bench + 1] || 1000);
  cases(); // warm up the JIT
  const t0 = performance.now();
  let bytes = 0;
  for (let i = 0; i < n; i++) for (const c of cases()) bytes += c.svg.length;
  const ms = performance.now() - t0;
  console.log(JSON.stringify({ runtime: `node ${process.version}`, iterations: n, ms, bytes }));
} else {
  const file = new URL("../tests/fixtures/iso_golden.json", import.meta.url);
  writeFileSync(file, JSON.stringify(cases(), null, 1) + "\n");
  console.log("wrote tests/fixtures/iso_golden.json");
}
