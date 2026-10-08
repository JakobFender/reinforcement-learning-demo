// Sanity checks for the pure RL code embedded in index.html (<script id="rl-core">).
// Run with:  node tests/core.test.js
'use strict';
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const src = html.match(/<script id="rl-core">([\s\S]*?)<\/script>/)[1];
// Evaluate in the main realm (a vm context makes global lookups ~15x slower).
const C = new Function(src + '\n;return { BanditExperiment, CastlePredictor, LECTURE_EPISODES, castleTrueValues, sampleCastleEpisode, streamRng, CliffExperiment, cliffGreedyPath, CLIFF_COLS };')();
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: got ${a}, expected ${b} ± ${tol}`);

// --- Castle: lecture episodes ---------------------------------------------
{
  const p = new C.CastlePredictor({ gamma: 1, alphaTD: 0.2, mcMode: 'avg', alphaMC: 0.2, v0: 0 });
  for (const ep of C.LECTURE_EPISODES) { for (const tr of ep) p.tdStep(tr); p.mcEpisode(ep); }
  close(p.Vmc.Start, -1.5, 1e-9, 'MC Start'); close(p.Vmc.Castle, 7 / 3, 1e-9, 'MC Castle'); close(p.Vmc.Dragon, 10 / 3, 1e-9, 'MC Dragon');
  close(p.Vtd.Start, -1.77, 0.005, 'TD Start'); close(p.Vtd.Castle, -0.25, 0.005, 'TD Castle'); close(p.Vtd.Dragon, 1.68, 0.005, 'TD Dragon');
  console.log('castle lecture  MC', p.Vmc.Start, p.Vmc.Castle, p.Vmc.Dragon, ' TD', p.Vtd.Start, p.Vtd.Castle, p.Vtd.Dragon);
}
// --- Castle: true values -----------------------------------------------------
{
  const vb = C.castleTrueValues(1, 1), vf = C.castleTrueValues(0, 1), vr = C.castleTrueValues(0.5, 1);
  close(vb.Start, -1, 1e-9, 'vπ bridge'); close(vb.Dragon, 2, 1e-9, 'vπ Dragon'); close(vf.CastleBack, 7.5, 1e-9, 'vπ CastleBack');
  close(vf.Start, 4.95, 1e-9, 'vπ forest'); close(vr.Start, (4.95 - 1) / 2, 1e-9, 'vπ random');
  // MC estimate converges to vπ (forest policy)
  const p = new C.CastlePredictor({ gamma: 1, alphaTD: 0.05, mcMode: 'avg', alphaMC: 0.1, v0: 0 });
  const rng = C.streamRng(3, 7);
  for (let i = 0; i < 20000; i++) { const ep = C.sampleCastleEpisode(rng, 0); for (const tr of ep) p.tdStep(tr); p.mcEpisode(ep); }
  close(p.Vmc.Start, 4.95, 0.15, 'MC → vπ'); close(p.Vtd.Start, 4.95, 0.6, 'TD → vπ');
  console.log('castle forest  vπ', vf.Start, ' MC', p.Vmc.Start.toFixed(3), ' TD', p.Vtd.Start.toFixed(3));
}
// --- Bandit: Fig. 2.2 ----------------------------------------------------------
{
  const e = new C.BanditExperiment({ k: 10, steps: 1000, runs: 2000, seed: 1, agents: [0, 0.01, 0.1].map(eps => ({ eps, mode: 'avg', alpha: 0.1, q0: 0 })) });
  while (e.step());
  const tail = (arr) => arr.slice(900).reduce((a, b) => a + b, 0) / 100;
  const r = e.agents.map(a => tail(Array.from(a.avgReward))), o = e.agents.map(a => tail(Array.from(a.optFrac)));
  console.log('bandit avg reward (last 100):', r.map(x => x.toFixed(3)), ' %opt:', o.map(x => (100 * x).toFixed(1)));
  close(r[2], 1.4, 0.06, 'ε=0.1 reward'); close(r[0], 1.0, 0.08, 'greedy reward'); assert.ok(r[1] > r[0] && r[1] < r[2]);
}
// --- Cliff: Example 6.6 --------------------------------------------------------
{
  const e = new C.CliffExperiment({ runs: 20, episodes: 500, eps: 0.1, decay: false, alpha: 0.5, gamma: 1, seed: 1 });
  while (e.step());
  const tail = arr => arr.slice(300).reduce((a, b) => a + b, 0) / 200;
  const [s, q] = e.agents.map(a => tail(Array.from(a.returns)));
  console.log('cliff avg return eps 300-500: SARSA', s.toFixed(1), ' Q-learning', q.toFixed(1));
  close(s, -25, 6, 'SARSA'); close(q, -50, 8, 'Q-learning');
  const gq = C.cliffGreedyPath(e.agents[1].Qs[0]), gs = C.cliffGreedyPath(e.agents[0].Qs[0]);
  console.log('greedy path lengths: SARSA', gs.path.length - 1, gs.outcome, ' Q-learning', gq.path.length - 1, gq.outcome);
  assert.strictEqual(gq.outcome, 'goal'); assert.strictEqual(gq.path.length - 1, 13);
  assert.strictEqual(gs.outcome, 'goal'); assert.ok(gs.path.length - 1 > 13);
  // ε-decay: both approach −13
  const d = new C.CliffExperiment({ runs: 10, episodes: 500, eps: 0.1, decay: true, alpha: 0.5, gamma: 1, seed: 1 });
  while (d.step());
  console.log('decay final returns:', d.agents.map(a => a.returns[499].toFixed(1)));
}
console.log('All checks passed.');
