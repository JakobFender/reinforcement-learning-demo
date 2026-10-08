# Tabular RL Playground

Interactive single-file web demo (`index.html`, vanilla JS + Chart.js from a CDN, no build step) for the tabular
reinforcement-learning topics of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.):

1. **Multi-armed bandit**: ε-greedy agents on the 10-armed testbed, sample-average vs. constant α, optimistic
   initial values. The defaults reproduce Fig. 2.2, and a preset reproduces Fig. 2.3.
2. **Castle MDP**: first-visit Monte Carlo vs. TD(0) prediction on the same episodes, compared against the true
   v_π from the Bellman equations. Includes a replay of the four lecture episodes with an automatic check.
3. **Cliff walking**: SARSA vs. Q-learning (Example 6.6), with greedy-policy arrows, a max-Q heatmap, the greedy
   path, and an animated "Watch" episode.

Every run uses a seeded PRNG (mulberry32), so the same seed and settings give the same result.

## Run locally

Open `index.html` in a browser. Nothing needs to be installed.

## Publish on GitHub Pages

*Settings → Pages → Build and deployment → Source: "Deploy from a branch"*, then select the branch (e.g. `main`)
and folder `/ (root)`. The site appears at `https://<user>.github.io/reinforcement-learning-demo/`.

## Tests

The algorithms are in `<script id="rl-core">` inside `index.html` and don't touch the DOM. `tests/core.test.js`
extracts that script and checks:

- the lecture values (MC −1.5 / 2.33 / 3.33; TD −1.77 / −0.25 / 1.68)
- the true v_π
- Fig. 2.2 (ε = 0.1 ≈ 1.4, greedy ≈ 1.0)
- Example 6.6 (SARSA ≈ −25, Q-learning ≈ −50, Q-learning takes the 13-step cliff-edge path)

```
node tests/core.test.js
```
