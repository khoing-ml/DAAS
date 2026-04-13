# Self-Evolve Guidance: A Technical Way To Guide Diffusion Models Toward Better Search Regions [cite: 1]

## 0. Modified Project Plan (IR-SEG v2)

This plan translates the proposed improvement into an implementation-first roadmap:
- initialize K particles,
- compute timestep intermediate rewards in a steering window,
- split good/bad by dynamic threshold $\tau_t$,
- run M inner Stein updates per controlled timestep,
- refresh labels after each update,
- benchmark on intermediate-reward protocols.

### 0.1 Goal
- Build a closed-loop SEG variant that uses timestep intermediate rewards, not only final reward, to estimate $p(y=1|x_t)$ and steer particles more stably.

### 0.2 Core Algorithm Change
At each controlled timestep $t \in [t_{start}, t_{end}]$:
1. Produce latent particles $x_t^{(1..K)}$.
2. Compute intermediate rewards $r_t(x_t)$.
3. Compute threshold $\tau_t$ from the current reward batch.
4. Create good/bad split:
     - good: $\{x_t: r_t(x_t) \ge \tau_t\}$
     - bad: $\{x_t: r_t(x_t) < \tau_t\}$
5. Estimate $\log p(y=1|x_t)$ and its score using Nadaraya-Watson kernel regression.
6. Apply M Stein updates with score decomposition:
     $$\nabla_x \log p(x_t|y=1) = \nabla_x \log p(x_t) + \nabla_x \log p(y=1|x_t)$$
7. Recompute intermediate rewards and labels after each update.
8. Continue denoising to $t-1$.

### 0.3 Implementation Phases

Phase 1: Intermediate Reward Plumbing (1 week)
- Add an intermediate reward hook in the diffusion loop so rewards can be evaluated at selected timesteps.
- Add a reward cache keyed by (loop, t, particle_id).
- Keep final reward path unchanged for backward compatibility.

Deliverables:
- timestep reward tensor shape [K] available at each controlled step.
- no regression for existing final-only SEG runs.

Phase 2: Dynamic Threshold Schedule (1 week)
- Implement per-timestep threshold policies:
    - quantile ($q$),
    - top-k,
    - second-best fallback for tiny K,
    - monotonic clamp (optional): $\tau_t \leftarrow \max(\tau_t, \tau_{t+1})$.
- Log threshold trajectory $\tau_t$ and good/bad counts per t.

Deliverables:
- deterministic thresholding with fixed seed.
- complete threshold logs for each loop.

Phase 3: Nadaraya-Watson Good-Probability Estimator (1 to 2 weeks)
- Add NW estimator for $p(y=1|x_t)$ with RBF kernel in latent space.
- Add bandwidth choices:
    - median heuristic,
    - fixed scalar,
    - timestep-scaled bandwidth.
- Return both value and gradient wrt $x_t$ for steering.

Deliverables:
- stable estimator output on small batches.
- finite gradients, clipped by max norm.

Phase 4: Inner Stein Loop Per Timestep (1 week)
- For each controlled timestep, run M inner updates:
    - compute guided score,
    - update particles,
    - recompute intermediate rewards,
    - refresh good/bad labels.
- Add guards for empty-good or empty-bad sets.

Deliverables:
- no crash in edge partitions.
- measurable within-step reward gain in debug runs.

Phase 5: Benchmark IR Protocol (1 week)
- Create benchmark_ir protocol:
    - fixed prompts,
    - fixed seeds,
    - compute budgets matched across methods.
- Compare:
    - base sampling,
    - SEG final-only,
    - IR-SEG (this plan).

Metrics:
- mean/final reward,
- best-of-K,
- diversity (latent spread or embedding spread),
- reward improvement per controlled timestep,
- runtime and memory overhead.

Deliverables:
- benchmark table + plots.
- recommendation for default policy.

### 0.4 File-Level Task Mapping
- `daas/experiments/seg_runner.py`
    - add per-timestep intermediate reward evaluation,
    - add inner Stein loop M times per controlled timestep,
    - refresh labels after each inner update.
- `daas/diffusions/evolution/score_estimators.py`
    - add NW estimator for $p(y=1|x_t)$ and gradient.
- `daas/diffusions/evolution/thresholds.py`
    - add per-timestep dynamic threshold scheduler utilities.
- `daas/experiments/logging.py`
    - add logs: t, $\tau_t$, good_count, bad_count, reward stats before/after Stein.
- `daas/experiments/rewards.py`
    - expose intermediate reward API reusable at timestep t.
- `config/experiments/*.toml`
    - add ir_seg config block:
        - `start_time`, `end_time`, `inner_stein_steps`,
        - `tau_policy`, `tau_quantile`,
        - `nw_bandwidth`, `nw_eps`,
        - `benchmark_protocol=benchmark_ir`.

### 0.5 Validation Checklist
- Unit tests
    - threshold policy on tiny K.
    - NW gradients finite and bounded.
    - no update when timestep is outside control window.
- Smoke tests
    - short run with K=4, M=2, 5 to 10 denoise steps.
- Acceptance criteria
    - IR-SEG improves mean reward versus final-only SEG under same compute by a predefined margin.
    - diversity drop stays below predefined tolerance.

### 0.6 Risks and Mitigations
- Risk: noisy intermediate rewards at high-noise timesteps.
    - Mitigation: control only mid-noise window and smooth thresholds.
- Risk: kernel bandwidth instability.
    - Mitigation: median heuristic fallback + gradient clipping.
- Risk: mode collapse from aggressive steering.
    - Mitigation: keep Stein repulsive term active, cap update norm, track diversity online.

### 0.7 Milestone Summary
- M1: intermediate reward + threshold logging running end-to-end.
- M2: NW probability score integrated and stable.
- M3: inner Stein loop with label refresh complete.
- M4: benchmark_ir completed with ablation report.

## 1. Abstract [cite: 2]
Inference-time guidance for diffusion models is often framed as sampling from a reward-tilted distribution, where high-reward outputs are favored without retraining the base generator[cite: 3]. Existing approaches such as particle search and Sequential Monte Carlo (SMC) can improve reward but may collapse to a few modes and underuse the information encoded by collected trajectories[cite: 4]. We propose **Self-Evolve Guidance (SEG)**, a guidance mechanism that iteratively distills "good" and "bad" sample sets into a time-dependent steering vector field[cite: 5]. SEG combines reward-induced partitioning, score approximation of the good region, and Stein-style transport to push particles from low-quality areas toward high-quality yet diverse modes[cite: 6]. We present a practical algorithmic realization and discuss how SEG complements existing global search methods[cite: 7].

---

## 2. Introduction [cite: 8]
Classical search and optimization methods serve as a core mechanism for decision making under limited compute[cite: 9]. Recently, diffusion models have shown strong performance across generation and planning-like tasks, but their default sampling procedure does not always align outputs with test-time objectives such as reward maximization, constraint satisfaction, or verifier preferences[cite: 10]. As a result, users often need many trials to obtain acceptable samples[cite: 11]. Inference-time scaling methods therefore allocate additional test-time compute to guide diffusion sampling toward desirable outputs[cite: 12]. Existing approaches broadly combine local refinement and global exploration, including particle methods and SMC-style resampling[cite: 13]. These methods can increase sample quality, but they may also over-concentrate on a few modes and underuse information from previously collected trajectories[cite: 14]. This paper introduces Self-Evolve Guidance (SEG), a guidance framework that evolves as samples are collected[cite: 15]. Rather than using reward only for selection, SEG learns a time-dependent steering signal from the empirical structure of good and bad trajectories, then feeds this signal back into the denoising dynamics to improve both objective value and coverage[cite: 16].

---

## 3. Motivation [cite: 17]
Search-based inference scaling and SMC-style methods can produce high-quality samples, but two limitations are frequently observed:
* **Mode collapse and diversity loss**: Repeatedly prioritizing top-reward particles can over-concentrate mass in a few regions, reducing sample diversity[cite: 18, 19].
* **Purely stochastic exploration**: Many search updates rely on random branching/resampling and do not fully exploit geometric information in previously collected samples[cite: 18, 20].

Our motivation is to convert sample history into a reusable guidance signal[cite: 22]. Intuitively, once we observe which trajectories are "good" and "bad," we should construct a vector field that moves future particles away from bad regions and toward good regions, rather than restarting from random perturbations alone[cite: 23]. This leads to a self-improving inference loop: collect samples, estimate structure, steer sampling, and repeat[cite: 24].

---

## 4. Technical Framework: Self-Evolve Guidance [cite: 25]

### 4.1 Problem Setup [cite: 26]
Let $\epsilon_{\theta}(x_{t},t,c)$ denote a pretrained conditional diffusion model under prompt/condition $c$, with score approximation:
$$s_{\theta}(x_{t},t|c)\approx\nabla_{x_{t}}\log p_{t}(x_{t}|c)$$ [cite: 26, 27, 28]

At test time, we define a verifier/reward function $f(x_{0})$ and target a reward-tilted distribution:
$$\tilde{p}_{0}(x_{0})\propto p_{0}(x_{0})f(x_{0})^{\lambda}$$ [cite: 29, 30, 31]
where $p_{0}$ is the base distribution induced by the pretrained model and $\lambda>0$ controls guidance strength[cite: 32].

### 4.2 Particle Partitioning [cite: 34]
At inference, we sample $N$ trajectories $\{x_{t}^{(n)}\}_{n=1}^{N}$ from $t=T$ to $0$, obtain final samples $x_{0}^{(n)}$, and evaluate reward $r(x_{0}^{(n)})$[cite: 33, 34, 35]. Using a threshold $\tau$, define labels $y_n$:
$$y_{n}=\begin{cases}0,&r(x_{0}^{(n)})<\tau \text{ (bad)}\\ 1,&r(x_{0}^{(n)})\ge\tau \text{ (good)}\end{cases}$$ [cite: 35, 36, 37, 38, 39]

For each timestep $t$, we view particle states as drawn from two latent regions:
* **Good region**: $q_{t}(x|c)=p(x_{t}=x|y=1;c)$ [cite: 40, 41]
* **Bad region**: $b_{t}(x|c)=p(x_{t}=x|y=0;c)$ [cite: 40, 41]
with empirical sets $G_{t}$ and $B_{t}$ (good and bad particles, respectively)[cite: 42].

### 4.3 Stein Variational Transport [cite: 53]
The SEG objective is to transform current particles approximately distributed as $b_{t}(\cdot|c)$ toward $q_{t}(\cdot|c)$ while preserving multi-modality[cite: 44]. We seek a transport field $\phi_{t}$ such that for small step size $\eta_{t}$:
$$T_{t}(x)=x+\eta\phi_{t}(x)$$ [cite: 45, 46, 47]
Finally, SEG combines base score and evolution steering:
$$\hat{s}_{t}(x_{t})=s_{\theta}(x_{t},t|c)+\gamma_{t}\phi_{t}(x_{t})$$ [cite: 48, 49]
where $\gamma_{t}$ controls guidance strength over time[cite: 50].

Given particles $\{x_{t}^{(j)}\}_{j=1}^{M}$, we instantiate $\phi_{t}$ using a Stein variational vector field (SVVF):
$$\tilde{\phi}_{t}(x)=\frac{1}{M}\sum_{j=1}^{M}[k(x_{t}^{(j)},x)\nabla_{x_{t}^{(j)}}\log q_{t}(x_{t}^{(j)}|c)+\nabla_{x_{t}^{(j)}}k(x_{t}^{(j)},x)]$$ [cite: 53, 54, 55, 56]
with RBF kernel $k(x,x^{\prime})=\exp(-\|x-x^{\prime}\|_{2}^{2}/\sigma_{t})$[cite: 57]. The first term attracts particles toward high-goodness-density regions; the second repulsive term prevents collapse and improves diversity[cite: 57, 58].

### 4.4 Score Approximation [cite: 60]
We approximate the good-probability score $\nabla_{x_{t}}\log q_{t}(x_{t}|c)$ by weighted anchors from good terminal samples $g_{0}^{(i)}\in G_{0}$:
$$\nabla_{x_{t}}\log q_{t}(x_{t}|c)\approx\sum_{i=1}^{N_{G}}w_{i}(x_{t})\nabla_{x_{t}}\log p(x_{t}|x_{0}=g_{0}^{(i)})$$ [cite: 60, 61, 62, 63]
where $w_{i}(x_{t})=p(x_{t}|x_{0}=g_{0}^{(i)})/Z$ and $Z=\sum_{i}p(x_{t}|x_{0}=g_{0}^{(i)})$[cite: 64, 67]. Under standard diffusion forward marginals:
$$p(x_{t}|x_{0}=g_{0}^{(i)})=\mathcal{N}(x_{t};\sqrt{\overline{\alpha}_{t}}g_{0}^{(i)},(1-\overline{\alpha}_{t})I)$$ [cite: 65, 69]
which gives a closed-form gradient:
$$\nabla_{x_{t}}\log p(x_{t}|x_{0}=g_{0}^{(i)})=-\frac{x_{t}-\sqrt{\overline{\alpha}_{t}}g_{0}^{(i)}}{1-\overline{\alpha}_{t}}$$ [cite: 70, 71]

Particles are updated by $\tilde{x}_{t}^{(j)}=x_{t}^{(j)}+\eta_{t}\hat{\phi}_{t}(x_{t}^{(j)})$ followed by a denoising step using the guided score $\hat{s}_{t}$[cite: 72, 73, 77].

---

## 5. Algorithm 1: Self-Evolve Guidance (SEG) [cite: 80, 81]

**Inputs**: Diffusion model $f_{\theta}$, prompt $y$, warmup size $K$, loops $N$, pool size $M$, timesteps $(t_s, t_e)$, base threshold $\tau_0$, Stein step size $\eta$, Stein inner steps $S$, reward function $\mathcal{R}(\cdot, y)$.

1.  **Warmup Phase**:
    * Sample $K$ trajectories (no Stein) and decode final images $\{I_{i}^{(0)}\}_{i=1}^{K}$ with latents $\{x_{0,i}^{(0)}\}_{i=1}^{K}$.
    * Compute rewards $r_{i}^{(0)}=\mathcal{R}(I_{i}^{(0)},y)$.
    * Define Accepted set $\mathcal{A}^{(0)} \leftarrow \{x_{0,i}^{(0)}:r_{i}^{(0)}\ge\tau_{0}\}$ and Rejected set $\mathcal{Q}^{(0)} \leftarrow \{x_{0,i}^{(0)}:r_{i}^{(0)}<\tau_{0}\}$.
    * Compute initial Stein support/vector field from $\mathcal{A}^{(0)}$ under threshold $\tau_0$.

2.  **Self-Evolve Loop (for $l=1$ to $N$):**
    * **Re-noise**: Move particles in $\mathcal{Q}^{(l-1)}$ to timestep $t_s$ to obtain $x_{t_{s}}^{(l)}$.
    * **Guided Denoise**: Denoise from $t_s \rightarrow t_e$ with Stein updates using support $\mathcal{A}^{(l-1)}$.
    * **Inner Steps**: For each denoising step $t$, apply $S$ Stein steps with step schedule $\eta_{t}$.
    * **Evaluate**: Decode final images and latents for steered particles, producing candidates $\tilde{\chi}^{(l)}$.
    * **Form Pool**: Create current pool $\mathcal{P}^{(l)}$ from kept strong particles and $\tilde{\chi}^{(l)}$.
    * **Resample**: Compute final rewards $\{r_{j}^{(l)}\}$ and resample particles based on potential to reach the good region.
    * **Update**: Update threshold $\tau_{l+1} \leftarrow$ THRESHOLDUPDATE$(\{r_{j}^{(l)}\}, \tau_{l})$.
    * **Partition**: Update $\mathcal{A}^{(l)} \leftarrow \{x \in \mathcal{P}^{(l)}:r(x) \ge \tau_{l+1}\}$ and $\mathcal{Q}^{(l)} \leftarrow \mathcal{P}^{(l)} \backslash \mathcal{A}^{(l)}$.
    * **Recompute**: Update the Stein support/vector field from $\mathcal{A}^{(l)}$.

3.  **Return**: Final pool, reward history, thresholds, and accepted/rejected histories.

---

## 6. Planned Experiments [cite: 82]
* **Metrics**: Average reward, best-of-$N$ reward, diversity metrics (pairwise distance/coverage), and compute-normalized performance[cite: 84].
* **Baselines**: Direct sampling, SMC, and tree-search methods[cite: 83].
* **Hypothesis**: SEG improves reward at fixed compute while retaining higher diversity than resampling-only baselines[cite: 85].

---

## 7. Repository Mapping (Current Status)

This codebase already contains a strong SEG-ready foundation under `daas/diffusions/evolution/`:

* `controller.py`:
    * `EvolutionSteerer` combines thresholding, score estimation, Stein vector field, gating, and per-step activation.
    * `StepWindow` already supports limiting guidance to a timestep interval.
* `score_estimators.py`:
    * `GoodSetScoreEstimator` implements the weighted-good-anchor score approximation used in Section 4.4.
    * `KernelDensityScoreEstimator` provides a KDE variant for smoother good-region modeling.
* `stein.py` and `kernels.py`:
    * `SteinVectorField` computes attractive + repulsive SVGD terms.
* `thresholds.py`:
    * `FixedThreshold`, `QuantileThreshold`, `TopKThreshold`, `SecondBestThreshold` exist.
* `gating.py`:
    * `DensityRatioGate` already addresses the question: "what if a sample already looks good-like?"
* `trajectories.py`:
    * `TrajectoryRecorder` and `TrajectoryBatch` support logging full per-step states and partitioning by reward.

In short, the mathematics in Sections 4.2-4.4 are largely implemented as reusable components; the main remaining work is system integration and experiment protocol.

---

## 8. Required Modifications For DAAS Integration

To make SEG fully operational in experiment scripts, prioritize the following modifications:

1. Add a closed-loop SEG runner in `daas/experiments/builders.py` (or a dedicated `daas/experiments/seg_runner.py`) that executes:
     * warmup sampling,
     * reward evaluation,
     * accepted/rejected partitioning,
     * re-noise + guided denoise loops,
     * pool update and threshold update.

2. Expose SEG configuration in experiment TOML files (`config/experiments/*.toml`):
     * threshold policy (`quantile`, `topk`, `fixed`, `second_best`),
     * gate policy (`constant`, `density_ratio`),
     * score estimator (`good_set`, `kde`),
     * `guidance_scale`, `max_update_norm`, `step_window`, `inner_stein_steps`.

3. Add "re-noise rejected particles" utility in the diffusion pipeline path (`daas/simple.py` and/or experiment builder path) so $\mathcal{Q}^{(l-1)} \to x_{t_s}^{(l)}$ is explicit and reproducible.

4. Add robust threshold scheduling helper (new utility in `daas/diffusions/evolution/thresholds.py` or `daas/experiments/rewards.py`) to support:
     * static threshold,
     * quantile-per-loop,
     * monotonic non-decreasing threshold,
     * second-best fallback for small batch cases.

5. Add history logging for analysis in `daas/experiments/logging.py` and `daas/experiments/io.py`:
     * per-loop threshold,
     * good/bad counts,
     * reward distribution summary,
     * diversity proxy (latent pairwise distance / CLIP embedding spread).

6. Add experiment-level tests in `examples/` to validate:
     * guidance inactive outside `StepWindow`,
     * no crash when bad set is empty,
     * threshold policy behavior on tiny batches,
     * reward improvement trend over short smoke loops.

---

## 9. Clarified Design Decisions

* **Threshold choice**:
    * Default to `QuantileThreshold(0.75)` for stable good-set size.
    * Use `SecondBestThreshold` when batch size is very small.
* **When to steer**:
    * Prefer mid-noise timesteps (for example 20%-80% via `StepWindow.from_fractions`) to avoid destabilizing high-noise exploration and low-noise detail synthesis.
* **What if a sample is already good-like?**
    * Use `DensityRatioGate`; it suppresses updates when $\log q_t(x)$ is already high relative to bad density.
* **How to prevent mode collapse**:
    * Keep Stein repulsive term active,
    * avoid overly aggressive `guidance_scale`,
    * track diversity metrics alongside reward.