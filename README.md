# Calibration Behaviour of Diffusion Models Under Distribution Shift

Code and results for a master's thesis investigating whether diffusion posterior samplers report honest uncertainty when their learned prior does not match the input.

**Arati Dipak Sutar** · MSc Data Science, University of Europe for Applied Sciences
Supervisors: Prof. Dr. Iftikhar Ahmed, Prof. Shan Faiz

## Repository
preliminary/reproduce_results.ipynb reproduces the preliminary results
phase_a/run_sampling.ipynb generates the six-level samples
phase_a/analyse_levels.ipynb computes the Phase A results


Sample images and model weights are not redistributed. See Reproducing.

---

## The question

Diffusion models used as image priors for inverse problems produce sharp, confident reconstructions even when the observation lies far outside their training distribution. They fill unconstrained regions with plausible detail drawn from the prior and give no signal that the detail is invented.

A model trained only on human faces, given a photograph of a tree at 4× super-resolution, reconstructs it with recognisable facial structure — in every independent sample.

Uncertainty in a posterior sampler is not expressed within one reconstruction; it lives in the variation across many. So the question is whether that variation grows in proportion to the actual error when the prior stops fitting.

## Two failure modes, often conflated

Bayesian inversion factorises as `p(x|y) ∝ p(y|x) · p(x)`. Two independent things can break.

| | What breaks | Can the model detect it? | Effect on calibration |
|---|---|---|---|
| **Ambiguity** | likelihood — the observation is uninformative | yes | none |
| **Prior mismatch** | prior — the model has never seen this content | **no** | degrades |

Ambiguity is a known unknown. Prior mismatch is an unknown unknown: nothing in the sampling procedure can report that the learned prior does not apply.

---

## Metrics

**Spread-to-error ratio.** For k samples of one input with ground truth T:

```
spread = mean over all pairs (i,j) of  mean|Sᵢ − Sⱼ|
error  = mean over all i         of    mean|Sᵢ − T|
ratio  = spread / error
```

If the sampler is calibrated, the truth is itself a valid draw from the posterior, so both quantities measure the separation of two independent draws from one distribution and the ratio should be near 1. Below 1 means samples agree with each other more than they agree with the truth.

**Conformal coverage.** Per pixel, interval `mean(S) ± λ·sd(S)`. λ is calibrated on in-distribution data to a 90% nominal level, then applied unchanged to shifted data.

**Gradient-restricted coverage.** The same, computed only over the highest-gradient fraction of pixels.

---

## Results

19 images (10 faces, 9 non-faces), 6 samples each, ≈60 GPU hours on a single Tesla T4.

### Ambiguity does not degrade calibration

| Downscale | Input | Mean ratio | sd |
|---|---|---|---|
| 4 | 64×64 | 0.864 | 0.072 |
| 16 | 16×16 | 0.984 | 0.057 |

Making the inverse problem more ill-posed moves calibration *toward* honesty. A correct prior with an exact likelihood yields a correct posterior regardless of how broad that posterior is. This is the negative control.

### Prior mismatch does

| Group | Mean spread | Mean error | Mean ratio | sd |
|---|---|---|---|---|
| Faces | 9.75 | 11.39 | 0.864 | 0.072 |
| Non-faces | 7.84 | 11.01 | 0.747 | 0.131 |

Samples cluster more tightly while remaining just as far from the truth.

### The two axes interact

| Face − non-face ratio gap | Downscale 4 | Downscale 16 |
|---|---|---|
| | 0.117 | 0.083 |

Prior mismatch is most detectable when the inverse problem is least ambiguous. Under strong ambiguity the posterior is legitimately broad, and that breadth conceals the mismatch.

### Not an artefact of the guidance scale

| Guidance | Faces | Non-faces | Gap |
|---|---|---|---|
| 0.1 | 0.894 | 0.867 | +0.027 |
| 0.3 | 0.884 | 0.709 | +0.175 |
| 0.6 | 0.869 | 0.664 | +0.205 |
| 1.0 | 0.889 | 0.712 | +0.177 |

Spread and error scale by the same factor as guidance varies, so the ratio is invariant. At guidance 0.1 the reconstruction error is 2.75× higher — samples have stopped satisfying the measurement constraint — so that setting is degenerate and excluded.

No value of the guidance scale restores calibration under prior mismatch.

### Pooled conformal coverage does not detect the failure

λ = 3.12 at downscale 4, calibrated on faces to 90%.

| | Downscale 4 | Downscale 16 |
|---|---|---|
| Faces | 0.900 | 0.900 |
| Non-faces | 0.869 | 0.868 |
| Gap | **0.031** | **0.032** |

This refuted the original plan, which was to demonstrate a coverage collapse and then correct it.

### Gradient-restricted coverage does

| Mask | Faces | Non-faces | Gap |
|---|---|---|---|
| All pixels | 0.900 | 0.869 | +0.031 |
| Top 50% gradient | 0.893 | 0.825 | +0.068 |
| Top 20% gradient | 0.883 | 0.760 | +0.123 |
| **Top 10% gradient** | **0.881** | **0.732** | **+0.150** |
| Top 5% gradient | 0.881 | 0.727 | +0.154 |
| Top 2% gradient | 0.879 | 0.726 | +0.152 |

The separation grows monotonically and saturates below 10%, so the threshold is justified by the shape of the curve rather than selected to maximise the effect.

**Controls** — all masks select exactly 10% of pixels:

| Mask | Faces | Non-faces | Gap |
|---|---|---|---|
| Top gradient | 0.881 | 0.732 | **+0.150** |
| Random | 0.900 | 0.871 | +0.029 |
| Bottom gradient | 0.901 | 0.924 | −0.023 |

A random mask of identical size reproduces the pooled gap, so the effect is not an artefact of the reduced pixel count. The lowest-gradient mask shows no gap at all. Coverage on faces is essentially unchanged by the restriction (0.900 → 0.881), so the mask isolates the failure rather than depressing all measurements.

Bootstrap over images (5,000 resamples): 95% CI for the top-10% gap **[0.094, 0.221]**, excluding zero.

### Mechanism: the failure sits in unconstrained spatial frequencies

Downsampling is a low-pass operation. A 64×64 measurement of a 256×256 image carries frequencies up to roughly 32 cycles per image width. Below that the likelihood constrains the reconstruction; above it the prior alone determines the output.

Decomposing spread and error into radial frequency bands:

| Band (cyc/img) | \[d4\] Faces | \[d4\] Non-f. | \[d4\] Gap | \[d16\] Faces | \[d16\] Non-f. | \[d16\] Gap |
|---|---|---|---|---|---|---|
| 0–8 | 1.022 | 1.051 | −0.028 | 1.118 | 1.081 | +0.037 |
| 8–16 | 0.996 | 0.904 | +0.091 | 0.931 | 0.869 | +0.063 |
| 16–24 | 0.884 | 0.720 | +0.164 | 0.917 | 0.846 | +0.071 |
| 24–32 | 0.837 | 0.608 | +0.229 | 0.906 | 0.812 | +0.094 |
| 32–48 | 0.812 | 0.554 | +0.258 | 0.882 | 0.793 | +0.089 |
| 48–64 | 0.786 | 0.462 | +0.325 | 0.878 | 0.727 | +0.151 |
| 64–96 | 0.773 | 0.382 | +0.391 | 0.878 | 0.660 | +0.218 |
| 96–128 | 0.832 | **0.363** | **+0.469** | 0.948 | 0.672 | +0.276 |

**What replicates.** The shape. In both configurations the ratio is near unity for both groups at the coarsest band and diverges monotonically thereafter, without exception across eight bands. At the coarsest scale the sampler is calibrated even out of distribution, because the measurement determines that content regardless of prior validity.

**What does not.** The prediction that a coarser measurement would move the divergence to lower frequencies. It does not — the gap is smaller at every band under downscale 16. The account in which the transition tracks the measurement cutoff is wrong.

**Revised reading.** Two effects compete as the measurement coarsens: the prior-dominated band widens, but the posterior also becomes genuinely broader, and that legitimate breadth conceals the mismatch. The second dominates. Supporting evidence: at downscale 16 the in-distribution ratio in the coarsest band is 1.118, above unity, indicating underconfidence.

This explains the gradient result. Gradient magnitude is a spatial proxy for high-frequency content, so the gradient mask selects the region where the prior operates without constraint.

## ## Phase A — graded severity axis

The preliminary work used a binary shift axis: faces against non-faces, two points. Phase A replaces it with six levels of natural photographs, each removing one property the prior depends on.

An earlier attempt using linear blends between a face and a tree was abandoned — intermediate blends are double exposures no camera could produce, so the axis mixed "distance from the training distribution" with "unnaturalness".

### The six levels

| Level | Content | Still has | Just lost | Source |
|---|---|---|---|---|
| 0 | FFHQ faces | everything | — | FFHQ-256 |
| 1 | CelebA-HQ portraits | human face, aligned | FFHQ's capture pipeline | CelebA-HQ 256×256 |
| 2 | Animal faces | face geometry | human species | AFHQ |
| 3 | Painted portraits | face geometry, human | photographic texture | Portrait Paintings |
| 4 | Single objects | centred subject, plain background | the face | own set |
| 5 | Landscapes | nothing | composition too | Landscape Pictures |

Ten images per level, six samples each. 360 samples, ≈23 GPU hours on a single T4.

All settings identical to the preliminary work: `scale_factor` 4, σ = 0.05, guidance 0.3, 1000 denoising steps, unmodified DPS, pretrained FFHQ checkpoint, no retraining. Only the image content changes between levels.

Levels 2, 3 and 5 take **evenly spaced** files from the sorted list rather than the first ten, so one species or one painter does not dominate. Deterministic, and no hand-picking.

### How the sampling was run

`phase_a/run_sampling.ipynb` was committed three times with `LEVELS` changed each time:

| Commit | `LEVELS` | Duration |
|---|---|---|
| 1 | `[0, 1, 2]` | 678.8 min |
| 2 | `[3, 4]` | ≈450 min |
| 3 | `[5]` | ≈230 min |

Kaggle cancels any committed job exceeding 12 hours. Three levels took 11.3 hours, leaving 42 minutes of margin, so later commits ran two levels at a time.

Commits 2 and 3 begin by restoring the previous run's samples from its notebook output; the `if all(...exists...)` check then skips anything already on disk. The outputs shown in the notebook are from commit 1.

### Spread-to-error ratio

| Level | Content | Spread | Error | Ratio | sd |
|---|---|---|---|---|---|
| 0 | FFHQ faces | 9.81 | 11.20 | **0.881** | 0.051 |
| 1 | CelebA portraits | 8.51 | 9.70 | **0.884** | 0.067 |
| 2 | Animal faces | 12.85 | 16.84 | **0.774** | 0.094 |
| 3 | Painted portraits | 11.06 | 17.80 | **0.624** | 0.087 |
| 4 | Single objects | 7.97 | 11.45 | **0.747** | 0.158 |
| 5 | Landscapes | 8.31 | 15.76 | **0.607** | 0.142 |

### Coverage

λ = 2.99, calibrated on level 0 to a 90% nominal level, then applied unchanged.

| Level | Pooled | Top 10% | Bottom 10% | Random 10% |
|---|---|---|---|---|
| 0 | 0.900 | 0.881 | 0.908 | 0.901 |
| 1 | 0.896 | 0.881 | 0.905 | 0.894 |
| 2 | 0.845 | 0.784 | 0.877 | 0.846 |
| 3 | 0.739 | 0.687 | 0.776 | 0.737 |
| 4 | 0.854 | 0.738 | 0.903 | 0.853 |
| 5 | 0.729 | 0.671 | 0.759 | 0.730 |

Gap from level 0:

| Level | Pooled | Top 10% | Improvement |
|---|---|---|---|
| 1 | +0.004 | −0.000 | — |
| 2 | +0.055 | +0.097 | 1.8× |
| 3 | +0.161 | +0.194 | 1.2× |
| 4 | +0.046 | **+0.143** | **3.1×** |
| 5 | +0.171 | +0.210 | 1.2× |

### Measured distance from the training distribution

Computed from image statistics the prior depends on: mean and standard deviation per colour channel, horizontal and vertical gradient magnitude, and log power in four radial frequency bands, each standardised against the level-0 distribution.

| Level | Content | Distance | sd |
|---|---|---|---|
| 0 | FFHQ faces | 0.95 | 0.30 |
| 1 | CelebA portraits | 1.47 | 0.55 |
| 2 | Animal faces | 1.62 | 0.81 |
| 4 | Single objects | 1.75 | 0.48 |
| 5 | Landscapes | 1.84 | 0.65 |
| 3 | Painted portraits | **1.89** | 0.73 |

Measured ordering: **L0 < L1 < L2 < L4 < L5 < L3**
Assumed ordering: L0 < L1 < L2 < L3 < L4 < L5

### Correlations across all 60 images

| Metric | r | After removing the single worst point |
|---|---|---|
| Spread-to-error ratio | −0.470 | −0.423 |
| Coverage, top 10% gradient | **−0.521** | −0.482 |
| Coverage, pooled | −0.427 | −0.377 |

Greater distance, worse calibration. Robust — removing any single image moves each correlation by about 0.04.

---

### What Phase A established

**Replication.** Level 0 gives 0.881 against 0.864 in the preliminary work, within the 0.019 run-to-run variation already measured. Level 4 matches exactly at 0.747. Separate runs, months apart.

**A different dataset is not a shift.** Level 1 is CelebA-HQ — different photographs, different collection, different capture conditions, same subject matter.

```
ratio          0.881 → 0.884   (unchanged)
pooled gap             +0.004
top-10% gap            −0.000
```

Completely unmoved. The sampler responds to content, not provenance. This rules out an obvious alternative explanation for the whole effect.

**Two distinct failure modes.**

| Group | Spread vs level 0 | Error | Interpretation |
|---|---|---|---|
| Animals, paintings | **rises** (12.85, 11.06 vs 9.81) | rises more | prior finds face-like structure, reports partial doubt, underestimates |
| Landscapes | **falls** (8.31) | rises a lot | prior finds nothing to latch onto, invents confidently |

Two ways calibration fails, not one.

**The assumed ordering was partly wrong — and the metrics already knew.** Level 3 scored worse than level 4 on both metrics (0.624 against 0.747; +0.194 against +0.143), which looked like an inversion under the assumed order. The measured distance independently places level 3 furthest from FFHQ. Paintings are not a mild shift: brushstroke texture, non-photographic colour and an unusual frequency profile put them further from the prior than any photograph of anything.

**Correction to the preliminary claim.** The proposal stated that pooled per-pixel conformal coverage fails to detect the degradation, based on a measured gap of 0.031. Phase A shows it reaches +0.161 on paintings and +0.171 on landscapes.

The preliminary non-face set was almost entirely single objects on plain backgrounds — level 4 here, which gives +0.046, close to the original figure. The claim was over-generalised from one kind of image. The narrower and better-supported version:

> Pooled per-pixel coverage is insensitive to prior mismatch when the shifted content has large low-gradient regions. On content-rich images it detects the shift. The gradient restriction recovers the signal precisely in the first case — a 3.1× improvement on single objects against 1.2× on landscapes — because it removes the flat regions that were diluting the estimate.

**The gradient restriction tracks severity, not just presence.** Top-10% coverage correlates with measured distance at −0.521, stronger than pooled at −0.427. It does not merely flag that shift has occurred; it tracks how much.

**Almost nothing is well calibrated.** Scarcely any image at any level sits above a ratio of 1.0, including FFHQ faces at 0.881. Consistent with Thong et al., who found diffusion posterior samplers overconfident even with an appropriate prior.

---

### Visual evidence

Inspecting the per-pixel disagreement across samples shows what the numbers do not.

**Level 0, faces.** Disagreement maps trace facial structure specifically — eyes, nostrils, lips, individual hair strands, the jaw outline. The sampler knows which parts of a face it is guessing at.

One image is instructive: an older woman's skin texture is smoothed away in every sample and she appears noticeably younger. The disagreement map is dark across the cheeks and forehead, exactly where the smoothing occurred. Even with a correct prior, the reported uncertainty does not cover the worst error.

**Level 3, painted portraits.** The three samples are three different people — different faces, ages and poses, one with a beard and hat appearing nowhere in the original. The sampler discards the painting and substitutes photographic portraits, a different one each run.

**Level 5, landscapes.** The 64×64 measurement clearly contains the coarse structure — rock formations, horizons, hillsides are all legible. The reconstructions discard every detail finer than that. Grass tussocks vanish, field boundaries disappear, rock faces become smooth beige blobs.

The disagreement maps are bright along structural boundaries — ridges, horizons, rock edges — and near zero across large smooth regions.

This resolves an apparent contradiction. Mean spread at level 5 is 8.31, *below* level 0's 9.81, despite far higher error. The sampler is not uniformly confident; it reports doubt at edges as it should. But a landscape is mostly flat region, which pulls the mean down, and it is precisely in those flat regions that real texture has been replaced with smooth gradients the sampler reports no uncertainty about.

It also explains why the gradient restriction helps less at level 5 (1.2×) than at level 4 (3.1×): at level 5 much of the failure lies *inside* the low-gradient regions, so restricting to edges does not capture it.

**The contrast.** In distribution, disagreement follows facial structure. Out of distribution, it follows generic edges while error is distributed throughout. The sampler retains a sense of *where* detail is uncertain but loses any sense of *what* that detail should be.

---

### Limitations specific to Phase A

1. **The distance uses hand-chosen image statistics** rather than a learned feature space. Chosen because it runs offline and every term is explicable, but it is not the standard approach and may miss semantic distance entirely.
2. **Datasets differ in more than content.** CelebA-HQ and FFHQ were both built from web photographs with face alignment; AFHQ was collected differently; paintings were digitised from museum scans. Capture conditions vary alongside subject matter.
3. **Reconstruction difficulty is not controlled.** A landscape contains more high-frequency content than a face and is harder to reconstruct regardless of which prior is used. The current design cannot separate prior mismatch from image difficulty.
4. **Ten images per level.** Level 4 has sd 0.158, the widest of any level, suggesting that group is not internally coherent.
5. **The ordering is still partly an argument.** Levels 0–2 were confirmed by measurement; 3–5 were not.
6. **Correlations are moderate.** r ≈ −0.5 means distance explains roughly a quarter of the variance across images. The trend is real; the prediction is weak.
7. **Level 3 is painted, not sculpted**, so it changes imaging medium as well as material.
8. **Visual inspection is qualitative.** The disagreement maps were examined for four images per level; the patterns described are consistent across those four but not quantified.

---

## Reproducing

`Diffusion_model sheet.ipynb` regenerates every number above from saved posterior samples. It requires no GPU.The notebook reads samples from Kaggle notebook outputs. To run it elsewhere, set BASE4, BASE16 and BASEG in the first cell to directories containing the corresponding sample folders.

### Data

Not redistributed here.

- **Prior checkpoint** — `ffhq_10m.pt` from [DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling)
- **In-distribution images** — FFHQ at 256×256 ([Karras et al., CVPR 2019](https://github.com/NVlabs/ffhq-dataset)). The first ten images by sorted filename.
- **Out-of-distribution images** — nine photographs of everyday objects and scenes. Provenance and licensing are undocumented, so these are not redistributed. See limitations.

### Sampling

Unmodified DPS via `sample_condition.py`. Only the task config is edited: data root, `scale_factor`, and conditioning `scale`. The loop is in the notebook appendix.

Two implementation notes:

- **Ground truth** must come from the pipeline's own `label` output, not the input file. DPS applies internal normalisation; comparing against an unnormalised source adds roughly 9.5 intensity units of spurious error.
- The repository pins PyTorch 1.11 but runs unmodified on 2.10 after stubbing the `motionblur` module, which is imported at the top of `measurements.py` but used only by the motion-deblurring task.

---

## Limitations

1. **Sample size.** 19 images at 6 samples each. Establishes direction, not magnitude. Distributions overlap.
2. **Metric interpretation.** The expected ratio for a calibrated sampler is not exactly 1.0 and depends on distributional shape. Claims are comparative, not absolute.
3. **Per-pixel coverage is the permissive formulation.** Image-level conformal prediction is substantially stricter.
4. **Coverage convergence unverified.** The convergence check was performed on the ratio only. Coverage from 6 samples underestimates interval width, so absolute figures are pessimistic; between-group comparison remains valid.
5. **Inverse crime.** The same forward operator generates the measurement and drives the reconstruction.
6. **Sampler approximation.** DPS approximates the likelihood at each step, so prior mismatch and likelihood approximation error are not fully separable here.
7. **Single sampler, prior, and task.** Generalisation to DDRM, ΠGDM, or other priors is untested.
8. **Compute.** Roughly 25–75 GPU hours at 256×256, against ~1,100 hours at 96×96 in comparable published work.
9. **Provenance.** The non-face set was collected ad hoc without documented licensing.
10. **Run-to-run variation.** The downscale-16 configuration was run twice with identical settings and different seeds; the in-distribution ratio differed by 0.019 on ten images. Figures here are from the second run.

---

## Key references

- Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems*, ICLR 2023
- Blau & Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018
- Thong, Mbakam & Pereyra, *Do Bayesian imaging methods report trustworthy probabilities?*, arXiv:2405.08179
- Burns & Fridovich-Keil, *When, why, and how do diffusion posterior samplers fail?*, arXiv:2605.30330
- Kheirandish, Hong & Fridovich-Keil, *KLIP*, CVPR 2026, arXiv:2605.31596
- Ekmekci & Cetin, *Conformalized Generative Bayesian Imaging*, arXiv:2504.07696
- Everink, Tamo Amougou & Pereyra, *Self-supervised Conformal Prediction for Imaging*, arXiv:2502.05127

Work in progress. Thesis submission February 2027.
