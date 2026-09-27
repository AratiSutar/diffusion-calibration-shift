# Calibration Behaviour of Diffusion Models Under Distribution Shift

Code and results for a master's thesis investigating whether diffusion posterior samplers report honest uncertainty when their learned prior does not match the input.

**Arati Dipak Sutar** · MSc Data Science, University of Europe for Applied Sciences
Supervisors: Prof. Dr. Iftikhar Ahmed, Prof. Shan Faiz

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
