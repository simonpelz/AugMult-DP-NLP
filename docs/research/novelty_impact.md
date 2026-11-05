# Novelty & impact assessment of continuation directions

*Research-only assessment, 2026-11-05. Inputs: the docs/CONTINUATION.md §4 brainstorm, a citation sweep and targeted searches.*

**User constraints:**
* No PhysioNet/MIMIC access; no cluster.
* A local GPU plus a modest cloud budget.
* Interested in small LLMs and LLM-generated augmentations.
* Explicit caveat: LLM-generated views mean the student partly learns *the generator's* distribution, i.e. distillation.

---

## 1. Prior-work check

**Method.**
* I pulled the papers that cite De et al. 2022 ([arXiv 2204.13650](https://arxiv.org/abs/2204.13650)) and Sander et al. 2023, TAN ([arXiv 2210.03403](https://arxiv.org/abs/2210.03403)), via the Semantic Scholar Graph API: **313 + 66 citing papers, 334 unique**.
* I regex-filtered the titles and abstracts for text/LM terms and augmentation terms, then reviewed by hand the ~100 text-related titles.
* OpenAlex lists only 85 citers of De et al. and was rate-limited, so Semantic Scholar was the broader source.
* Targeted web searches covered each direction below.

**Result.**
* **No citing paper applies augmentation multiplicity (multiple views averaged before clipping) to text or language models.**
* The augmentation-related citing papers are all vision or diffusion papers:
  * DP-Mix ([2311.01295](https://arxiv.org/abs/2311.01295))
  * Augment-then-Smooth ([2306.08656](https://arxiv.org/abs/2306.08656))
  * DP diffusion ([2210.09929](https://arxiv.org/abs/2210.09929))
  * Patch-level DP via random cropping ([2603.24695](https://arxiv.org/abs/2603.24695))
* *Caveats:* Semantic Scholar's coverage is incomplete and many abstracts were missing. Treat this as "no evidence found", not proof of absence.

**Closest prior work, by theme:**

| Theme | Closest work | How it differs from our idea |
|---|---|---|
| LLM-generated augmentation under DP | Song et al. 2024, *LLM-based Privacy Data Augmentation Guided by KD with a Distribution Tutor* ([2402.16515](https://arxiv.org/abs/2402.16515)) | PATE-style DP filtering of *released* synthetic data. Not per-example views inside DP-SGD. |
| Distillation with a frozen public teacher under DP-SGD | **DP-OPD** (Khadem et al., Apr 2026, [2604.04461](https://arxiv.org/abs/2604.04461)) | DP-SGD only on the student, with a frozen non-private teacher and token-level KD. Generation (perplexity), on-policy samples, no multiplicity. |
| DP distillation via synthetic text | DistilDP ([2403.00932](https://arxiv.org/abs/2403.00932)); DPRefine / self-distillation, NAACL 2025 ([2410.17566](https://arxiv.org/abs/2410.17566)); *How to Train Private Clinical LMs* ([2511.14936](https://arxiv.org/abs/2511.14936)) | Teacher trained privately, or released synthetic data. Not per-example multi-view objectives. |
| Public data + DP zeroth-order | PAZO, NeurIPS 2025 ([2511.10859](https://arxiv.org/abs/2511.10859)) | Mixes public first-order gradients into DP-ZO. No augmentation views. |
| User-level DP for LMs (same "average then clip" path) | Chua et al. 2024 ([2406.14322](https://arxiv.org/abs/2406.14322)); Charles et al., SaTML 2025 ([2407.07737](https://arxiv.org/abs/2407.07737)) | Average over a user's records. No within-record views or two-level averaging. |
| Calibration under DP | Bu et al. ([2106.07830](https://arxiv.org/abs/2106.07830)); Zhang et al. 2022, *A Closer Look at the Calibration of DP Learners* ([2210.08248](https://arxiv.org/abs/2210.08248)): DP temperature scaling | I found no 2024–26 work on augmentation or label smoothing as a DP calibration fix. Label smoothing for LLM calibration is non-private only ([2508.00264](https://arxiv.org/abs/2508.00264)). |
| Ghost clipping for transformers | Li et al. 2022 ([2110.05679](https://arxiv.org/abs/2110.05679)); Opacus ghost mode; DPFormer phantom clipping ([2305.17633](https://arxiv.org/abs/2305.17633)) | None handles multiple views per example. |
| Correlated noise | BandMF ([2306.08153](https://arxiv.org/abs/2306.08153)); BLT ([2408.08868](https://arxiv.org/abs/2408.08868)) | Experiments are on CIFAR. I found no source on its interaction with AugMult, and nothing on text. |
| Auditing | One-run audit (NeurIPS 2023); metagradient canaries, ICLR 2026 ([2507.15836](https://arxiv.org/abs/2507.15836)) | No study isolates augmentation or multiplicity as a variable. |
| DP-LoRA on small decoders | Yu et al. 2022 ([2110.06500](https://arxiv.org/abs/2110.06500)); *Revisiting privacy/utility/efficiency…* ([2502.13313](https://arxiv.org/abs/2502.13313)); *Re-examining LoRA for private fine-tuning* ([2510.01137](https://arxiv.org/abs/2510.01137)) | Mostly RoBERTa, GPT-2 or Llama-2. Little on Qwen3/Gemma 3; nothing with AugMult. |

**The distillation caveat is well grounded in the literature**, and the design must control for it:
* Function matching (Beyer et al., CVPR 2022, [2106.05237](https://arxiv.org/abs/2106.05237)). KD works best when teacher and student see the *same* aggressive augmented views. "LLM views" are a weak, implicit form of this.
* *Does Knowledge Distillation Really Work?* (Stanton et al. 2021, [2106.05945](https://arxiv.org/abs/2106.05945)).
* Subliminal learning (Cloud et al. 2025, [2507.14805](https://arxiv.org/abs/2507.14805)). Teacher-generated data transmits hidden teacher traits, but **only when student and teacher share an initialization**.
  * *Design rule:* use generator and student from different model families, or measure transfer explicitly.
* Model collapse (Shumailov et al., Nature 2024): diversity shrinks when training on generated data.
* *Does training with synthetic data truly protect privacy?* ([2502.12976](https://arxiv.org/abs/2502.12976)).
* Mireshghallah et al., DP model compression ([2206.01838](https://arxiv.org/abs/2206.01838)). A teacher fit on private data must itself be DP. Ours is *public*, so no extra cost.

**Privacy rule for every LLM-view design:**
* Views and teacher outputs computed *per example, inside the clip*, from public models, are free under DP.
* Anything that pools across examples is not free. That includes choosing a filter threshold from the private-data distribution, or fine-tuning the generator on private data.
* Stored views are as sensitive as the raw data and must never be released.

---

## 2. Scored directions

Scale is 1–5; higher is better. For Risk, 5 means low risk; for Speed, 5 means fastest to a first result. Feasibility is judged on the user's compute: one local GPU plus some cloud, with no MIMIC.

| # | Direction | Novelty | Impact | Feasibility | Risk | Speed | **Σ** | One-line rationale |
|---|---|---|---|---|---|---|---|---|
| b | AugMult as a **calibration fix** / rising-loss mechanism | 4 | 3 | 5 | 4 | 5 | **21** | Builds on the thesis's strongest finding; cheap to test with label smoothing, ECE and temperature scaling; nothing found 2024–26. Audience: DP and calibration researchers. |
| e | **Ghost-clipping AugMult** (concatenate K views along the sequence axis) | 4 | 3 | 4 | 4 | 4 | **19** | The math is exact for linear layers; it makes LoRA and full fine-tuning with AugMult affordable. Impact comes mainly through tooling (an Opacus PR) and as an enabler for the other directions. |
| a | **AugMult for DP text revisited properly** (regimes × compute-matched × data size) | 3 | 3 | 5 | 3 | 4 | **18** | First real test of AugMult in text. The risk is a null result, which is still informative given the vision hype. |
| d | **Per-example multi-view objectives** (KL / function matching to a *public* teacher on views, consistency) | 4 | 4 | 4 | 3 | 3 | **18** | Turns the distillation caveat into the method. DP-OPD (2026) is adjacent (generation, on-policy) but not multi-view classification. |
| f | **AugMult for DP zeroth-order fine-tuning** of small LLMs | 4 | 3 | 5 | 2 | 4 | **18** | Trivial to implement (average K scalar losses) with inference-only memory. Risk: the variance of the random ZO direction may swamp view variance, giving tiny effects. |
| c | **LLM-generated views + label-preservation filter** | 3 | 4 | 4 | 3 | 3 | **17** | High interest, but on its own it is confounded with distillation. Only meaningful combined with (a)/(d) controls. |
| j | **DP-LoRA on small decoders** (Qwen3-0.6/1.7B, Gemma-3 270M/1B, VaultGemma-1B) ± AugMult | 2 | 4 | 3 | 3 | 3 | **15** | DP-LoRA itself is well studied. It is valuable as a *regime* inside (a), not as a standalone contribution. |
| g | **User-level DP sharing the AugMult code path** (two-level averaging) | 3 | 3 | 3 | 3 | 2 | **14** | User-level DP already exists (Charles et al., SaTML 2025); "within-record views" is a modest add-on and needs user-partitioned public data (Reddit, StackOverflow, Amazon). |
| i | **Empirical auditing ± AugMult** (one-run canaries, LiRA, augmentation-aware MIAs) | 3 | 3 | 3 | 2 | 2 | **13** | Novel as a variable, but needs many runs and canaries. Theory says ε is unchanged, so findings may be subtle. |
| h | **AugMult + correlated noise** (BandMF/BLT) | 2 | 2 | 2 | 3 | 2 | **11** | An orthogonal "stacking" result. Needs a JAX port; low novelty per unit of effort. |

---

## 3. Recommendation

### Primary: "Variance or distillation? Augmentation multiplicity for differentially private text classification"

This merges a + c + d, uses j as one regime, relies on e as the enabler, and reports b as an analysis axis.

**Research question.** When K views per example help DP fine-tuning of text models, how much comes from:
* **(i) variance reduction** before clipping, which is AugMult proper;
* **(ii) knowledge transferred from the public model** that generated the views or scores them, which is distillation;
* **(iii) the regularizing effect of label-noisy views**, which shows up as calibration?

Can we make (ii) explicit and better with per-example function matching?

**Falsifiable hypotheses.** All are judged at matched ε, δ=1/n^1.1, and compute (number of forward passes), with ≥10 paired seeds.
* **H1:**
  * Teacher-free views (dropout masks, token cutoff; no external model) improve over the compute-matched K=1 baseline by **≥ 0.5 pt** only when n ≤ 5k, or when the regime trains ≥ LoRA-scale parameters.
  * Otherwise the effect is within noise. This replicates the thesis's ~+1 pt in the regime where it should exist.
* **H2:**
  * LLM-paraphrase views beat teacher-free views.
  * The gap shrinks to ≤ ⅓ when the *same public model* is used as an explicit teacher via the per-example KL term.
  * That would mean the "LLM-view gain" is mostly distillation.
* **H3:** At small C, AugMult reduces ECE and NLL by a margin comparable to label smoothing tuned to the same accuracy. In other words, the rising-loss fix is a calibration effect.

**Minimal first experiment.** Public data only; one 24 GB GPU; LLM views generated once on a rented A100.

* **Data:**
  * SST-2 and QNLI, for comparability with Li et al. 2022 and Yu et al. 2022.
  * One domain task where a public LLM is weak zero-shot. HoC or ChemProt (BLURB), or a legal task (LEDGAR/ContractNLI). Check licenses; NLI4CT is a possible MedNLI stand-in.
  * Subsample n ∈ {1k, 5k, full}.
* **Students:**
  * RoBERTa-base or ModernBERT-base, in head-only (cached features) and LoRA (r=8) regimes.
  * Then Qwen3-0.6B or Gemma-3-1B with LoRA as a decoder regime.
* **Views:**
  * dropout-K (free);
  * back-translation (NLLB);
  * LLM paraphrases from a *different-family* local instruct model (~8B, via vLLM), with and without a public-NLI label-preservation filter at a fixed threshold.
  * Rough cost estimate (unverified): 5k examples × 16 views ≈ 3–5M generated tokens, under an hour on one A100.
* **Arms:**
  1. DP K=1;
  2. DP K=1 with K× batch (compute-matched);
  3. AugMult with each view source;
  4. AugMult + per-example KL to the public teacher on the same views;
  5. label smoothing;
  6. reference lines for the **teacher zero-shot** and the non-DP ceiling.
* **Privacy:** ε ∈ {1, 8}, PRV accountant, Poisson sampling.
* **Metrics:** test accuracy/F1, ECE/NLL, per-example gradient-norm statistics (fraction clipped, norm concentration), and the view label-flip rate (the thesis's "inherent error").

**Success criteria:** H1 holds on at least one task and regime, *and* either H2 or H3 holds. Together that is a clean mechanistic story with an actionable recipe.

**Kill / pivot criteria:**
* No view source beats the compute-matched K=1 baseline by more than 0.5 pt anywhere with n ≤ 5k: publish the negative result plus calibration (Backup B).
* The teacher's zero-shot accuracy beats every DP student on all tasks: reframe as "when is DP fine-tuning worth it versus a public model", or move to tasks with idiosyncratic labels.

**Headline figure.**
* x-axis: training-set size n (log scale). y-axis: gain over the compute-matched DP K=1 baseline.
* One line per view source: dropout → back-translation → LLM paraphrase → LLM + filter → LLM + per-example KL.
* Shaded band for the teacher's zero-shot accuracy.
* Panels by regime (head-only / LoRA / decoder-LoRA).
* Inset: ECE vs clipping norm C, ± AugMult, ± label smoothing.

**How the thesis results slot in:**
* The head-only regime is the thesis setting. MedNLI ε=1 (BT +~1 pt, EDA −1 pt) becomes the motivating pilot.
* The "inherent error" plot becomes the view label-flip metric.
* The K-scaling plot becomes the K-axis ablation.
* The rising-loss plot motivates H3.

### Backup B (also the fast first paper): "Augmentation multiplicity fixes the miscalibration of small-clip DP-SGD"

* **Why first:** it is the cheapest and most distinctive result, and it reuses the head-only cached-feature setup.
* **Data:** SST-2, QNLI and HoC/ChemProt in place of MedNLI.
* **Comparison:**
  * AugMult vs label smoothing vs DP temperature scaling (Zhang et al. 2022) vs a larger C;
  * across C ∈ {1e-4 … 1} and ε ∈ {1, 8}.
* **Hypothesis:** H3 above.
* **Kill criterion:** if rising loss does not reproduce on public data, or label smoothing fully explains it, keep it as a section of the primary paper rather than a standalone.
* **Headline figure:** a calibration curve and validation NLL over steps, coloured by C, with three panels: baseline / AugMult / label smoothing.

**Cheap side probe (direction f):** add K-view averaging to an existing DP-ZO implementation on Qwen3-0.6B / SST-2. It takes about 1–2 days. Report it only if the effect is more than seed noise.

---

## 4. Venues and timeline (as of 2026-11-05; verify dates)

| Venue | Deadline | Fit |
|---|---|---|
| ACL 2027 (ARR January cycle) | **2027-01-04** ([official call](https://2027.aclweb.org/calls/main/)) | Tight. Only realistic for Backup B as a short paper if Phase 0 finishes in November. |
| ICML 2027 | ~late Jan 2027 (estimate; [CFP not yet posted](https://icml.cc)) | Primary only if H2 yields a method; unlikely in time. |
| TPDP 2027 | 2026 edition was ~Feb 18 (abstract, non-archival) ([2026 CFP](https://differentialprivacy.org/tpdp2026/)); 2027 not yet announced | Good for Backup B or early primary results. Non-archival, so it does not block later publication. |
| PrivateNLP 2027 | 2026 edition was March (colocated with ACL) ([2026 site](https://sites.google.com/view/privatenlp2026/)); 2027 not yet announced | Workshop paper for either direction; non-archival option available. |
| EMNLP 2027 (ARR) | ~May 2027 (estimate; [trackers](https://trybibby.com/conference-deadlines/emnlp-2027)) | **Main target for the primary paper.** |
| NeurIPS 2027 | ~May 2027 (estimate) | Alternative if the method (per-example KD) is strong. |
| SaTML 2028 | ~late Sep 2027 (the SaTML 2027 deadline of ~2026-09-30 [has passed](https://mldeadlines.com/conference/satml/)) | Fallback venue for a privacy-centric framing (auditing extension). |

**Suggested schedule:**
* Phase 0 (revive, tests, public datasets): Nov 2026.
* Backup B experiments: Dec 2026; submit to TPDP or PrivateNLP 2027.
* Primary paper experiments: Jan–Apr 2027; submit to EMNLP 2027 in ~May.

---

## 5. References not linked above

* Hinton, Vinyals, Dean 2015, *Distilling the Knowledge in a Neural Network* ([1503.02531](https://arxiv.org/abs/1503.02531)).
* Shumailov et al. 2024, *AI models collapse when trained on recursively generated data*, Nature.
* Cegin et al., NAACL 2025, *LLMs vs Established Text Augmentation Techniques for Classification* ([ACL Anthology](https://preview.aclanthology.org/setup/2025.naacl-long.526)).
* Term-wise DP-SGD (Takahashi et al., [2006.11204](https://arxiv.org/abs/2006.11204)): sensitivity when adding divergence terms. This is relevant for the per-example KL objective, although per-example clipping of the total loss's gradient already bounds sensitivity.
* McKenna et al., ICML 2025, *Scaling Laws for DP LMs* ([2501.18914](https://arxiv.org/abs/2501.18914)): the noise-batch ratio as the lens for compute-matching.
