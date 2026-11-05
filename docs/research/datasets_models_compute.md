# Datasets, models and compute: feasibility without PhysioNet or a cluster

*Research notes, Nov 2026. Constraints: **no PhysioNet credentials** (MedNLI and MIMIC are unavailable) and **no cluster**. Compute is a local machine (GPU unknown) plus rented cloud GPUs.*
*How to read the marks: ✅ means confirmed from a primary or near-primary source during this research. ⚠️ means it comes from secondary sources or memory and must be checked on the dataset or model card before use.*

---

## 1. Datasets

### 1.1 Clinical and biomedical, no credentialing needed

| Dataset | Task | Size | Licence / access | HF id | Fit for this project |
|---|---|---|---|---|---|
| **NLI4CT / NLI4CT-P** (SemEval-2023/24 Task 2) | Clinical-trial-report NLI (entail/contradict), single or comparison CTRs, breast cancer | ~1.7k train / 200 dev / ~2.1k–5.5k test (counts differ by mirror) ✅ | Licence not declared on BigBio (`UNKNOWN`) ⚠️; data from clinicaltrials.gov (public) | `bigbio/sem_eval_2024_task_2` ✅ | **Best MedNLI stand-in**: clinical NLI and *small*, so the data-scarce regime where AugMult should matter. Premises are long CTR sections; truncation at 128 tokens is not enough, so 512+ or section selection is needed. |
| **BioNLI** | Biomedical NLI (mechanistic premises from abstracts; adversarial negatives) | Large and balanced versions, shared test set | CC BY 4.0 ✅; test set gated or on request ✅ | `clinicalnlplab/BioNLI_test` (gated) | Good second NLI task; negatives are synthetic. |
| **SciNLI** | Scholarly NLI (NLP papers) | 107k pairs (101k auto-labelled train) ✅ | CC BY-SA 4.0 ✅ | GitHub `msadat3/SciNLI` | Larger, not clinical; useful for the data-size curve. |
| **PubMedQA (pqa_labeled)** | Yes/no/maybe QA over abstracts | 1k expert-labelled | MIT ⚠️ | `qiaojin/PubMedQA` ⚠️ | Tiny and hard; a classification view of QA. |
| **HoC** (Hallmarks of Cancer) | Multi-label sentence classification (10 labels) | ~1.8k abstracts | ⚠️ verify (BigBio) | `bigbio/hallmarks_of_cancer` ⚠️ | Already half-supported in the repo (`LocalHoCDataset`); multi-label needs a fix. |
| **ChemProt** (BLURB) | Relation classification | ~18k | ⚠️ varies by BLURB task | `bigbio/chemprot` ⚠️ | Optional. |
| **MedMCQA** | Medical exam MCQ | 194k | Apache-2.0 on HF vs CC BY-NC-SA elsewhere (conflict) ⚠️ | `openlifescienceai/medmcqa` ✅ | Multiple-choice, not a clean classification fit. Lower priority. |
| **Asclepius synthetic clinical notes** | 157k synthetic discharge summaries (GPT-3.5 from PMC-Patients case reports) plus instruction/QA | 157k | **CC BY-NC-SA 4.0** ✅ (research use OK) | `starmpcc/Asclepius-Synthetic-Clinical-Notes` ✅ | **"Privacy-plausible" clinical text with zero credentialing.** You could derive a classification task (e.g. note-type or QA-task-type labels). Caveat: it is synthetic GPT-3.5 text, so distribution artefacts are possible. |

**What makes a stand-in "privacy-plausible":** real clinical *style* (abbreviations, notes) but publicly releasable. NLI4CT (public trial reports) and Asclepius (synthetic notes) are the closest honest stand-ins for MedNLI. Neither contains real patient records, so *frame the experiments as "clinical-domain DP-NLP", not "patient data"*.

### 1.2 Privacy-motivated public sets from DP-NLP papers

| Dataset | Used by | Notes |
|---|---|---|
| **SST-2, QNLI, MNLI, QQP** (GLUE) | Li et al. 2022, Yu et al. 2022, and most DP fine-tuning work | The **standard comparability set**. MNLI is the natural general-domain analogue of MedNLI. HF: `nyu-mll/glue` ✅ |
| **Yelp** (review polarity / full; Yelp Open Dataset) | Yue et al. 2023, Aug-PE | HF `yelp_polarity`, `yelp_review_full`. The Open Dataset has **user_id**, which suits user-level DP; academic-use licence ⚠️ |
| **PubMed abstracts (post-2023)** | Yue et al. 2023, Aug-PE | Collected after model cutoffs to limit pretraining contamination; available in the Aug-PE repo (`AI-secure/aug-pe`) ⚠️ |
| **OpenReview (ICLR 2023)** | Aug-PE | Review text with ratings and areas as labels. |
| **TAB / ECHR** | Text anonymisation | 1,268 ECHR cases annotated for personal information; licence of the corpus itself is unclear (MIT on a derivative HF card; the paper is CC BY-NC-ND) ⚠️ |
| **Enron emails** | Memorisation and extraction studies | Grouped by sender, so usable for user-level DP. |
| **IMDb, AG News** | Common DP classification baselines | Easy; good for sanity checks and CPU-scale runs. |
| **StackOverflow (TFF), CC-News** | Charles et al. 2024 (user-level DP for LLMs) ✅ | **Standard user-grouped datasets.** Use these for user-level DP experiments (roadmap idea 17). |
| **Amazon Reviews 2023** (McAuley lab) | Recsys / NLP | Has user ids, so also user-grouped ⚠️ (check licence) |

### 1.3 Recommendation

* **Primary (core study, Phase 2):**
  * **MNLI** and **SST-2** (comparability with Li et al. 2022). MNLI replaces MedNLI's role as "NLI".
  * **NLI4CT** as the clinical, data-scarce NLI task.
  * **Subsampled MNLI/SST-2** (500 / 1k / 2k / 5k / full) for the data-scarcity curve, which tests the thesis's own "AugMult helps when data are scarce" claim.
* **Secondary:**
  * **QNLI** and **QQP**, for completing the Li et al. table.
  * **Asclepius**-derived classification, for clinical *style* at scale.
  * **Yelp / PubMed (Aug-PE)**, for a head-to-head against DP synthetic data.
  * **StackOverflow / Yelp-by-user**, for user-level DP + AugMult.

---

## 2. Models

### 2.1 Encoders (DP fine-tuning backbones for classification)

| Model | Sizes | Licence | Released | Notes |
|---|---|---|---|---|
| BERT-base-uncased | 110M | Apache-2.0 | 2018 | The thesis baseline; keep it for continuity only. |
| RoBERTa-base/large | 125M / 355M | MIT | 2019 | **The Li et al. 2022 DP backbone, for comparability.** |
| DeBERTa-v3-base/large | 184M / 435M | MIT | 2021 | Strong on NLI. |
| **ModernBERT-base/large** | 149M / 395M | Apache-2.0 | Dec 2024 | Fast (unpadding, FlashAttention), 8k context. Opacus hook coverage needs checking (RMSNorm/GLU; Opacus 1.5.4 added an RMSNorm hook) ⚠️ |
| **Ettin encoders** (`jhu-clsp/ettin-encoder-{17m,32m,68m,150m,400m,1b}`) | 17M–1B | MIT ✅ | Jul 2025 (ICLR 2026) ✅ | **Fully open training data** (good for contamination control) and paired decoders with the identical recipe. **The 17M/32M sizes make CPU-only DP experiments feasible.** Beats ModernBERT at matched size per the authors ✅ |
| NeoBERT | 250M | MIT ⚠️ | 2025 | Optional. |
| BioClinical ModernBERT | 150M / 396M | ⚠️ | Jun 2025 | **Pretrained on MIMIC-IV notes** (2.6M notes, about 1.77B tokens) ✅. Avoid it as a backbone if MIMIC-derived eval data is ever used (Tramèr et al. overlap). On NLI4CT/Asclepius the overlap is less of an issue, but disclose it. |
| Clinical ModernBERT (Lee et al.) | — | ⚠️ | Apr 2025 | Also trained on PubMed + **MIMIC-IV** ✅ |

### 2.2 Small decoders (DP-LoRA backbones, local view generators)

| Model | Sizes | Licence | Released | Notes |
|---|---|---|---|---|
| **Qwen3.5 Small** | 0.8B / 2B / 4B / 9B | Apache-2.0 (secondary sources) ⚠️ | Feb–Mar 2026 ⚠️ | Latest small Qwen line, natively multimodal. Strong paraphraser at 4B. |
| Qwen3 | 0.6B / 1.7B / 4B | Apache-2.0 ⚠️ | Apr 2025 | Mature tooling. |
| **Gemma 4 E2B / E4B** | ~2B / ~4B effective | **Apache-2.0** (first Gemma under Apache) ✅ secondary | Apr 2026 ✅ | Small Gemma 4 sizes, 128k context. |
| Gemma 3 | 270M / 1B / 4B | Gemma Terms of Use, gated ✅ | 270M: Aug 2025 ✅ | 270M is a cheap DP-LoRA testbed. |
| **VaultGemma 1B** | 1B | Gemma terms, gated ⚠️ | Sep/Oct 2025 ✅ | **DP-pretrained from scratch** (sequence-level ε≤2, δ≤1.1e-10) ✅. Gives a clean "no private data in pretraining" backbone story. 1024-token context ✅. HF: `google/vaultgemma-1b` (in transformers) ✅ |
| Llama 3.2 | 1B / 3B | Llama community licence, gated | Sep 2024 | Widely used baseline. |
| SmolLM2 / **SmolLM3** | 135M–1.7B / 3B | Apache-2.0 (SmolLM3 via secondary sources) ⚠️ | 2024 / Jul 2025 | Fully open data recipe. |
| Phi-4-mini | 3.8B | MIT ⚠️ | Feb 2025 | Good instruction follower (paraphrasing). |
| OLMo 2 1B / **OLMo 3** 7B, 32B | 1B / 7B / 32B | Apache-2.0 | 2025 / Dec 2025 ✅ | **Fully open data (Dolma 3) plus checkpoints** ✅, which allows checking whether eval or view text was in pretraining. OLMo 3 has no 1B size ✅ |
| Pythia | 70M–12B | Apache-2.0 | 2023 | Open data (Pile) and checkpoints; the classic memorisation-study model. |
| Ettin decoders | 17M–1B | MIT ✅ | Jul 2025 | Paired with the encoders, so the **encoder vs decoder comparison is controlled under DP.** |

### 2.3 Role assignment

| Role | Recommended | Why |
|---|---|---|
| **(a) DP fine-tuning backbone (classification)** | RoBERTa-base/large (comparability); **Ettin-encoder 150M/400M** (open data, modern); DeBERTa-v3-base (NLI strength); Ettin-17M/32M for CPU pilots | Covers comparability, modernity and cheap pilots. |
| **(a′) DP-LoRA decoder backbone** | **VaultGemma-1B** (DP-pretrained), Qwen3/3.5 0.6–2B, Gemma 3 270M/1B, Ettin-decoder 400M/1B (paired control) | Small enough for one consumer GPU. |
| **(b) Local paraphrase / view generator** | Qwen3.5-4B or Gemma 4 E4B instruct (quality); Qwen3-1.7B (fast); **OLMo-3-7B-Instruct** when you need to audit the generator's training data | Runs under vLLM on 16–24 GB. ⚠️ Views carry the generator's distribution (see the distillation notes in another worker's report). |
| **(c) Public teacher / label-preservation filter** | NLI cross-encoders: `cross-encoder/nli-deberta-v3-base`, `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` ⚠️ (check ids); for clinical text, an NLI model fine-tuned on public biomedical NLI (BioNLI/SciNLI) | Scores whether a view keeps the label or entailment. Uses only the example plus a public model, so it is DP-free. **Caution:** a teacher trained on MNLI must not be used to filter MNLI experiments without disclosure; it leaks the task signal. |

### 2.4 "Jav": interpretation

* No notable 2025–26 small LLM family is named "Jav…". The only hits were a Javanese-translation thesis and Infosys **NT-Java-1.1B** (a Java *code* model, irrelevant here).
* Most likely this is a speech-to-text rendering of **"Gemma"** (Gemma 3 270M/1B, **Gemma 4 E2B/E4B**, **VaultGemma**). This fits best because VaultGemma is the DP-relevant one.
* Less likely candidates:
  * **"Jamba"**: AI21's hybrid SSM-Transformer. I believe a small *Jamba Reasoning 3B* exists (late 2025) ⚠️ unverified. It is a poor fit for Opacus (Mamba layers lack per-sample grad hooks).
  * **"Jais"**: an Arabic LLM.

---

## 3. Compute estimates

### 3.1 Assumptions

* **Per-token FLOPs:** about 6·P per training token, and about 2·P per inference token. With LoRA, backward still flows through activations, so assume about 4·P–6·P.
* **Sustained throughput:**
  * RTX 4090 / L4-class with bf16 at realistic utilisation: about 25–40 TFLOP/s on a 4090, about 15–25 on an L4, about 100+ on an A100/H100.
* **DP overhead vs non-private:**
  * About 1.5–2× time for ghost clipping, which costs one extra backward pass (Li et al. 2022 ✅).
  * Others report 2–4× slowdown and up to 5× memory for older per-sample codebases ✅ (second-hand).
* **Ghost clipping and LoRA:** ghost clipping helps little with LoRA, because the effective layer width is small ✅ (Opacus mixed-precision blog). For LoRA, plain per-sample grads (hooks or functorch) are fine.
* **AugMult** multiplies the forward and backward work by K, and activation memory by K per physical example. Lower the physical batch by K (as the repo already does).
* **Sequence length:** pad dynamically (the repo pads everything to 128).
  * Assume SST-2 ≈ 32 tokens, MNLI ≈ 64, NLI4CT ≈ 384–512 (long trial sections).

### 3.2 Scenarios (single RTX 4090 / L4 unless noted)

| Scenario | K=1 | K=4 | K=8 | K=16 | Memory |
|---|---|---|---|---|---|
| **Head-only with cached encoder features** (thesis regime): one-off encoder forward per view, then a tiny MLP | Caching: SST-2 67k×32 tok ≈ 2e15 FLOPs, **< 1 min GPU**; **about 10–30 min CPU** for BERT-base; ×K for views | — | — | Caching SST-2 ×16 ≈ 10 min GPU / 3–6 h CPU | < 4 GB |
| Training on cached features | seconds–minutes **on CPU** per run | | | | **Lets you run hundreds of seeds on a laptop** |
| **Base encoder (110–150M) full FT, DP ghost clipping, SST-2** (67k, about 3 epochs) | ≈ 6·1.2e8·2.1e6 tok ≈ 1.5e15 per epoch → **≈ 1–2 min per epoch** (incl. DP overhead) | ≈ 5–8 min/ep | ≈ 10–15 min/ep | ≈ 20–30 min/ep → **≈ 1–1.5 h per run** | 6–12 GB at physical batch 64/K |
| Base encoder, ~10k NLI (MNLI subset / NLI4CT at 512 tok) | MNLI-10k: < 1 min/ep; NLI4CT (1.7k×512): < 1 min/ep | | | ≈ 10–15 min/ep | 8–16 GB at 512 tok (use smaller physical batch) |
| **0.5–1.7B decoder, DP-LoRA (r=8–16)**, 10k examples × 128 tok | ≈ 5·1e9·1.3e6 ≈ 6.5e15 → **≈ 4–8 min/ep (1B)**, ×1.7 for 1.7B | ≈ 20–30 min/ep | ≈ 45–60 min/ep | ≈ 1.5–2 h/ep | 1B bf16 weights 2 GB plus activations; **fits in 24 GB with gradient checkpointing**; 12 GB fits 0.5–1B at small physical batch |
| Same on CPU | Impractical (hours per epoch even at K=1) except Ettin-17M/32M or Gemma-270M pilots | | | | |

**View generation** (vLLM, 2–4B instruct generator, ~40 output plus ~120 prompt tokens per view):

* SST-2: 67k × 16 views ≈ 1.1M generations ≈ 45M output tokens.
* MNLI-10k pairs × 16 × 2 sentences ≈ 13M output tokens.
* NLI4CT: 1.7k × 16 × 2 ≈ 2M tokens (but long premises; cache them and paraphrase only the statement or selected sentences).
* Assume aggregate offline throughput (high batch) of **≈ 3–8k tok/s on an L4/4090** and **≈ 10–25k tok/s on an A100/H100** for 2–4B models ⚠️. This is an assumption, so benchmark with vLLM's own scripts. Batch size dominates throughput ✅.
* Estimates:
  * SST-2 ×16: **≈ 2–4 h on a 4090** or **≈ 0.5–1.5 h on an A100**.
  * MNLI-10k ×16: **< 1 h on a 4090**.
  * The NLI filter pass (DeBERTa-large cross-encoder over all views) is small next to generation: minutes to under an hour.

### 3.3 Cloud GPU prices (snapshot, Nov 2026; check live)

| GPU | Marketplace (Vast / RunPod Community) | Managed (RunPod Secure / Lambda) | Hyperscaler |
|---|---|---|---|
| RTX 4090 24 GB | $0.14–0.35/h | ≈ $0.74/h (RunPod Secure) | — |
| L4 24 GB | from $0.13/h (Vast spot) | $0.49/h (RunPod Secure), $0.80/h (Northflank) | ⚠️ GCP g2: check live |
| A100 80 GB | from $0.68/h (Vast spot) | ≈ $1.39/h (RunPod), ≈ $2.79/h (Lambda) | GCP ≈ $3.67–5.03/h |
| H100 | ≈ $1.49–2.40/h (Vast) | $2.89–3.49/h (RunPod), $3.29–3.99/h (Lambda) | Azure up to ≈ $12/h |

Sources: [gpuperhour.com (live, Nov 2026)](https://gpuperhour.com/), [Spheron 2026 comparison](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/), [Northflank cheapest providers](https://northflank.com/blog/cheapest-cloud-gpu-providers), [Tech Insider RunPod vs Lambda vs Vast](https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/), [CloudZero comparison](https://www.cloudzero.com/blog/cloud-gpu-pricing-comparison/). Marketplace hosts have no SLA. Checkpoint often; the repo's requeue logic already assumes preemption.

### 3.4 Budget for a "Phase 2 core study"

Assumed mix:
* About 300 head-only runs on cached features (CPU).
* About 200 base-encoder runs, LoRA or full, on MNLI-10k / NLI4CT / SST-2 subsets, mixed K (average K≈6).
* About 40 full SST-2 runs at K up to 16.
* About 30 DP-LoRA 1B decoder runs.
* View generation for SST-2, MNLI-10k and NLI4CT.

| Item | GPU-hours (4090/L4) | Cost @ $0.35–0.75/h |
|---|---|---|
| Feature caching (all datasets × 16 views) | 2–5 | $1–4 |
| 300 head-only runs | 0 (CPU) | $0 |
| 200 base-encoder runs (≈ 0.3–0.7 h each) | 60–140 | $20–105 |
| 40 full SST-2 runs (≈ 1–3 h each) | 40–120 | $15–90 |
| 30 DP-LoRA 1B runs (≈ 1–4 h each) | 30–120 | $10–90 |
| View generation and filtering | 5–10 | $2–8 |
| Debugging, failed and preempted runs (+50%) | 70–200 | $25–150 |
| **Total** | **≈ 200–600 GPU-h** | **≈ $75–450** on marketplace 4090/L4; **≈ $400–1,500** on Lambda A100; 3–5× more on hyperscalers |

### 3.5 Three local tiers

| Tier | What is feasible |
|---|---|
| **CPU-only laptop** | Head-only / cached-feature AugMult studies (the thesis regime) with many seeds; Ettin-17M/32M full DP fine-tuning on SST-2/MNLI subsets; unit tests (AugMult grad == `torch.func` reference on a tiny BERT); NLI-filter scoring of small view sets. **Enough for Phase 0–1 and the calibration / rising-loss study.** |
| **Single consumer GPU 12–24 GB** | Everything in Phase 2 at base size: full/LoRA DP on base encoders with K ≤ 16 (smaller physical batch); DP-LoRA on 0.3–1.7B decoders (24 GB comfortable, 12 GB tight); vLLM view generation with 2–4B generators. |
| **Rented cloud** | Use for bursts: full SST-2 / MNLI sweeps at K=16, RoBERTa-large / 400M Ettin comparisons with Li et al., 4B+ decoders, and the bulk view generation if no local GPU. Marketplace 4090/L4 is the best $/run for base-size DP; A100/H100 only for large-K or 1B+ throughput. |

---

## 4. DP on small LLMs: practicality

* **Opacus 1.6.0** (May 2026) ✅:
  * Requires torch ≥ 2.6.
  * `wrap_model=False` (non-wrapping mode) for HF Transformers / Accelerate compatibility.
  * FSDP, mixed and low precision, custom accountants.
  * Earlier: v1.5.3 brought ghost clipping for embeddings, adaptive clipping, per-sample grads under ghost clipping, and a **LoRA + PEFT tutorial**; v1.5.4 added an RMSNorm hook and `register_full_backward_hook` ✅.
  * Source: [Opacus releases](https://github.com/meta-pytorch/opacus/releases).
* **Precision** ✅ ([PyTorch blog](https://pytorch.org/?p=4746)):
  * DP utility is stable under FP32/BF16/mixed precision **with LoRA**, but degrades when normalisation layers are trained in low precision.
  * So: use LoRA for bf16 DP, and keep norm params frozen or in fp32.
* **Ghost clipping vs LoRA:** ghost clipping gives little memory benefit for LoRA (narrow effective layers), so use standard per-sample grads for LoRA. Ghost clipping pays off for full fine-tuning of wide linear layers ✅.
* **Reported pitfall (unverified, single source):** an Opacus issue (#820, May 2026) reportedly involved DP-LoRA with PEFT 0.18.x silently not updating LoRA weights. It was attributed to device-placement order: move the base model to CUDA *before* `get_peft_model` ([blog post](https://imranahamed.substack.com/p/the-dp-lora-silent-corruption-how)) ⚠️. **Mitigation:** assert in tests that LoRA params change after one DP step, and that `grad_sample` is populated for every trainable param.
* **Gradient checkpointing:**
  * It re-runs the forward pass during backward, which can make forward hooks fire twice. Hook-based per-sample grads (Opacus `GradSampleModule`, *and this repo's AugMult GSM*) can double-count or misalign activations.
  * Prefer functorch/`torch.func` per-sample grads, or verify against a non-checkpointed reference on a tiny model ⚠️ (known class of issue; confirm on the current Opacus version).
* **AugMult on new architectures:** the repo's GSM only has K-aware samplers for `nn.Linear` and `nn.LayerNorm`.
  * ModernBERT/Ettin/Gemma/Qwen use RMSNorm, gated MLPs and rotary embeddings.
  * Use a **functional AugMult** (`vmap(grad(mean-over-K loss))`) or LoRA-only trainable params (all `nn.Linear`), rather than extending the hook fork.
* **JAX-Privacy 1.0** (Nov 2025) ✅:
  * DP-SGD and DP-FTRL, Keras and Flax APIs, auditing; it powered VaultGemma.
  * A natural fit for **Gemma/VaultGemma via KerasHub**, and AugMult-style multiplicity exists in its image pipelines.
  * Worth it only if you are willing to move to JAX. Otherwise stay on PyTorch + Opacus.
* **fastDP** (`awslabs/fast-differential-privacy`, Bu et al.): book-keeping / mixed ghost clipping plus BiTFiT. Fast for full fine-tuning of HF encoders. Less active maintenance ⚠️ (check its last release before relying on it).
* **Accounting:** use the PRV/PLD accountant (Opacus `accountant="prv"`) or Google `dp_accounting`, and δ ≪ 1/n.

---

## 5. Sources

* NLI4CT: [BigBio sem_eval_2024_task_2](https://huggingface.co/datasets/bigbio/sem_eval_2024_task_2/blob/main/sem_eval_2024_task_2.py), [community mirror](https://huggingface.co/datasets/AshtonIsNotHere/nli4ct_semeval2024/blame/main/README.md), [NLI4CT-P paper](https://arxiv.org/abs/2404.04963v1)
* [BioNLI](https://github.com/StonyBrookNLP/BioNLI), [SciNLI](https://github.com/msadat3/SciNLI)
* [Asclepius synthetic notes](https://huggingface.co/datasets/starmpcc/Asclepius-Synthetic-Clinical-Notes/blob/3af51d604650fe71ce271df822d7a8d314964fab/README.md?code=true)
* [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), [TAB](https://arxiv.org/abs/2202.00443)
* User-level: [Google blog on user-level DP fine-tuning](https://research.google/blog/fine-tuning-llms-with-user-level-differential-privacy/), [Chua et al.](https://arxiv.org/abs/2406.14322)
* Models:
  * [Ettin blog](https://huggingface.co/blog/ettin), [Ettin card](https://huggingface.co/jhu-clsp/ettin-encoder-32m), [Ettin ICLR 2026](https://iclr.cc/virtual/2026/poster/10006491)
  * [BioClinical ModernBERT](https://arxiv.org/pdf/2506.10896), [Clinical ModernBERT](https://arxiv.org/pdf/2504.03964)
  * [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m), [Gemma 4 (SiliconANGLE)](https://siliconangle.com/?p=737125)
  * [VaultGemma docs](https://huggingface.co/docs/transformers/model_doc/vaultgemma), [VaultGemma report](https://arxiv.org/pdf/2510.15001v2)
  * [Qwen3.5 Small](https://machineherald-io.pages.dev/article/2026-03/15-alibaba-qwen-35-small-models-pack-multimodal-intelligence-into-sub-10-billion-parameters-for-edge-devices)
  * [OLMo 3](https://www.emergentmind.com/papers/2512.13961), [SmolLM3](https://osai-index.eu/model/smollm), [NT-Java-1.1B](https://huggingface.co/infosys/NT-Java-1.1B/commit/1101810720a9d54377b5d0deaca2a83a4d7dbd06)
* Compute and DP practice:
  * [Li et al. 2022](https://arxiv.org/pdf/2110.05679v6.pdf), [DP-BiTFiT](https://arxiv.org/pdf/2210.00036)
  * [Opacus releases](https://github.com/meta-pytorch/opacus/releases), [Opacus mixed precision](https://pytorch.org/?p=4746)
  * [vLLM hyperparameter study](https://arxiv.org/pdf/2408.01050), [offline batching guide](https://www.parasail.io/guides/offline-llm-inference-batching-throughput)
  * GPU price pages listed in §3.3.
