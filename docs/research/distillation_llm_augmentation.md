# LLM-generated views in AugMult = distillation? Literature, framing, and controls

*Research note, Nov 2026. Scope: before committing to LLM paraphrases or other generator-made views as AugMult views, understand how far the DP student learns the **generator's** knowledge rather than the private data, and design experiments that can tell the two apart.*

Citation markers:
* ✓ = title, venue and link verified by search for this note.
* † = cited from memory (well known, but not re-verified here).

---

## 1. TL;DR

1. **The privacy guarantee survives, but only under conditions.** Views must come from a *fixed, public* generator that sees **only that one example** (no other private examples in the prompt, no fine-tuning on private data). Under those conditions, everything is a per-example computation that happens before clipping, so sensitivity and ε/δ are unchanged. **Breaking any condition breaks the guarantee** (see §5.2).
2. **Utility conclusions are where the danger lies.** With LLM views, AugMult becomes a hybrid: *private labels* combined with a *public teacher's notion of which inputs are equivalent* (vicinal risk minimization with a learned vicinity). If the generator can already do the task, or has seen the benchmark (SST-2, MNLI and QNLI are inside common instruction-tuning mixtures), then paraphrasing can covertly "clean up" or relabel inputs. Gains would then be attributed to AugMult when they are really **knowledge transfer from the teacher**.
3. **Model collapse is a weak risk here.** The generator is fixed, the loop is not recursive, and the real example is always in the per-example average, which is the "accumulate" regime where collapse does not occur. The real risk is **tail erosion**: paraphrasers normalize rare phrasings and jargon, so the student under-trains on exactly the rare forms that matter.
4. **The distillation framing suggests a better design.** A *public* teacher can supply soft targets on each view (function matching, Beyer et al. 2022) *inside* the per-example loss, which is still DP. The KD ≈ learned-label-smoothing result (Yuan et al. 2020) also links to the project's calibration and "rising loss" thread. But this makes the public-only baselines mandatory.
5. **The minimal set of controls** (§6) is:
   * (a) public-only baselines (teacher zero/few-shot, student distilled on public data);
   * (b) a generator ladder from weak to strong, plus non-generator views at equal compute;
   * (c) label-flip and semantic-drift measurements of the views;
   * (d) contamination checks, using fully open models (OLMo/Pythia) for the main claims;
   * (e) report **Δ_priv**: the DP student minus the best public-only baseline.

---

## 2. Knowledge distillation: what transfers and how

| Work | Key point for us |
|---|---|
| Hinton, Vinyals, Dean 2015, *Distilling the Knowledge in a Neural Network* † [arXiv 1503.02531](https://arxiv.org/abs/1503.02531) | Soft targets carry "dark knowledge", i.e. the relative probabilities of wrong classes. This is the *output-side* transfer channel. |
| Kim & Rush 2016, *Sequence-Level Knowledge Distillation*, EMNLP † [arXiv 1606.07947](https://arxiv.org/abs/1606.07947) | **Data-centric KD.** Training a student on the teacher's *generated outputs* (hard sequences) is itself distillation. Any time we train on generator text, we are doing a form of this. |
| Furlanello et al. 2018, *Born-Again Neural Networks*, ICML † [arXiv 1805.04770](https://arxiv.org/abs/1805.04770) | Even an *equal-capacity* teacher improves the student. KD acts partly as a regularizer, not only as capacity transfer. |
| Beyer et al. 2022, *Knowledge Distillation: A Good Teacher Is Patient and Consistent*, CVPR ✓ [arXiv 2106.05237](https://arxiv.org/abs/2106.05237) | KD works best as **function matching**: the teacher and student see the *same augmented view*, with aggressive augmentation and long schedules. This is essentially "AugMult + soft targets", which makes it the closest template for a teacher-in-the-loop AugMult variant. |
| Stanton et al. 2021, *Does Knowledge Distillation Really Work?*, NeurIPS † [arXiv 2106.05945](https://arxiv.org/abs/2106.05945) | Students often generalize better without matching the teacher closely (low fidelity). Augmentations increase teacher–student agreement. So "the student learned the teacher's function" is a *measurable* quantity, via agreement on held-out data. |
| Yuan et al. 2020, *Revisiting KD via Label Smoothing Regularization*, CVPR ✓ [arXiv 1909.11723](https://arxiv.org/abs/1909.11723) | KD ≈ **learned label smoothing**, and a "teacher-free" virtual teacher recovers much of the gain. **Implication:** some apparent "teacher knowledge" may just be regularization or calibration. Test it with a label-smoothing control. |
| Müller, Kornblith, Hinton 2019, *When Does Label Smoothing Help?*, NeurIPS ✓ [arXiv 1906.02629](https://arxiv.org/abs/1906.02629) | Label smoothing improves calibration but makes a *worse teacher*, because it erases inter-class similarity in the logits. This is relevant if we use our own DP model or a smoothed model as a teacher. |
| Chapelle et al. 2001, *Vicinal Risk Minimization*, NeurIPS † | The formal lens on augmentation: we minimize risk over a *vicinity distribution* around each training point. With LLM views, the vicinity is **defined by the generator**. |

**Two transfer channels.** These should be kept apart in all analyses:
* **Output-side (labels/soft targets):** the teacher tells the student *what the answer is*. This is classical KD.
* **Input-side (vicinity):** the teacher tells the student *which inputs should get the same answer*. Plain AugMult with LLM paraphrases uses only this channel, *as long as* the paraphrase does not change the label-relevant content.

The problem is that a task-capable paraphraser can leak the output-side channel *through* the input side. For example, it can rewrite an ambiguous review into an unambiguously positive one, or "normalize" a hypothesis so that it is entailed.

---

## 3. LLM-generated data as distillation

| Work | Key point |
|---|---|
| Meng et al. 2022, *Generating Training Data with Language Models: Towards Zero-Shot Language Understanding* (SuperGen), NeurIPS † [arXiv 2202.04538](https://arxiv.org/abs/2202.04538) | Label-conditioned generation from a PLM, used to train a small classifier with **no real data**. This is pure teacher transfer, and an upper bound on what "public-only" can do. |
| Ye et al. 2022, *ZeroGen: Efficient Zero-shot Learning via Dataset Generation*, EMNLP ✓ [arXiv 2202.07922](https://arxiv.org/abs/2202.07922) | Small task models trained only on PLM-generated data can be competitive with prompting the PLM itself. A **mandatory public-only baseline** for SST-2-like tasks. |
| Ye et al. 2022, *ProGen*, Findings of EMNLP † (arXiv 2210.12329, ID unverified) | Feedback-guided generation, with similar results using about 1% of the data. |
| Gao et al. 2023, *Self-Guided Noise-Free Data Generation (SunGen)*, ICLR † | Reweights noisy synthetic samples. The noise in generated labels is a known first-order problem. |
| West et al. 2022, *Symbolic Knowledge Distillation*, NAACL ✓ [arXiv 2110.07178](https://arxiv.org/abs/2110.07178) | Generate, then filter with a **critic**, then train a student that is 100× smaller and surpasses the teacher. Filtering is what makes generated data useful. |
| Jung et al. 2024, *Impossible Distillation*, NAACL ✓ [arXiv 2305.16635](https://arxiv.org/abs/2305.16635) / [ACL](https://aclanthology.org/2024.naacl-long.250) | High-quality **paraphrase and summary** datasets distilled from small (GPT-2-scale) LMs with filtering. Weak generators plus filters can make good paraphrasers, which is relevant to a cheap, *weak-teacher* view generator. |
| Hsieh et al. 2023, *Distilling Step-by-Step!*, Findings of ACL ✓ [arXiv 2305.02301](https://arxiv.org/abs/2305.02301) | LLM rationales as extra supervision. This is output-side transfer through auxiliary targets. |
| Dai et al. 2023, *AugGPT: Leveraging ChatGPT for Text Data Augmentation* ✓ [arXiv 2302.13007](https://arxiv.org/abs/2302.13007) | ChatGPT paraphrases with large few-shot gains, including on medical text. This is the canonical "LLM views help" result, but in few-shot, non-DP, closed-teacher settings. |
| Li et al. 2023, *Synthetic Data Generation with LLMs for Text Classification: Potential and Limitations*, EMNLP ✓ [arXiv 2310.07849](https://arxiv.org/abs/2310.07849) | Synthetic data helps less on **subjective** tasks, because diversity is low and the LLM's neutrality fails to match the majority labels. SST-2 is partly subjective, and MedNLI depends on clinical judgement. |
| Cegin et al. 2025, *LLMs vs Established Text Augmentation Techniques for Classification: When do the Benefits Outweigh the Costs?*, NAACL ✓ [ACL 2025.naacl-long.526](https://preview.aclanthology.org/setup/2025.naacl-long.526) | LLM augmentation beats classic methods **only in some regimes**, mostly with very few seeds/examples. The costs often outweigh the gains. |
| Longpre et al. 2020, *How Effective is Task-Agnostic Data Augmentation for Pretrained Transformers?*, Findings of EMNLP † [arXiv 2010.01764](https://arxiv.org/abs/2010.01764) | EDA and back-translation give **little or no gain** for pretrained transformers, even at small n. This is consistent with the thesis's SST-2 null result. Pretraining already encodes most of the invariances that classic augmentation adds. |
| Ding et al. 2024, *Data Augmentation using LLMs* survey ✓ [arXiv 2403.02990](https://arxiv.org/abs/2403.02990) | Taxonomy and open problems, including label consistency and diversity–fidelity trade-offs. |

**Takeaways for us:**
* **When generated data helps:** low data, objective tasks, filtered views, and a teacher that is *more* capable than the student on the task. That last condition is exactly when the confound is largest.
* **When it hurts:** subjective labels, unfiltered label drift (the thesis's "inherent error"), low diversity, and strong pretrained students that already know the invariances.
* **Fidelity vs diversity:** a high-fidelity paraphrase adds little new invariance, while a high-diversity one risks flipping the label. The useful operating point has to be *measured* (§6, C4), not assumed.

---

## 4. Model collapse and "learning the generator's distribution"

| Work | Key point |
|---|---|
| Shumailov et al. 2024, *AI models collapse when trained on recursively generated data*, Nature 631:755–759 ✓ [doi 10.1038/s41586-024-07566-y](https://doi.org/10.1038/s41586-024-07566-y) | **Tails disappear first** under recursive training on model output. |
| Alemohammad et al. 2024, *Self-Consuming Generative Models Go MAD*, ICLR ✓ [arXiv 2307.01850](https://arxiv.org/abs/2307.01850) | Without enough *fresh real data* each generation, precision or diversity degrades. |
| Gerstgrasser et al. 2024, *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data* ✓ [arXiv 2404.01413](https://arxiv.org/abs/2404.01413) | **Accumulating** real and synthetic data (rather than replacing) keeps the error bounded. |

**How this applies to AugMult.** Our setting differs from the collapse setting in three ways:
* It is not recursive: the generator is never retrained on student outputs.
* The real example is always present in each per-example average (view 0 is the original).
* The student is a *discriminative* model, not a generator.

Collapse in the strict sense is therefore not expected. Two residual effects remain:
1. **Tail erosion inside views.** Paraphrasers regress toward typical phrasing. They expand or normalize clinical abbreviations, drop hedges and negation scope, and smooth sarcasm in reviews. With K−1 generated views and one real view, the real phrasing gets weight 1/K in the per-example gradient.
   * Mitigations: give the original a fixed weight (e.g. w₀ = 0.5, with the remaining views sharing 0.5); use diversity-promoting decoding; track the gain on rare-token or long-tail test slices.
2. **Covariate shift.** The student trains mostly on generator-style text and is tested on human text. The thesis's "accuracy deficit on augmented samples" is partly this effect. Measure it: the student's accuracy on real vs generated views, and a domain-classifier separability check.

---

## 5. DP × distillation / public teachers

### 5.1 Literature

| Work | Key point |
|---|---|
| Papernot et al. 2017, *Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data* (PATE), ICLR ✓ [arXiv 1610.05755](https://arxiv.org/abs/1610.05755) | Private teachers, noisy aggregation, and a student trained on *public unlabeled* data. Here distillation *is* the privacy mechanism. |
| Papernot et al. 2018, *Scalable Private Learning with PATE*, ICLR ✓ [arXiv 1802.08908](https://arxiv.org/abs/1802.08908) | PATE at many classes and with noisy data. |
| Mireshghallah et al. 2022, *Differentially Private Model Compression*, NeurIPS ✓ [proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bd6bb13e78da078d8adcabbe6d9ca737-Abstract.html) | DP knowledge distillation and pruning for BERT on GLUE. KD on private data must itself be DP. |
| Flemings & Annavaram 2024, *DistilDP: DP KD via Synthetic Text Generation* ✓ [arXiv 2403.00932](https://arxiv.org/abs/2403.00932) | A DP-trained teacher generates synthetic text, and the student learns from hard and soft labels. A compression route with *private* teachers. |
| Wang et al. 2024, *Can Public LLMs Help Private Cross-device Federated Learning?*, Findings of NAACL ✓ (search-verified; [ACL Anthology](https://aclanthology.org/) entry 2024.findings-naacl, pp. 934–949) | **Distilling a public LLM** into the on-device model before DP training gives a sizeable gain at tight ε. This is the "public teacher as prior" pattern, i.e. the baseline our controls must include. |
| Yue et al. 2023, *Synthetic Text Generation with DP: A Simple and Practical Recipe*, ACL ✓ [arXiv 2210.14348](https://arxiv.org/abs/2210.14348) | DP-fine-tune a generator, then train on its samples. |
| Xie et al. 2024, *Aug-PE*, ICML ✓ [arXiv 2403.01749](https://arxiv.org/abs/2403.01749) | DP synthetic text via API-only LLM "variation" calls. The LLM's paraphrasing ability is central, and the private data only *selects* among candidates. |
| *How to DP-fy Your Data* 2025 ✓ [arXiv 2512.03238](https://arxiv.org/abs/2512.03238) | Practical guide to DP synthetic data. |
| Tramèr, Kamath, Carlini 2024, *Considerations for DP Learning with Large-Scale Public Pretraining*, ICML ✓ [arXiv 2212.06470](https://arxiv.org/abs/2212.06470) | Benchmarks overstate the value of public knowledge when the public and private distributions overlap. "Public" generators may contain the private distribution, or even the benchmark itself. |
| Contamination: Sainz et al. 2023, *NLP Evaluation in Trouble*, Findings of EMNLP †; Deng et al. 2024 survey ✓ [arXiv 2406.14644](https://arxiv.org/abs/2406.14644); Cheng et al. 2025 survey ✓ [arXiv 2502.14425](https://arxiv.org/abs/2502.14425) (names GLUE as typical pretraining contamination); Fu et al. 2024 ✓ [arXiv 2410.18966](https://arxiv.org/abs/2410.18966) (instance-level detectors ≈ random) | **SST-2/GLUE contamination is likely** in web-scale generators. Instance-level detection is unreliable, so prefer models whose training data is *known*. |
| Flan Collection (Longpre et al. 2023) † [arXiv 2301.13688](https://arxiv.org/abs/2301.13688) | Instruction-tuning mixtures include GLUE tasks such as SST-2, MNLI and QNLI as *supervised tasks*. An instruct-tuned paraphraser may literally have been trained to classify our benchmark. |
| OLMo ✓ [arXiv 2402.00838](https://arxiv.org/abs/2402.00838), Dolma ✓ [arXiv 2402.00159](https://arxiv.org/abs/2402.00159), WIMBD ✓ [arXiv 2310.20707](https://arxiv.org/abs/2310.20707), Pythia/The Pile † [arXiv 2304.01373](https://arxiv.org/abs/2304.01373), infini-gram † [arXiv 2401.17377](https://arxiv.org/abs/2401.17377) | Fully open models and **searchable** pretraining corpora. WIMBD found parts of GLUE/SuperGLUE in common corpora. These make contamination checks *possible* rather than guesswork. |

### 5.2 Exactly when the DP guarantee holds with generated views

Let the per-example loss be
`L_i(θ) = Σ_k w_k · ℓ(f_θ(v_ik), y_i)` with views `v_ik = G(x_i; ξ_ik)`.

DP-SGD clips `∇L_i` and adds noise, so the guarantee is unchanged **iff** each `∇L_i` depends only on `(x_i, y_i, θ, public randomness, public models)`.

| Practice | Safe? | Why |
|---|---|---|
| A fixed public paraphraser / MT model prompted with `x_i` only | ✅ | A per-example randomized transform. |
| Precomputing and caching views offline | ✅ | Equivalent to the above, **but the cache is private data**. Never release or share it, and do not send it to hosted APIs (DUA/ToS). |
| Filtering or weighting views with a public NLI or teacher model, using a *fixed* threshold | ✅ | Per-example. |
| Choosing the filter threshold, prompt or generator by looking at private training data statistics or private validation accuracy | ⚠️ | Hyperparameter selection leakage, the same issue as all DP tuning. Tune on public proxies or account for it (Papernot & Steinke 2022). |
| **Few-shot prompts containing *other* private examples** | ❌ | Mixes examples before clipping. One example then influences many per-example gradients, so sensitivity grows. |
| Label-conditioned generation using `y_i` | ✅ privacy, ⚠️ semantics | Still per-example, but it leaks the label *into the input*. This is no longer "augmentation" and inflates the label-preservation statistics. |
| A generator fine-tuned (non-DP) on the private set | ❌ | The generator becomes a function of the whole dataset. |
| A teacher providing soft targets `p_T(·|v_ik)` inside `L_i` (public teacher) | ✅ | Per-example. A DP form of function matching. |
| Using the *student's own* (DP) predictions on views, e.g. for weighting or consistency | ✅ | θ is already a DP output (post-processing). |
| Generator memorized the public benchmark (contamination) | ✅ privacy, ❌ **validity** | Does not affect DP, but inflates utility and invalidates conclusions. |

**Conceptual framing for the paper.** DP-AugMult with generator views minimizes, privately, a *vicinal* risk whose vicinity is a **public prior** supplied by G. The ε guarantee covers the private data only. All of G's knowledge is "free", which is legitimate public-data-assisted DP. The scientific claim "AugMult helps" must therefore be decomposed:

`Gain = (variance/clipping-bias reduction from averaging views)  +  (invariance prior from G)  +  (covert label-side transfer from a task-capable / contaminated G)  −  (label drift + covariate shift)`

Only the first term is "AugMult" in the De et al. sense. The second is legitimate public-knowledge transfer. The third is a confound to eliminate or report.

---

## 6. Experimental controls

Ordered by priority. Costs assume SST-2 plus one or two other public datasets (no PhysioNet access) and local or cheap cloud GPUs.

| ID | Control | Separates | Cost |
|---|---|---|---|
| **C1** | **Public-only baselines:** (a) the generator/teacher zero-shot and few-shot (public demos only) on the test set; (b) ZeroGen/SuperGen-style: a student trained on generator-*synthesized, label-conditioned* data with no private data; (c) the student fine-tuned non-privately on a *public* related dataset (e.g. IMDB/Yelp for SST-2, MNLI for other NLI). | Defines the floor. **Report Δ_priv = DP student − max(C1a, C1b, C1c).** If Δ_priv ≤ 0, the DP training is not justified. | Low–medium |
| **C2** | **Generator ladder at equal K:** none (K=1) → model-space views (dropout, token/span cutoff, embedding noise) → classic (EDA/AEDA, Opus-MT and NLLB back-translation) → weak LM paraphraser (Pythia-410M/1.4B, OLMo-1B, Impossible-Distillation-style) → strong instruct LLM (OLMo-2-7B-Instruct; optionally a closed-data model as a contrast). Record each generator's *own task accuracy*. | If the gain tracks the generator's task accuracy rather than view-quality metrics, that is evidence of **teacher transfer**. If it tracks view diversity at fixed label preservation, that is evidence of an **AugMult/invariance** effect. | Medium (views precomputed once) |
| **C3** | **Equal-compute non-LLM views:** match FLOPs and wall clock. AugMult-K with dropout or cutoff views vs LLM views, and vs no AugMult with a K× larger batch at the same ε. | Separates the "averaging/variance" benefit from "generator content". | Low |
| **C4** | **View diagnostics** (computed with *public* models only): label-flip rate (a public NLI or sentiment model run on original vs view), semantic similarity (embedding cosine, BERTScore), lexical diversity (self-BLEU, distinct-n), length change, rare-token retention, and a domain-classifier AUC (real vs generated). | Explains *why* views help or hurt. The thesis's "inherent error" becomes quantitative. | Low |
| **C5** | **Prompt-blind vs task-aware generation:** generic "paraphrase this" vs "paraphrase, preserving sentiment" vs label-conditioned. Measure the label-flip *direction*: does the paraphraser push ambiguous items toward its own predicted label? | Detects **covert relabeling**, i.e. the output-side channel leaking through inputs. | Low |
| **C6** | **Contamination:** run the main claims with **fully open-data generators** (OLMo/Dolma, Pythia/Pile), and check n-gram overlap of the benchmark test sets with WIMBD or infini-gram. Report results on (i) a dataset released after the generator's data cutoff, or (ii) a perturbed/held-out test variant. Avoid instruction-tuned generators whose mixtures include the target task (Flan includes SST-2/MNLI/QNLI). | Separates genuine invariance from memorized benchmark knowledge. | Medium |
| **C7** | **Regularization-only controls:** label smoothing, a confidence penalty, Tf-KD-style virtual teacher, and DP temperature scaling, each at matched ε. | KD ≈ label smoothing (Yuan et al.). Checks that "teacher knowledge" is not just calibration. This doubles as the project's rising-loss/calibration test. | Low |
| **C8** | **Teacher–student agreement:** fidelity (top-1 agreement, KL) between the student and the generator-as-classifier on held-out *real* data, compared with no-AugMult and non-LLM-view students. | Directly measures how much the student "became the teacher" (Stanton et al. 2021). | Low |
| **C9** | **Data-size sweep** (n = 500 → full) for all of the above. | Teacher transfer should dominate at small n and fade at large n. A pure AugMult variance effect should behave differently. | Medium |
| **C10** | **Real-view weight:** w₀ ∈ {1/K, 0.25, 0.5}, plus rare-phrasing and long-tail test slices. | Tail erosion and covariate shift (§4). | Low |
| **Diag.** | *Non-DP diagnostic only:* swap view sets between examples with the same label. | Whether gains come from per-example invariance or just from extra in-class text. Not DP-valid; for analysis only. | Low |

**Reporting rule.** Every table gets a column for the best public-only baseline (C1) and the generator's identity, training-data openness, and its own task accuracy.

---

## 7. Does the distillation framing suggest a better design?

Yes. Three DP-valid variants of the per-example loss, in increasing reliance on the teacher:

1. **Teacher-weighted AugMult (input-side only).** Set `w_k ∝ p_T(y_i | v_ik)` (or a hard filter) with a *public* NLI or sentiment model T, keeping w₀ fixed. This directly suppresses label-breaking views, i.e. the thesis's main failure mode, while transferring very little output-side knowledge.
2. **Consistency AugMult (no teacher).** `L_i = mean_k CE(f(v_ik), y_i) + λ · mean_k KL(f(x_i) ‖ f(v_ik))`, in the R-Drop/UDA style. The student is its own teacher, so this is post-processing-safe and needs no extra public knowledge. It is a clean way to get an invariance benefit *without* teacher transfer.
3. **Function-matching AugMult (output-side, Beyer et al. 2022).** `L_i = mean_k [ (1−α) CE(f(v_ik), y_i) + α · KL(p_T(·|v_ik) ‖ f(v_ik)) ]`, with the teacher and student seeing the *same* view.
   * This is still per-example, so ε is unchanged.
   * It is probably the strongest variant for utility.
   * It is also the most likely to fix DP miscalibration, because soft targets act like learned label smoothing.
   * But it is unambiguously distillation. It must be compared against C1(b/c), the teacher distilled on *public* data with no DP at all. A cheaper alternative with the same knowledge is to do public KD first (non-private, free), then DP-AugMult fine-tuning on private data. Report both.

**Recommendation.** Position the paper around 1 and 2 as "AugMult proper", with 3 as an upper-bound/ablation, and report Δ_priv throughout. This keeps the scientific claim (*per-example multi-view objectives under DP*) separate from the engineering claim (*a public LLM makes DP-NLP better*). The second claim is true, but it is already known (public-data-assisted DP).

---

## 8. Practical notes for this repo

* Precompute views into a versioned store (`example_id, view_id, generator, decoding, prompt_id, text_a, text_b, flip_score, sim_score`). Keep it **out of git**, because it is derived from the dataset.
* Use only local or open-weight generators for anything derived from credentialed data. For public datasets (SST-2 etc.) hosted APIs are allowed privacy-wise but hurt reproducibility and contamination control. Prefer OLMo/Pythia for the headline results.
* Fix decoding seeds, and log the generator revision hash.

---

## 9. References

✓ = verified for this note, † = from memory.

* ✓ Beyer et al. 2022, *Knowledge Distillation: A Good Teacher Is Patient and Consistent*, CVPR. https://arxiv.org/abs/2106.05237
* † Hinton, Vinyals, Dean 2015, *Distilling the Knowledge in a Neural Network*. https://arxiv.org/abs/1503.02531
* † Kim & Rush 2016, *Sequence-Level Knowledge Distillation*, EMNLP. https://arxiv.org/abs/1606.07947
* † Furlanello et al. 2018, *Born-Again Neural Networks*, ICML. https://arxiv.org/abs/1805.04770
* † Stanton et al. 2021, *Does Knowledge Distillation Really Work?*, NeurIPS. https://arxiv.org/abs/2106.05945
* ✓ Yuan et al. 2020, *Revisiting KD via Label Smoothing Regularization*, CVPR. https://arxiv.org/abs/1909.11723
* ✓ Müller, Kornblith, Hinton 2019, *When Does Label Smoothing Help?*, NeurIPS. https://arxiv.org/abs/1906.02629
* † Chapelle, Weston, Bottou, Vapnik 2001, *Vicinal Risk Minimization*, NeurIPS.
* † Meng et al. 2022, *Generating Training Data with Language Models* (SuperGen), NeurIPS. https://arxiv.org/abs/2202.04538
* ✓ Ye et al. 2022, *ZeroGen*, EMNLP. https://arxiv.org/abs/2202.07922
* † Ye et al. 2022, *ProGen*, Findings of EMNLP (arXiv ID 2210.12329 unverified).
* † Gao et al. 2023, *Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning* (SunGen), ICLR.
* ✓ West et al. 2022, *Symbolic Knowledge Distillation*, NAACL. https://arxiv.org/abs/2110.07178
* ✓ Jung et al. 2024, *Impossible Distillation*, NAACL. https://aclanthology.org/2024.naacl-long.250 / https://arxiv.org/abs/2305.16635
* ✓ Hsieh et al. 2023, *Distilling Step-by-Step!*, Findings of ACL. https://arxiv.org/abs/2305.02301
* ✓ Dai et al. 2023, *AugGPT*. https://arxiv.org/abs/2302.13007
* ✓ Li et al. 2023, *Synthetic Data Generation with LLMs for Text Classification: Potential and Limitations*, EMNLP. https://arxiv.org/abs/2310.07849
* ✓ Cegin et al. 2025, *LLMs vs Established Text Augmentation Techniques for Classification*, NAACL. https://preview.aclanthology.org/setup/2025.naacl-long.526
* † Longpre et al. 2020, *How Effective is Task-Agnostic Data Augmentation for Pretrained Transformers?*, Findings of EMNLP. https://arxiv.org/abs/2010.01764
* ✓ Ding et al. 2024, *Data Augmentation using LLMs: Data Perspectives, Learning Paradigms and Challenges*. https://arxiv.org/abs/2403.02990
* ✓ Shumailov et al. 2024, *AI models collapse when trained on recursively generated data*, Nature 631. https://doi.org/10.1038/s41586-024-07566-y
* ✓ Alemohammad et al. 2024, *Self-Consuming Generative Models Go MAD*, ICLR. https://arxiv.org/abs/2307.01850
* ✓ Gerstgrasser et al. 2024, *Is Model Collapse Inevitable?* https://arxiv.org/abs/2404.01413
* ✓ Papernot et al. 2017, *Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data* (PATE), ICLR. https://arxiv.org/abs/1610.05755
* ✓ Papernot et al. 2018, *Scalable Private Learning with PATE*, ICLR. https://arxiv.org/abs/1802.08908
* ✓ Mireshghallah et al. 2022, *Differentially Private Model Compression*, NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2022/hash/bd6bb13e78da078d8adcabbe6d9ca737-Abstract.html
* ✓ Flemings & Annavaram 2024, *Differentially Private Knowledge Distillation via Synthetic Text Generation* (DistilDP). https://arxiv.org/abs/2403.00932
* ✓ Wang et al. 2024, *Can Public Large Language Models Help Private Cross-device Federated Learning?*, Findings of NAACL.
* ✓ Yue et al. 2023, *Synthetic Text Generation with Differential Privacy: A Simple and Practical Recipe*, ACL. https://arxiv.org/abs/2210.14348
* ✓ Xie et al. 2024, *Differentially Private Synthetic Data via Foundation Model APIs 2: Text* (Aug-PE), ICML. https://arxiv.org/abs/2403.01749
* ✓ *How to DP-fy Your Data: A Practical Guide to Generating Synthetic Data With Differential Privacy*, 2025. https://arxiv.org/abs/2512.03238
* ✓ Tramèr, Kamath, Carlini 2024, *Position: Considerations for DP Learning with Large-Scale Public Pretraining*, ICML. https://arxiv.org/abs/2212.06470
* † Papernot & Steinke 2022, *Hyperparameter Tuning with Renyi Differential Privacy*, ICLR. https://arxiv.org/abs/2110.03620
* † Sainz et al. 2023, *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark*, Findings of EMNLP.
* ✓ Deng et al. 2024, *Unveiling the Spectrum of Data Contamination in Language Models*. https://arxiv.org/abs/2406.14644
* ✓ Cheng et al. 2025, *A Survey on Data Contamination for Large Language Models*. https://arxiv.org/abs/2502.14425
* ✓ Fu et al. 2024, *Does Data Contamination Detection Work (Well) for LLMs?* https://arxiv.org/abs/2410.18966
* † Longpre et al. 2023, *The Flan Collection*. https://arxiv.org/abs/2301.13688
* ✓ Groeneveld et al. 2024, *OLMo*. https://arxiv.org/abs/2402.00838 · ✓ Soldaini et al. 2024, *Dolma*. https://arxiv.org/abs/2402.00159 · ✓ Elazar et al. 2024, *What's In My Big Data?* https://arxiv.org/abs/2310.20707
* † Biderman et al. 2023, *Pythia*. https://arxiv.org/abs/2304.01373 · † Liu et al. 2024, *infini-gram*. https://arxiv.org/abs/2401.17377
