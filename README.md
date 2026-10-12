# CD-GAN reviewer revision -- Rev 4 (`cdgan_reviewer_revision_rev4/`)

Rev 4 answers the second-round reviewer audit of the updated manuscript
(20 points). The model is **unchanged** from Rev 3 (row critic +
original-column feature critics, separate per-head GP, covariance L_dep,
fixed lambda_f); Rev 3 stays untouched in `cdgan_reviewer_revision/`.
Rev 4 changes the data pipeline, adds one optional loss variant, and
replaces the evaluation and reporting layer so every table in the paper
comes from one set of run folders.

## Reviewer point -> code change

| # | Reviewer point | Rev 4 change | Where |
|---|---|---|---|
| 1 | Preprocessing not fully specified | Train-only imputation (numerical: train median; categorical: explicit `__MISSING__` category; ARFF `?` treated as missing); predictors entirely missing in train are dropped and listed; unseen test categories counted; per-run `configs/preprocessing_report.json` + `preprocessing_columns.csv` (split, target mapping, encoding, scaler mean/SD, imputation value, missing counts, covariance-loss space) | `preprocessing.py` |
| 1 | **Bug in Rev 3** | Rev 3 had no imputation; `StandardScaler` passes NaN through, so datasets with missing numerical values (e.g. SICK) fed NaN into the GAN. Reproduced on a toy file: 796 NaN cells in `X_train`. Fixed. | `preprocessing.py` |
| 2, 16 | WD misused on mixed data; evaluation too narrow | Type-aware fidelity: numerical -> normalized WD + KS; categorical -> TV distance + JSD (WD never on categorical). Dependency: mixed association-matrix MAE, class-conditional association MAE, numerical Pearson MAE/Frobenius. Real-vs-synthetic detection AUC (LR and depth-3 gradient boosting, 5-fold CV). | `quality_metrics.py` |
| 3, 14 | Table 4 vs Table 5 inconsistency; negative "F1" | `--mode ablation` now reuses the `full` run folders from `--mode single` (skip-if-complete), so the ablation "Full" row *is* the main-table CD-GAN row. `make_tables.py` writes absolute ablation scores and deltas in separate tables and refuses to write anything if runs disagree. | `run_experiment.py`, `make_tables.py` |
| 6, 15, 18 | Statistics, effect sizes, subgroups | Per metric: mean +/- SD per dataset; Friedman chi2, df, p; Iman-Davenport F, df, p; Nemenyi q_alpha, CD, all-pairs p-matrix, CD diagram; CD-GAN vs each method: Wilcoxon signed-rank with Holm correction, rank-biserial r, median paired difference with bootstrap 95% CI, win/tie/loss; per-dataset seed-paired differences with 95% t-CI and Cohen's d_z; ranks within method families (oversampling / deep generative / statistical). | `stats_analysis.py` |
| 8 | Baseline details; baseline provenance | Baselines on the identical split and preprocessor, scored by the same evaluator; every run stores library versions, hyperparameters, epochs, sample counts, class ratio, runtime. `split_fingerprint.json` (hash of row positions) in every run folder; `make_tables.py` refuses to build tables if two methods used different splits or seed sets. External generators (TWAE, TabDDPM, CTAB-GAN+, TABULA) are trained on exported splits and scored identically. `REAL` (TRTR) reference row. | `run_baselines.py`, `run_utils.py` |
| 9 | Conditioning unclear | `--synthetic-class-ratio {train,balanced}`; `generation_report.json` records requested proportions, generated counts and how z and c are sampled; explicit binary-only check (cond_dim = 1). | `run_experiment.py` |
| 10 | Only "w/o Corr" tested | `--dependency-loss {covariance,pearson}` and ablation variant `pearson_corr` (scale-invariant Pearson-correlation loss); with `wo_corr` and `full` this gives none / covariance / Pearson. | `model.py`, `trainer.py` |
| 11, 12 | GP / L_dep every 5 steps unjustified | Sensitivity axes `gp_every`, `corr_every` (default grid 1, 2, 5, 10) and `lambda_gp` (1, 5, 10), validation split only, each row with training time and s/epoch. | `sensitivity.py` |
| 13 | Cost | `runtime.json`: training time, seconds per epoch, generation time, peak GPU memory (reset per epoch and before generation), parameter count; `table_cost.csv` for all methods. | `run_experiment.py`, `make_tables.py` |
| 17 | Mode collapse not measured | Unique-row ratio, within-synthetic duplicate ratio, category and rare-category (<5%) coverage, nearest-neighbour diversity ratio (syn->syn vs real->real), density/coverage. | `quality_metrics.py` |
| 19 | No minority-class analysis | Utility adds macro-F1, balanced accuracy, MCC, minority precision/recall/F1/PR-AUC (minority label from the real training split); fidelity, association error and coverage recomputed on minority rows only. | `quality_metrics.py` |
| -- | torch >= 2.6 | Checkpoint resume failed under torch >= 2.6 (`weights_only=True` default). Fixed. | `trainer.py` |
| -- | 14 datasets | Generic JSON dataset spec loader (`--dataset SPEC --data-path spec.json`); example in `dataset_specs/`. | `datasets.py` |

Points 4, 5, 7 and 20 (novelty table, references, "generalization", language) are manuscript-only.

## Recommended run order (per dataset; 5 seeds)

```bash
# 0. checks (CPU, then GPU)
python sanity_checks.py --device cpu
python sanity_checks.py --device cuda

# 1. CD-GAN main runs (+ export the splits for external baselines)
python run_experiment.py --mode single --dataset SICK --data-path dataset_38_sick.arff \
    --seeds 0 1 2 3 4 --epochs 1000 --batch-size 128 --lambda-corr 0.05 --lambda-feature 1.0 \
    --n-critic 2 --gp-every 5 --corr-every 5 --device cuda --results-root results --export-splits splits

# 2. ablations (reuses results/SICK/full; trains only the other variants)
python run_experiment.py --mode ablation --dataset SICK --data-path dataset_38_sick.arff \
    --ablation-variants full wo_gp wo_corr wo_dual pearson_corr --seeds 0 1 2 3 4 --epochs 1000 \
    --batch-size 128 --lambda-corr 0.05 --lambda-feature 1.0 --n-critic 2 --gp-every 5 --corr-every 5 \
    --device cuda --results-root results

# 3. baselines on the same splits
python run_baselines.py --dataset SICK --data-path dataset_38_sick.arff \
    --methods REAL SMOTE ADASYN GaussianCopula CTGAN TVAE CopulaGAN --seeds 0 1 2 3 4 \
    --sdv-epochs 300 --results-root results
#    external methods: train on splits/SICK/seed<k>/train.csv, save to external_synth/SICK/seed<k>/TWAE.csv, then
python run_baselines.py --dataset SICK --data-path dataset_38_sick.arff --methods EXTERNAL:TWAE \
    --external-dir external_synth --seeds 0 1 2 3 4 --results-root results

# 4. schedule sensitivity (validation split only; 2-3 datasets, 3 seeds is enough)
python run_experiment.py --mode sensitivity --sensitivity-stage gp_every --dataset SICK \
    --data-path dataset_38_sick.arff --seeds 0 1 2 --epochs 300 --device cuda --results-root results
python run_experiment.py --mode sensitivity --sensitivity-stage corr_every --dataset SICK \
    --data-path dataset_38_sick.arff --seeds 0 1 2 --epochs 300 --device cuda --results-root results

# 5. all paper tables + statistics, after every dataset is done
python make_tables.py --results-root results --out-dir paper_tables
```

Other datasets: write a JSON spec (see `dataset_specs/EXAMPLE_adult_openml1590.json`) and use
`--dataset SPEC --data-path dataset_specs/<name>.json` in every command.

## Output of `make_tables.py`

`table_main_utility.csv`, `table_fidelity.csv`, `table_diversity.csv`, `table_minority.csv`,
`table_privacy.csv`, `table_cost.csv`, `table_ablation_absolute.csv`, `table_ablation_delta.csv`,
`consistency_report.txt`, `all_runs_long.csv`, and `stats/` (Friedman/Nemenyi/Wilcoxon-Holm, effect
sizes, CD diagrams, `summary.md` with the exact numbers to quote).

## What was verified in this revision (and what was not)

Executed for real, on CPU with PyTorch 2.14 (first end-to-end execution of this package):

* `sanity_checks.py --device cpu`: all Rev 3 checks plus the new Section 27 checks pass.
* `--mode single` and `--mode ablation` end-to-end on two small synthetic mixed-type datasets with
  missing values, an all-missing column and `?` tokens (2 seeds, 3-4 epochs); ablation correctly
  reused the `full` folders.
* `run_baselines.py` with REAL, SMOTE, ADASYN, GaussianCopula, CTGAN, TVAE (SDV 1.38.5) and an
  EXTERNAL file; `make_tables.py` consistency checks (they caught a real split mismatch caused by a
  pandas upgrade during testing, which led to the row-position fingerprint); `stats_analysis.py`
  reproduces the manuscript's Table 4a average ranks and gives Nemenyi CD = 4.035 for k = 11, N = 14.
* `--mode sensitivity` (gp_every axis) and `--mode regression_check`.

Not verified here: GPU execution, the real benchmark files (including the SICK loader with the new
imputation), long training, and LightGBM (not installed here; evaluation skips it with a warning).
The toy-data scores are meaningless (3-4 epochs) and were used only to test the pipeline.

---

# CD-GAN reviewer revision (`cdgan_reviewer_revision/`)

A controlled reviewer revision of CD-GAN: the code is corrected so it
**strictly matches** the architecture and mathematical assumptions in the
manuscript. No new architecture, no PCGrad, no HSIC, no importance
weighting, no extra discriminators, no attention/transformer components
were added -- see "What was deliberately NOT added" below.

**This is Rev 3** of the package (final pre-training revision). Rev 1
fixed the row/feature-duplication bug with a feature head shared across
every *encoded* dimension; Rev 2 fixed a subtler remaining issue -- a
high-cardinality one-hot column dominating the feature signal -- with the
original-column feature head; **Rev 3 (this pass)** fixes the training-loop
scheduling to literally match the manuscript's n_critic semantics, switches
fidelity evaluation to real-TRAIN-vs-synthetic, completes the NOMAO/MV/
KDDCup09-Upselling loaders, and adds a leak-free hyperparameter-sensitivity
mode plus a performance-regression guard -- see "What changed in Rev 3"
below. Every Rev-1/Rev-2 architectural fix is preserved unchanged.

## Files

| file | contents |
|---|---|
| `model.py` | `Generator`, `RowCriticMLP`, `DualHeadDiscriminator` with an **original-column** feature head (`nn.ModuleList` of column-specific critics), separate `row_gradient_penalty` / `feature_gradient_penalty`, `covariance_correlation_loss`, parameter counting -- **unchanged in Rev 3** |
| `preprocessing.py` | leak-free split -> fit-on-train-only OHE/StandardScaler -> transform, original-column feature-group bookkeeping, train-derived normalization bounds and binary-category orders, categorical-block one-hot projection, inverse-transform, **NEW:** `validate_binary_target`, `prepare_tuning_split` (leak-free 3-way GAN-train/validation/test split) |
| `datasets.py` | generic `DatasetSpec` interface (now with **strict** predictor-column-coverage validation) + per-dataset wrappers -- **SICK, NOMAO, MV, KDDCup09-Upselling are all now fully implemented** with the exact column definitions supplied; raw-space diagnostics printed at load time |
| `trainer.py` | `CDGANConfig` (`feature_group_slices`, `feature_head_version`, **NEW:** `training_loop_version`, `measure_grad_cosine`, `grad_cosine_every`), `CDGANTrainer` with the **corrected per-minibatch n_critic scheduling** and **global, never-epoch-reset** `global_critic_step`/`global_generator_step` counters, expanded CSV logging, stricter checkpoint signature |
| `evaluation.py` | TSTR/augmented downstream utility, KLD/JSD/raw+normalized-Wasserstein/KS, categorical JSD/TV-distance, mixed-type pairwise associations with a deterministic point-biserial mapping, density/coverage, privacy diagnostics (DCR/NNDR/copy ratio) -- **unchanged in Rev 3** |
| `ablation.py` | the four required ablation variants + absolute/aggregated/delta reporting tables -- **unchanged in Rev 3** |
| `common_eval.py` | **NEW:** shared "train one configuration, evaluate it" helper used by BOTH `regression_check.py` and `sensitivity.py`, so they can never silently diverge from `run_experiment.py`'s own single-seed path |
| `regression_check.py` | **NEW (Section 14):** performance-regression guard -- trains a "reference" and a "final" configuration on the identical split and prints a Metric/Reference/Final/Difference table |
| `sensitivity.py` | **NEW (Sections 15-20):** staged, leak-free hyperparameter-sensitivity sweep (`lambda_corr` -> `lambda_feature` -> optional `batch_size`), internal-validation-only, with a documented/configurable selection score |
| `sanity_checks.py` | all Rev-2 checks preserved, **plus NEW Section-26 checks**: exact D-step/G-step counts per minibatch for n_critic=2/3, multi-minibatch global-counter increments, and counter continuity across checkpoint/resume; prints `ALL CD-GAN ARCHITECTURE SANITY CHECKS PASSED` |
| `run_experiment.py` | single CLI entry point, **`--mode sanity\|single\|ablation\|sensitivity\|regression_check`**; fidelity metrics now compare real TRAIN vs synthetic (was real TEST vs synthetic) |

## What changed in Rev 2 (this pass)

1. **Original-column feature critic (the headline fix).** The feature head
   now produces exactly **P** scores (P = number of original predictor
   columns), not D (encoded dimensions). A categorical column's complete
   one-hot block is treated as ONE unit. Mechanically:
   `preprocessing.FittedPreprocessor.original_feature_groups` builds
   `(start, end, name, dtype)` for every original column (categorical:
   its full OHE block; numerical: its single standardized dimension),
   fit strictly on TRAIN. `model.DualHeadDiscriminator` takes these slices
   verbatim (`feature_group_slices` -- never guesses them) and builds
   `nn.ModuleList([_make_column_critic(width_j, cond_dim, hidden_dim) for
   width_j in ...])` -- **one small critic sub-network per original
   column**. **Feature-head parameters are COLUMN-SPECIFIC, not shared**
   (different columns have different input widths, so a single shared MLP
   -- Rev 1's design -- cannot literally apply here). This mechanism
   (ModuleList of per-original-column heads) mirrors your earlier
   `cd_gan_step3_validatedclaude.py`'s `feature_level='original_column'`,
   reused for the grouping/ModuleList structure but WITHOUT its learned
   column embedding (`nn.Embedding`), which Section 27 rules out as
   unnecessary complexity here. `D_feat(x_i,c_i) = (1/P) * sum_j
   D_f,j(x_i^(j), c_i)`.
2. **Reproducibility ordering.** `set_all_seeds(seed)` is now called
   immediately before `build_models(...)` for **every** (dataset, seed,
   configuration) triple -- including every individual ablation variant at
   a given seed (`ablation.run_single_ablation_variant`), which previously
   inherited whatever RNG state training the *previous* variant left
   behind. The same raw train/test split is computed once per seed and
   passed unchanged into all four ablation variants (`run_experiment.py`'s
   `--mode ablation` path; unchanged from Rev 1, re-verified here).
3. **`sanity_checks.py` CUDA bug fixed.** Every tensor/module used in a
   check is now explicitly `.to(device_t)` / constructed with
   `device=device_t`; `python sanity_checks.py --device cuda` no longer
   mixes CPU and CUDA tensors.
4. **Train-only-preprocessing test strengthened.** Replaced the old
   `_check(..., True)` placeholder with a controlled TRAIN={A,B,C} /
   TEST={A,B,C,TEST_ONLY_CATEGORY} example that programmatically asserts
   `"TEST_ONLY_CATEGORY" not in fp.ohe.categories_[0]`, that `transform()`
   still succeeds (`handle_unknown="ignore"`), and that `fp.scaler.mean_`
   matches the TRAIN-only mean and explicitly does **not** match the
   combined train+test mean.
5. **Original feature groups now come from preprocessing, stored in
   config/metadata.** `FittedPreprocessor.original_feature_groups` /
   `.feature_group_slices` is the single source of truth, saved into
   `configs/preprocessor_summary.json` and into every `CDGANConfig`
   (hence every `config.json` and checkpoint signature).
6. **`gp_every`/`corr_every` production default restored to 5** (was left
   at 1 in Rev 1, which silently changed your previous training
   frequency). See the CLI help text / `trainer.py` comment for the
   cost/staleness rationale; both remain fully configurable.
7. **Checkpoint signature is stricter.** Now includes
   `feature_head_version` (so a Rev-1 encoded-dimension checkpoint can
   never be resumed into this Rev-2 original-column model),
   `feature_group_slices_hash`, `lambda_gp`, `lambda_feature`,
   `lambda_corr`, `learning_rate_g`, `learning_rate_d`, `adam_betas`,
   `batch_size`, `n_critic`, `gp_every`, `corr_every`, `use_gp`,
   `use_corr`, `use_class_weights` -- any scientifically relevant setting
   change + `--resume` now refuses to load, rather than silently resuming
   under different hyperparameters.
8. **Normalized Wasserstein distance.** `evaluation.numerical_column_metrics`
   / `statistical_quality_report` now return BOTH `wasserstein_raw` and
   `wasserstein_normalized` (robust min-max via the REAL TRAINING 1st/99th
   percentile, from `FittedPreprocessor.numerical_percentile_bounds`,
   never the test set).
9. **Categorical univariate quality metrics.** New
   `evaluation.categorical_quality_report` -> `categorical_metrics.csv`
   per seed: categorical JSD, total-variation distance, and its
   complement, per original categorical column, against the TRAIN-fitted
   category universe (a category entirely missing from one side is still
   scored, not silently dropped). Evaluation-only, never fed into training.
10. **Point-biserial determinism fixed.** `evaluation.point_biserial` now
    takes an explicit `mapping` argument instead of inferring
    `{first-seen: 0, second-seen: 1}` from each series independently (which
    could flip sign between the real-data and synthetic-data computation).
    `pairwise_relationship_matrix`/`_error` take
    `FittedPreprocessor.binary_category_orders` (train-derived) and use
    membership in that dict -- not a per-call `nunique()` -- to decide
    whether a column is treated as binary, so real and synthetic data are
    always scored under an identical convention.
11. **Privacy/memorization diagnostics.** New
    `evaluation.privacy_diagnostics`: DCR (distance to closest real
    training record), NNDR (nearest-neighbour distance ratio), and an
    exact/near-copy ratio (near-copy threshold derived from the real
    training data's own nearest-neighbour distances, not a fixed magic
    number). Computed REAL TRAINING vs SYNTHETIC (never real test data).
    Saved per-seed (`privacy_metrics.csv`) and aggregated mean+-SD across
    seeds (`privacy_summary.csv`) in `--mode single`. No claim of formal
    differential privacy is made anywhere.
12. **`synthetic_generation_time_sec`** added to `runtime.json` in both
    `run_experiment.py` and `ablation.py` (previously only in
    `run_experiment.py`, and named `generation_time_sec`).

## What changed in Rev 3 (this pass -- final pre-training revision)

1. **n_critic scheduling fix (the headline fix).** `CDGANTrainer.train()`
   now runs, for EVERY minibatch: exactly `n_critic` discriminator updates
   (fresh `z` each), THEN exactly ONE generator update (fresh `z`) --
   replacing Rev 2's interleaved global-counter scheme, which alternated
   D/D->G *across* successive batches rather than doing `n_critic` inner
   D-steps *per batch*. `gp_every`/`corr_every` now gate on two GLOBAL,
   never-epoch-reset counters: `global_critic_step` (incremented once per
   D-step) and `global_generator_step` (incremented once per G-step).
   Verified by the new Section-26 sanity checks: on a single-minibatch toy
   dataset, `n_critic=2` produces exactly 2 D-steps + 1 G-step, `n_critic=3`
   produces exactly 3 D-steps + 1 G-step; on a 3-minibatch epoch,
   `global_critic_step` increases by `3*n_critic` and `global_generator_step`
   by 3.
2. **Checkpoint payload carries the two global counters.** `save_checkpoint`/
   `load_checkpoint` now save/restore `global_critic_step` and
   `global_generator_step` EXACTLY (never reconstructed from
   `epoch * n_batches`, which would drift if `batch_size` or dataset size
   ever changed between runs). A new `training_loop_version` field was
   added to the checkpoint signature specifically so a checkpoint saved
   under Rev 2's OLD scheduling semantics can never be silently resumed
   under Rev 3's new one (the two counters mean different things in each).
3. **GP/covariance logging: skipped vs. applied-and-zero.** `training_history.csv`
   gained `gp_row_applied_mean`, `gp_feature_applied_mean`,
   `gp_total_applied_mean`, `n_gp_evaluations`, `corr_loss_applied_mean`,
   `n_corr_evaluations` -- averaged ONLY over steps where the term was
   actually evaluated (gated by `gp_every`/`corr_every`), with a count of
   how many such evaluations occurred that epoch. The OLD columns
   (`gp_row`, `gp_feature`, `gp_total`, `g_corr_loss`, which average over
   every step including skipped-as-zero ones) are kept unchanged for
   backward compatibility.
4. **Fidelity evaluation is now real-TRAIN-vs-synthetic**, not real-TEST-
   vs-synthetic. `statistical_metrics.csv` / `categorical_metrics.csv` /
   `pairwise_metrics.csv` (density+coverage included) in `run_experiment.py`
   now compare `split["train_df"]` (the RAW dataframe actually used to fit
   the preprocessor) against the synthetic data. This makes the three
   evaluation protocols in this package explicit: **(A) fidelity** = real
   TRAIN vs synthetic; **(B) TSTR** = synthetic-train / real TEST
   (`utility.csv`, unchanged); **(C) privacy** = real TRAIN vs synthetic
   (`privacy_metrics.csv`, was already correct in Rev 2). A `comparison`
   column was added to the fidelity CSVs recording which protocol was used.
5. **NOMAO / MV / KDDCup09-Upselling loaders are now fully implemented**
   with the exact column definitions supplied (verbatim), replacing the
   Rev-2 `NotImplementedError` templates -- see `datasets.py` for the exact
   categorical-column lists and target mappings, and the flagged NOMAO
   polarity ambiguity below.
6. **Strict dataset schema validation.** `DatasetSpec.__post_init__` now
   requires `categorical_cols | numerical_cols` to equal EXACTLY the set
   of non-target columns in the dataframe -- an omitted predictor column or
   an unknown extra entry now raises immediately, rather than silently
   under- or over-specifying the schema. Raw-space diagnostics (rows,
   #predictors, #categorical, #numerical, class counts/percentages) are
   printed at load time; encoded-space diagnostics (`encoded_dim`,
   `#original_feature_groups`) are printed once the preprocessor is fit
   for a given seed (in `run_experiment.py`).
7. **Strict binary-target validation.** New `preprocessing.validate_binary_target`
   is called after every raw-label -> {0,1} mapping (train fit, test/
   validation transform, dataset-level diagnostics) and raises if the
   mapped labels are not exactly `{0, 1}` (both classes present, no stray
   values) -- catching a mis-specified `target_mapping` immediately instead
   of producing a silently degenerate training set.
8. **Performance regression guard (`regression_check.py`, `--mode
   regression_check`).** Trains a "reference" and a "final" configuration
   on the IDENTICAL split/seed/preprocessing/epochs/evaluation protocol
   (internal VALIDATION partition only -- never the final TEST partition),
   and prints a Metric/Reference/Final/Difference table covering F1,
   ROC-AUC, PR-AUC, normalized Wasserstein distance, categorical
   TV-complement, pairwise MAE, density, and coverage. Prints
   `PERFORMANCE REGRESSION DETECTED -- INVESTIGATE BEFORE PRODUCTION` if
   any metric regresses beyond `--regression-tolerance` (default 5%
   relative). **Never auto-reverts anything** -- diagnostic only.
9. **Leak-free hyperparameter sensitivity mode (`sensitivity.py`, `--mode
   sensitivity`).** Staged: Stage A sweeps `lambda_corr` (default grid
   `[0.01, 0.05, 0.1, 0.5, 1.0]`, `lambda_feature` fixed at 1.0); Stage B
   sweeps `lambda_feature` (default `[0.5, 1.0, 2.0]`) using the Stage-A
   winner; optional Stage C compares `batch_size` (default `[64, 128]`).
   `lambda_gp=10`, `n_critic=2`, LR=`2e-4`, `betas=(0.5, 0.9)` stay fixed
   unless explicitly overridden. Every run is scored on an internal
   VALIDATION partition (`preprocessing.prepare_tuning_split`, fit
   preprocessing on the internal GAN-TRAIN subset only) -- the final TEST
   partition is never touched during tuning. The selection score
   (`sensitivity.default_selection_score` = equal-weighted mean of F1/
   ROC-AUC/PR-AUC) is an explicit, named, overridable function, never a
   silently-weighted combination; the full metric table (including every
   fidelity column) is always written out regardless.
10. **`corr_weighted_loss` / `corr_to_adv_ratio` diagnostics.** Logged per
    epoch (evaluated-steps-only mean): `mean(lambda_corr * L_corr)` and its
    ratio to `mean|g_adv_loss|` (epsilon-guarded) -- surfaces whether the
    covariance term is actually competing with the adversarial term at a
    given `lambda_corr`, rather than assuming any particular value is
    "better" a priori.
11. **Generator-side row/feature/combined adversarial decomposition.**
    `g_row_adv`, `g_feature_adv`, `g_combined_adv` logged per epoch
    (diagnostic only -- `g_combined_adv` is the same quantity as the
    existing `g_adv_loss`, never a separate loss term).
12. **Optional diagnostic-only gradient-cosine-similarity** (`--measure-grad-cosine`,
    OFF by default; `--grad-cosine-every`, default 50). Computed with
    `torch.autograd.grad` (never populates `.grad`, so it cannot influence
    `opt_g.step()`); logs the cosine similarity between the generator
    gradients induced by the row-adversarial term alone vs. the
    feature-adversarial term alone. **No PCGrad, no gradient surgery** --
    gradients are never modified based on this measurement.
13. **Reproducibility / ablation-split discipline preserved and extended**
    to the two new tools: `set_all_seeds(seed)` immediately before
    `build_models(...)` for every configuration trained by
    `regression_check.py`/`sensitivity.py`; `sklearn` calls seeded;
    `prepare_tuning_split` is called ONCE per seed and its split reused
    across an entire sweep/comparison.
14. **`sanity_checks.py` Section 26** (see item 1 above for what it
    verifies).

## What was fixed in Rev 1/Rev 2 and PRESERVED here (not undone)

Genuine row/feature branches with disjoint parameters; the **original-column**
feature critic (one score per original predictor column P, categorical
columns' complete one-hot block treated as one unit, equal/unweighted
averaging over P); separate `GP_row`/`GP_feature` computed on the shared
interpolated sample; the covariance-based correlation regularizer
(second-order, linear-dependency only); strict TSTR as the default
evaluation protocol with `augmented` as an explicit secondary mode; a
valid "w/o Dual" ablation (`use_feature_head=False` never instantiates
feature-head parameters, so nothing is double-counted); leak-free
train-only preprocessing; checkpointing/resume; runtime + parameter-count
logging; optional, off-by-default class weighting; privacy/memorization
diagnostics (DCR/NNDR/copy ratio, empirical only); absolute + aggregated +
delta ablation reporting kept in separate tables/columns.

## What was deliberately NOT added

Per your explicit instructions: no Transformer/attention, no HSIC, no
mutual-information loss, no diffusion, no PCGrad, no gradient surgery (the
Rev-3 gradient-cosine-similarity diagnostic in Section 12 above MEASURES
but never MODIFIES gradients), no extra discriminators, no
learned/importance-based feature-aggregation weighting (equal mean only,
over P original columns), no Gumbel-softmax type-aware generator output
(plain linear output layer -- categorical projection happens *after*
generation, in `preprocessing.project_categorical_blocks`, never inside
the generator's forward pass), no learned column embeddings in the feature
head.

## Flagged ambiguities / decisions (NOT silently resolved)

1. **`lambda_corr`**: manuscript reports 1.0, prior code used 0.05. CLI
   default kept at 0.05 (only because it was the more recent operating
   value) -- you choose and pass this explicitly; both are logged.
2. **`batch_size`**: manuscript reports 64, prior code used 128. Same
   treatment.
3. **`gp_every`/`corr_every` default**: you asked for the production
   default to be restored to 5 "unless there's a strong implementation
   reason not to" and to explain rather than silently override. I agree
   and set both to 5 -- there is no *correctness* reason to prefer 1 over
   5 (both are mathematically valid schedules for a WGAN-GP/covariance
   regularizer that changes slowly relative to one mini-batch); 1 only
   costs more compute (an extra double-backward GP pass, and a D×D
   covariance matrix, every single step instead of every 5th). Restored
   to 5 as requested, fully overridable via `--gp-every`/`--corr-every`.
4. **Dataset column definitions for Nomao / MV / KDDCup09-Upselling**:
   RESOLVED in Rev 3 -- you supplied the exact column lists and target
   mappings, now implemented verbatim in `datasets.py`. One sub-ambiguity
   remains and is flagged separately (item 6 below): NOMAO's target
   polarity.
5. **Near-copy threshold in `privacy_diagnostics`**: defined as the 1st
   percentile of real-to-real nearest-neighbour distances (a data-driven,
   auditable choice, returned as `near_copy_threshold` in every output row)
   rather than a fixed absolute distance, since an appropriate absolute
   threshold is dataset/scale-dependent and you did not specify one.
   Flagging this choice explicitly rather than picking a magic number
   silently.
6. **NOMAO target-label polarity (NEW in Rev 3)**: you specified
   `target="class"` "mapped explicitly to {0,1}" for NOMAO, but -- unlike
   MV (`"P"->1, "N"->0`) and KDDCup09-Upselling (`"-1"->0, "1"->1`) -- you
   did not give the raw-label -> {0,1} DIRECTION for NOMAO's two class
   values. `datasets.load_nomao()` accepts an explicit `target_mapping`
   argument; if omitted, it auto-detects the two raw values, maps them by
   SORTED ORDER, and PRINTS the resolved mapping loudly at load time (it
   is never silently guessed without you being told). **Verify the printed
   mapping matches your intended polarity before trusting NOMAO results**,
   or pass `target_mapping={...}` explicitly.

## Verification notes (this assistant's sandbox still has no GPU / no PyTorch)

* All 11 `.py` files pass `python3 -m py_compile` and `python3 -m pyflakes`
  (no syntax errors, no undefined names, no unused imports), including the
  three new modules (`common_eval.py`, `regression_check.py`,
  `sensitivity.py`).
* `preprocessing.py` and `datasets.py` (no PyTorch dependency at import
  time) were integration-tested for real in this environment:
  `prepare_tuning_split` was run against a synthetic 500-row toy dataset
  and produced exactly 320/80/100 (GAN-train/validation/test) rows for a
  400/100 dev/test split with `validation_size=0.2` -- confirming the
  three-way split accounts for every row and the arithmetic is correct;
  `validate_binary_target` was exercised directly and correctly raises on
  a single-class array and on a non-{0,1} array, and passes on a proper
  binary array; `DatasetSpec`'s new strict predictor-coverage check was
  exercised directly and correctly raises on both an omitted predictor
  column and an unlisted extra column, and accepts a valid spec;
  `_normalize_pm1_target` was verified to map a MIXED int/float/string
  `[-1, 1, -1.0, "1", "-1", 1.0]` series to the uniform string labels
  `['-1','1','-1','1','-1','1']`; `_resolve_column_name` was verified to
  resolve `"upselling"` to the real column `"UPSELLING"` case-insensitively.
  Everything already verified in Rev 1/Rev 2 for `preprocessing.py`/
  `evaluation.py` (leak-free fit/transform, normalized-WD scale-invariance,
  categorical JSD/TV-distance, point-biserial determinism, privacy
  diagnostics) is unchanged in Rev 3 and still holds.
* `run_experiment.py`'s new CLI surface (`--mode sensitivity`, `--mode
  regression_check`, and every new flag) was exercised with a stub `torch`
  module standing in for the real one (since this sandbox has neither):
  `build_arg_parser()` builds successfully and every new flag
  (`--reference-lambda-corr`, `--final-lambda-corr`, `--sensitivity-stage`,
  `--run-stage-c`, etc.) parses to the expected value.
* The n_critic-scheduling logic in `trainer.py`'s `train()` was traced by
  hand against the Section-26 sanity-check expectations (single-minibatch
  n_critic=2 -> 2 D-steps+1 G-step; n_critic=3 -> 3 D-steps+1 G-step;
  3-minibatch epoch -> `global_critic_step` +=`3*n_critic`,
  `global_generator_step` += 3; checkpoint/resume restores both counters
  exactly) but **could not be executed end-to-end here** (no `torch`, no
  GPU) -- `sanity_checks.py` now contains these exact checks as real,
  executable PyTorch code (Section 26) for you to run.
* `model.py`, `trainer.py`, `ablation.py`, `common_eval.py`,
  `regression_check.py`, `sensitivity.py`, `sanity_checks.py`, and the
  `--mode single`/`--mode ablation`/`--mode sensitivity`/`--mode
  regression_check` paths of `run_experiment.py` still could **not** be
  executed end-to-end here (no `torch`, no GPU). **You must run the
  commands in "Before any production training" below on your GPU machine
  and confirm they all succeed before anything longer.**

## Exact commands

```bash
conda activate gan_env
cd cdgan_reviewer_revision
pip install -r requirements.txt
```

### Before any production training -- run ONLY these three next (in order)

**A) CPU sanity check:**
```bash
python sanity_checks.py --device cpu
```

**B) GPU sanity check:**
```bash
python sanity_checks.py --device cuda
```
Expect `ALL CD-GAN ARCHITECTURE SANITY CHECKS PASSED` from both, including
the new Section-26 n_critic-scheduling checks. If either fails, stop and
fix it before proceeding -- do not run anything below.
(`run_experiment.py --mode sanity --device cuda` is equivalent to B.)

**C) 5-epoch SICK smoke test:**
```bash
python run_experiment.py \
    --mode single --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 5 --batch-size 128 \
    --lambda-gp 10 --lambda-corr 0.05 --lambda-feature 1.0 --n-critic 2 \
    --gp-every 5 --corr-every 5 --evaluation TSTR --device cuda \
    --results-root results_smoke5
```

**Do not run anything below -- including the 20-epoch smoke test -- until
A, B, and C have all succeeded.**

---
### D) 20-epoch SICK smoke test
```bash
python run_experiment.py \
    --mode single --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 20 --batch-size 128 \
    --lambda-gp 10 --lambda-corr 0.05 --lambda-feature 1.0 --n-critic 2 \
    --gp-every 5 --corr-every 5 --evaluation TSTR --device cuda \
    --results-root results_smoke20
```

### E) Small lambda_corr sensitivity test (leak-free, internal validation only)
```bash
python run_experiment.py \
    --mode sensitivity --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 50 --batch-size 128 \
    --lambda-gp 10 --lambda-feature 1.0 --n-critic 2 --gp-every 5 --corr-every 5 \
    --sensitivity-stage lambda_corr --lambda-corr-grid 0.01 0.05 0.1 0.5 1.0 \
    --test-size 0.2 --validation-size 0.2 --device cuda \
    --results-root results_sensitivity
```
(Omit `--sensitivity-stage` -- default is `staged`, i.e. Stage A ->
Stage B -> optional Stage C in one call; add `--run-stage-c` for Stage C.)

### F) SICK batch_size comparison (64 vs 128, lambda_corr=0.05 fixed)
```bash
python run_experiment.py \
    --mode sensitivity --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 50 --lambda-gp 10 --lambda-feature 1.0 \
    --lambda-corr 0.05 --n-critic 2 --gp-every 5 --corr-every 5 \
    --sensitivity-stage batch_size --batch-size-grid 64 128 \
    --test-size 0.2 --validation-size 0.2 --device cuda \
    --results-root results_sensitivity
```

SICK lambda_corr=0.05 vs 1.0 comparison (same protocol, via the
performance-regression guard rather than the sensitivity sweep, since it
is exactly a two-point A/B comparison):
```bash
python run_experiment.py \
    --mode regression_check --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 50 --batch-size 128 \
    --lambda-gp 10 --lambda-feature 1.0 --n-critic 2 --gp-every 5 --corr-every 5 \
    --reference-lambda-corr 0.05 --final-lambda-corr 1.0 \
    --test-size 0.2 --validation-size 0.2 --regression-tolerance 0.05 \
    --device cuda --results-root results_sensitivity
```

### G) Performance regression check (generic reference vs. final)
```bash
python run_experiment.py \
    --mode regression_check --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --config-name full --seeds 0 --epochs 200 --batch-size 128 \
    --lambda-gp 10 --lambda-feature 1.0 --n-critic 2 --gp-every 5 --corr-every 5 \
    --reference-lambda-corr 0.05 --final-lambda-corr 0.1 \
    --test-size 0.2 --validation-size 0.2 --regression-tolerance 0.05 \
    --device cuda --results-root results_regression
```
Prints a Metric/Reference/Final/Difference table and
`PERFORMANCE REGRESSION DETECTED -- INVESTIGATE BEFORE PRODUCTION` if any
metric regresses beyond `--regression-tolerance`. Never auto-reverts
anything -- read the table and decide.

### H) Three-seed production run
```bash
python run_experiment.py \
    --mode single --dataset NOMAO --data-path /path/to/nomao.arff \
    --config-name full --seeds 0 1 2 --epochs 1000 --batch-size 128 \
    --lambda-gp 10 --lambda-corr 0.05 --lambda-feature 1.0 --n-critic 2 \
    --gp-every 5 --corr-every 5 --evaluation TSTR --device cuda \
    --resume --results-root results
```
(`--resume` is safe to always include: with no existing checkpoints it
just starts from epoch 0.)

### I) Ablation run (all four configurations, 3 seeds)
```bash
python run_experiment.py \
    --mode ablation --dataset SICK --data-path /path/to/dataset_38_sick.arff \
    --ablation-variants full wo_gp wo_corr wo_dual --seeds 0 1 2 \
    --epochs 1000 --batch-size 128 --lambda-gp 10 --lambda-corr 0.05 \
    --lambda-feature 1.0 --gp-every 5 --corr-every 5 --evaluation TSTR \
    --device cuda --results-root results
```
Produces `results/SICK/ablation_table_absolute.csv`,
`ablation_table_aggregated.csv`, and `ablation_table_delta_vs_full.csv`
(three separate files/tables -- absolute scores and deltas are never
mixed in the same column).

### J) Resume interrupted training
Identical to whichever command above was interrupted, with `--resume`
added (or already present). If ANY scientifically relevant flag differs
from the original run (a different `--lambda-corr`, `--batch-size`,
`--gp-every`, or the training-loop scheduling version itself), `trainer.py`
will refuse to resume rather than silently mixing settings or
semantics -- see "Checkpoint signature" below. On successful resume,
`global_critic_step`/`global_generator_step` are restored EXACTLY (never
reconstructed) and printed at startup.

**MV and KDDCup09-Upselling** use the same `--dataset MV --data-path
/path/to/mv.arff` / `--dataset KDDCUP09_UPSELLING --data-path
/path/to/upselling.csv` pattern as NOMAO/SICK above.

## Output structure (Section 24)

```
results/
  SICK/
    full/
      seed0/
        checkpoints/checkpoint_epoch0999.pt
        configs/preprocessor.joblib
        configs/preprocessor_summary.json   # includes original_feature_groups
        training_history.csv
        utility.csv
        statistical_metrics.csv             # wasserstein_raw + wasserstein_normalized
        categorical_metrics.csv             # NEW: categorical JSD / TV-distance
        pairwise_metrics.csv
        privacy_metrics.csv                 # NEW: DCR / NNDR / copy ratios
        runtime.json                        # + synthetic_generation_time_sec
        param_counts.json
        config.json
        package_versions.json
      seed1/  seed2/  ...
      utility_all_seeds.csv
      utility_summary.csv
      privacy_all_seeds.csv
      privacy_summary.csv
    wo_gp/  wo_corr/  wo_dual/
    ablation_table_absolute.csv
    ablation_table_aggregated.csv
    ablation_table_delta_vs_full.csv
    regression_check/                       # NEW (--mode regression_check)
      seed0/
        preprocessor_summary.json
        reference/  final/                  # each: same per-run layout as above
        regression_check_table.csv          # Metric | Reference | Final | Difference | regression_flag
    sensitivity/                             # NEW (--mode sensitivity)
      lambda_corr/seed0/
        0p01/ 0p05/ 0p1/ 0p5/ 1p0/           # one dir per grid point
        sensitivity_lambda_corr_table.csv    # sorted best-first by selection_score
      lambda_feature/seed0/  batch_size/seed0/
```

## Checkpoint signature (Rev 2 Section 13, extended in Rev 3)

A checkpoint can only be resumed into a run whose `dataset_name`,
`config_name`, `seed`, `encoded_dim`, `cond_dim`, `latent_dim`,
`hidden_dim`, `use_feature_head`, `feature_head_version`,
**`training_loop_version`** (NEW in Rev 3 -- distinguishes Rev 2's
interleaved-counter scheduling from Rev 3's per-minibatch scheduling),
`feature_group_slices` (hashed), `lambda_gp`, `lambda_feature`,
`lambda_corr`, `learning_rate_g`, `learning_rate_d`, `adam_betas`,
`batch_size`, `n_critic`, `gp_every`, `corr_every`, `use_gp`, `use_corr`,
and `use_class_weights` all match exactly -- otherwise `load_checkpoint`
raises rather than silently resuming under different settings or
semantics. On a successful resume, `global_critic_step` and
`global_generator_step` (NEW in Rev 3) are restored EXACTLY from the
checkpoint payload, never reconstructed from `epoch * n_batches`.
`measure_grad_cosine`/`grad_cosine_every` are intentionally NOT part of
the signature -- they are diagnostic-only and cannot change what is
actually trained.

## Class weighting (Section 7)

`--use-class-weights` is OFF by default (the manuscript's mathematical
formulation of `L_G,adv` is unweighted). If you pass it, an
inverse-class-frequency weight (normalized to mean 1.0) is applied ONLY to
the generator's adversarial term, and `use_class_weights` plus the
resolved per-class weights are printed and logged in every
`training_history.csv` row / `config.json`.
