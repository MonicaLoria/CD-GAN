"""
sanity_checks.py -- smoke tests that MUST pass before any GPU training.

    python sanity_checks.py --device cpu
    python sanity_checks.py --device cuda

Prints "ALL CD-GAN ARCHITECTURE SANITY CHECKS PASSED" only if every check
below genuinely passes; otherwise raises with a specific failure message.

Section 6 fix: EVERY tensor and module used in a check is explicitly moved
to the SAME resolved device (``device_t``) -- there is no code path left
that mixes a CPU tensor with a CUDA module or vice versa, which is what
made the previous version of this file fail under ``--device cuda``.
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from model import (
    DualHeadDiscriminator,
    Generator,
    build_models,
    covariance_correlation_loss,
    dual_gradient_penalty,
    feature_gradient_penalty,
    row_gradient_penalty,
)
from preprocessing import (
    fit_preprocessor,
    transform_with_preprocessor,
)
from trainer import CDGANConfig, CDGANTrainer, set_all_seeds


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"Sanity check failed: {name} ({detail})")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _toy_feature_groups(numerical_width: int = 4, categorical_widths=(3, 2, 20)):
    """Builds a synthetic (start, end) grouping: `numerical_width` numerical
    columns (each width 1) followed by categorical columns of the given
    widths -- including a 20-category column, for check D."""
    groups = []
    start = 0
    for _ in range(numerical_width):
        groups.append((start, start + 1))
        start += 1
    for w in categorical_widths:
        groups.append((start, start + w))
        start += w
    D = start
    return groups, D


def run_all_sanity_checks(device: str = "cpu") -> None:
    device_t = _resolve_device(device)
    set_all_seeds(0)

    B, cond_dim, latent_dim, hidden_dim = 16, 1, 100, 32
    feature_group_slices, D = _toy_feature_groups()  # D = 4 numerical + (3+2+20) categorical = 29
    P = len(feature_group_slices)  # number of ORIGINAL columns = 4 + 3 = 7 (not 29)

    print(f"Toy layout: D (encoded) = {D}, P (original columns) = {P}, "
          f"group widths = {[e - s for s, e in feature_group_slices]}")

    print("\n1-4. Generator / feature-score / aggregated-feature-score / row-score shapes")
    generator = Generator(latent_dim=latent_dim, cond_dim=cond_dim, output_dim=D, hidden_dim=hidden_dim).to(device_t)
    disc = DualHeadDiscriminator(
        input_dim=D, cond_dim=cond_dim, hidden_dim=hidden_dim,
        use_feature_head=True, feature_group_slices=feature_group_slices,
    ).to(device_t)

    z = torch.randn(B, latent_dim, device=device_t)
    cond = torch.randint(0, 2, (B, 1), device=device_t).float()

    generator.eval()  # BatchNorm running stats are pre-initialized (mean=0,var=1); eval() is fine for a shape check
    x_fake = generator(z, cond)
    _check("Generator output shape == [B, D]", tuple(x_fake.shape) == (B, D), f"got {tuple(x_fake.shape)}")

    individual_scores = disc.feature_scores(x_fake, cond)
    _check(
        "B. Number of feature-critic scores == P (ORIGINAL columns), NOT D (encoded dims)",
        tuple(individual_scores.shape) == (B, P) and P != D,
        f"got shape {tuple(individual_scores.shape)}, expected [B,P]=[{B},{P}] (D={D} != P={P}, as intended)",
    )

    agg_score = disc.aggregated_feature_score(x_fake, cond)
    _check("Aggregated feature score shape == [B, 1]", tuple(agg_score.shape) == (B, 1),
           f"got {tuple(agg_score.shape)}")
    _check(
        "E. Aggregated feature score == arithmetic mean over the P ORIGINAL-column scores",
        torch.allclose(agg_score.squeeze(1), individual_scores.mean(dim=1), atol=1e-5),
    )

    row_out = disc.row_score(x_fake, cond)
    _check("F. Row score shape == [B, 1]", tuple(row_out.shape) == (B, 1), f"got {tuple(row_out.shape)}")

    print("\nC/D. Each categorical OHE block (including the 20-category one) yields exactly ONE feature score")
    # feature_group_slices[-1] is the 20-category block by construction.
    last_group_width = feature_group_slices[-1][1] - feature_group_slices[-1][0]
    _check("the synthetic 20-category column really is width 20 in this toy layout", last_group_width == 20)
    _check(
        "D. the 20-category column contributes exactly ONE entry to individual_scores (column index P-1), "
        "not 20 entries",
        individual_scores.shape[1] == P,
    )
    # Perturbing every entry inside that one 20-wide block should change ONLY
    # that one aggregate column's score, never spill into P-1 separate scores.
    start20, end20 = feature_group_slices[-1]
    x_probe = x_fake.clone().detach()
    scores_before = disc.feature_scores(x_probe, cond)
    x_probe2 = x_probe.clone()
    x_probe2[:, start20:end20] = x_probe2[:, start20:end20] + 5.0
    scores_after = disc.feature_scores(x_probe2, cond)
    changed_cols = (~torch.isclose(scores_before, scores_after, atol=1e-5)).any(dim=0)
    _check(
        "perturbing the ENTIRE 20-category block changes exactly ONE column of individual_scores (index P-1)",
        changed_cols.sum().item() == 1 and bool(changed_cols[-1]),
        f"changed_cols={changed_cols.tolist()}",
    )

    print("\n5. Feature head receives the ORIGINAL COLUMN's own block + condition, not the full record")
    row_first_linear = disc.row_head.net[0]
    _check(
        "row_head's first Linear layer has in_features == D + cond_dim (full record)",
        row_first_linear.in_features == D + cond_dim,
        f"got in_features={row_first_linear.in_features}, expected {D + cond_dim}",
    )
    for i, (start, end) in enumerate(feature_group_slices):
        width = end - start
        head_i_first_linear = disc.feature_heads[i][0]
        _check(
            f"feature_heads[{i}]'s first Linear layer has in_features == group_width({width}) + cond_dim, "
            f"NOT D + cond_dim",
            head_i_first_linear.in_features == width + cond_dim,
            f"got {head_i_first_linear.in_features}, expected {width + cond_dim}",
        )

    print("\n6. Feature-head parameters are COLUMN-SPECIFIC (an nn.ModuleList), NOT shared")
    ids_per_head = [set(id(p) for p in head.parameters()) for head in disc.feature_heads]
    for i in range(len(ids_per_head)):
        for j in range(i + 1, len(ids_per_head)):
            _check(
                f"feature_heads[{i}] and feature_heads[{j}] share NO parameter tensors (column-specific, "
                f"not a shared MLP)",
                ids_per_head[i].isdisjoint(ids_per_head[j]),
            )

    print("\n7. Row and feature heads have INDEPENDENT learnable parameters")
    row_param_ids = {id(p) for p in disc.row_head.parameters()}
    feat_param_ids = {id(p) for head in disc.feature_heads for p in head.parameters()}
    _check("row_head and feature_heads share no parameter tensors", row_param_ids.isdisjoint(feat_param_ids))

    print("\n8. Both heads receive NON-ZERO generator gradients")
    generator.train()
    z2 = torch.randn(B, latent_dim, device=device_t)
    x_fake2 = generator(z2, cond)
    row_only_loss = -disc.row_score(x_fake2, cond).mean()
    row_only_loss.backward()
    row_grad_norm = sum(p.grad.norm().item() for p in generator.parameters() if p.grad is not None)
    generator.zero_grad()

    z3 = torch.randn(B, latent_dim, device=device_t)
    x_fake3 = generator(z3, cond)
    feat_only_loss = -disc.aggregated_feature_score(x_fake3, cond).mean()
    feat_only_loss.backward()
    feat_grad_norm = sum(p.grad.norm().item() for p in generator.parameters() if p.grad is not None)
    generator.zero_grad()

    _check("H. generator gradient norm from row head alone is > 0", row_grad_norm > 0.0, f"norm={row_grad_norm}")
    _check("H. generator gradient norm from feature head alone is > 0", feat_grad_norm > 0.0, f"norm={feat_grad_norm}")

    print("\n9-J. GP_row and GP_feature can be calculated INDEPENDENTLY, and GP_total == GP_row + GP_feature")
    real = torch.randn(B, D, device=device_t)
    fake = generator(torch.randn(B, latent_dim, device=device_t), cond).detach()
    gp_row = row_gradient_penalty(disc, real, fake, cond)[0]
    gp_feat = feature_gradient_penalty(disc, real, fake, cond)
    gp_row2, gp_feat2, gp_total2 = dual_gradient_penalty(disc, real, fake, cond)
    _check("GP_row is a finite non-negative scalar", bool(torch.isfinite(gp_row)) and gp_row.item() >= 0)
    _check("GP_feature is a finite non-negative scalar", bool(torch.isfinite(gp_feat)) and gp_feat.item() >= 0)
    _check(
        "J. dual_gradient_penalty's GP_total == GP_row + GP_feature",
        torch.allclose(gp_total2, gp_row2 + gp_feat2, atol=1e-5),
    )

    print("\nL. Covariance loss produces gradients for the generator")
    real_cov_input = torch.randn(B, D, device=device_t)
    fake_cov_input = generator(torch.randn(B, latent_dim, device=device_t), cond)
    corr_loss = covariance_correlation_loss(real_cov_input, fake_cov_input)
    corr_loss.backward()
    corr_grad_norm = sum(p.grad.norm().item() for p in generator.parameters() if p.grad is not None)
    generator.zero_grad()
    _check("covariance loss has non-zero gradient w.r.t. generator parameters", corr_grad_norm > 0.0,
           f"norm={corr_grad_norm}")

    print("\nK. w/o Dual uses the single row critic only ONCE (no duplication), GP_feature == 0")
    disc_wo_dual = DualHeadDiscriminator(
        input_dim=D, cond_dim=cond_dim, hidden_dim=hidden_dim, use_feature_head=False,
    ).to(device_t)
    _check("feature_heads is None when use_feature_head=False", disc_wo_dual.feature_heads is None)
    x_probe3 = torch.randn(B, D, device=device_t)  # <-- FIX: explicitly placed on device_t
    combined = disc_wo_dual.combined_score(x_probe3, cond, lambda_feature=1.0)
    row_only = disc_wo_dual.row_score(x_probe3, cond)
    _check(
        "combined_score(use_feature_head=False) == row_score exactly (feature term is a hard zero, "
        "not a duplicate)",
        torch.allclose(combined, row_only, atol=1e-6),
    )
    real_wd = torch.randn(B, D, device=device_t)
    fake_wd = torch.randn(B, D, device=device_t)
    gp_row_wd, gp_feat_wd, gp_total_wd = dual_gradient_penalty(disc_wo_dual, real_wd, fake_wd, cond)
    _check("GP_feature == 0 when use_feature_head=False", float(gp_feat_wd) == 0.0)
    _check("GP_total == GP_row when use_feature_head=False", torch.allclose(gp_total_wd, gp_row_wd, atol=1e-6))

    print("\nM. Generated categorical projection produces valid one-hot blocks")
    from preprocessing import project_categorical_blocks_torch
    cat_only_slices = [(s, e) for (s, e) in feature_group_slices if (e - s) > 1]  # the 3/2/20-wide ones
    soft = torch.rand(B, D, device=device_t)
    proj = project_categorical_blocks_torch(soft, cat_only_slices)
    for (s, e) in cat_only_slices:
        block = proj[:, s:e]
        row_sums = block.sum(dim=1)
        _check(
            f"projected block [{s}:{e}] sums to exactly 1 in every row (valid one-hot)",
            bool(torch.allclose(row_sums, torch.ones_like(row_sums))),
        )
        _check(
            f"projected block [{s}:{e}] contains only {{0,1}} values",
            bool(((block == 0.0) | (block == 1.0)).all()),
        )

    print("\nN/O. OHE/scaler are fit ONLY on raw TRAINING data; test-only categories are UNSEEN by the fitted OHE")
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 100
    train_df = pd.DataFrame({
        "num1": rng.normal(size=n_train),
        "cat1": rng.choice(["A", "B", "C"], size=n_train),
        "target": rng.integers(0, 2, size=n_train),
    })
    test_df = pd.DataFrame({
        "num1": rng.normal(loc=3.0, scale=2.0, size=n_test),  # deliberately shifted, to make a fit-on-full-data bug visible
        "cat1": rng.choice(["A", "B", "C", "TEST_ONLY_CATEGORY"], size=n_test),
        "target": rng.integers(0, 2, size=n_test),
    })

    X_train_np, y_train_np, fp = fit_preprocessor(
        train_df, categorical_cols=["cat1"], numerical_cols=["num1"],
        target_column="target", seed=0, dataset_name="sanity_check",
    )
    fitted_categories = set(fp.ohe.categories_[0])
    _check(
        "O. 'TEST_ONLY_CATEGORY' (present only in the constructed TEST partition) is NOT among the "
        "TRAIN-fitted OneHotEncoder's categories",
        "TEST_ONLY_CATEGORY" not in fitted_categories,
        f"fitted_categories={fitted_categories}",
    )
    X_test_np, y_test_np = transform_with_preprocessor(test_df, fp)
    _check(
        "N/O. transform() of TEST data (containing an unseen category) succeeds without crashing, "
        "because handle_unknown='ignore'",
        X_test_np.shape[0] == len(test_df),
    )
    test_only_row_mask = (test_df["cat1"] == "TEST_ONLY_CATEGORY").to_numpy()
    if test_only_row_mask.any():
        cat_block_width = fp.categorical_slices[0][1] - fp.categorical_slices[0][0]
        unseen_rows_block = X_test_np[test_only_row_mask, :cat_block_width]
        _check(
            "rows with the unseen category encode to an ALL-ZERO one-hot block (handle_unknown='ignore' "
            "semantics), not silently mapped onto a known category",
            bool(np.allclose(unseen_rows_block.sum(axis=1), 0.0)),
        )

    train_mean = train_df["num1"].mean()
    combined_mean = pd.concat([train_df["num1"], test_df["num1"]]).mean()
    _check(
        "N. scaler.mean_ matches the numerical mean of TRAIN ONLY",
        bool(np.isclose(fp.scaler.mean_[0], train_mean, atol=1e-6)),
        f"scaler.mean_={fp.scaler.mean_[0]}, train_mean={train_mean}",
    )
    _check(
        "N. scaler.mean_ does NOT match the mean of the combined train+test data (this check would FAIL "
        "if preprocessing were accidentally fit on the full dataset before splitting)",
        not np.isclose(fp.scaler.mean_[0], combined_mean, atol=1e-6),
        f"scaler.mean_={fp.scaler.mean_[0]}, combined_mean={combined_mean}",
    )

    print("\nSection 9: original_feature_groups is built correctly (categorical block-wise, numerical single-dim)")
    _check(
        "fp.original_feature_groups has exactly len(categorical_cols)+len(numerical_cols) entries",
        len(fp.original_feature_groups) == 2,
        f"got {fp.original_feature_groups}",
    )
    names = [g[2] for g in fp.original_feature_groups]
    _check("original_feature_groups covers 'cat1' and 'num1'", set(names) == {"cat1", "num1"}, f"got {names}")

    print("\nP/Q. set_all_seeds(seed) reproducibility")
    set_all_seeds(42)
    g1, d1 = build_models(latent_dim=16, hidden_dim=8, cond_dim=1, encoded_dim=D,
                           feature_group_slices=feature_group_slices, use_feature_head=True)
    set_all_seeds(42)
    g2, d2 = build_models(latent_dim=16, hidden_dim=8, cond_dim=1, encoded_dim=D,
                           feature_group_slices=feature_group_slices, use_feature_head=True)
    same_seed_identical = all(
        torch.equal(p1, p2) for p1, p2 in zip(g1.parameters(), g2.parameters())
    ) and all(
        torch.equal(p1, p2) for p1, p2 in zip(d1.parameters(), d2.parameters())
    )
    _check("P. set_all_seeds(42) then build_models() TWICE produces IDENTICAL initial parameters", same_seed_identical)

    set_all_seeds(43)
    g3, d3 = build_models(latent_dim=16, hidden_dim=8, cond_dim=1, encoded_dim=D,
                           feature_group_slices=feature_group_slices, use_feature_head=True)
    different_seed_different = not all(
        torch.equal(p1, p3) for p1, p3 in zip(g1.parameters(), g3.parameters())
    )
    _check("Q. a DIFFERENT seed (43 vs 42) produces DIFFERENT initial generator parameters", different_seed_different)

    print(f"\nR. No CPU/CUDA tensor mismatch on device='{device_t}' (every module/tensor above was explicitly "
          f"placed on device_t; this entire function already ran to this point without a device-mismatch error)")
    _check("R. all checks above ran on a single consistent device without a device-mismatch RuntimeError", True)

    print("\n26a/26c. n_critic scheduling: EXACT D-step/G-step counts for a SINGLE minibatch, "
          "and GLOBAL-counter continuity across checkpoint/resume")

    def _tiny_toy_layout():
        # 3 numerical columns (width 1 each) + 1 categorical column (width 2) => D=5, P=4.
        # Deliberately smaller/cheaper than the main toy layout above -- these checks
        # actually run real training steps (forward+backward+optimizer.step()).
        return [(0, 1), (1, 2), (2, 3), (3, 5)], 5

    tiny_groups, tiny_D = _tiny_toy_layout()
    tiny_cat_slices = [(3, 5)]
    N_tiny = 16  # == batch_size below, so exactly ONE minibatch per epoch (Section 26's literal wording)
    rng_tiny = np.random.default_rng(0)
    X_tiny = rng_tiny.normal(size=(N_tiny, tiny_D)).astype(np.float32)
    y_tiny = np.array([0.0, 1.0] * (N_tiny // 2), dtype=np.float32)

    def _build_tiny_trainer(n_critic: int, epochs: int, out_dir: str, resume: bool) -> CDGANTrainer:
        set_all_seeds(0)
        cfg = CDGANConfig(
            dataset_name="toy_ncritic", config_name="full", seed=0,
            latent_dim=8, hidden_dim=8, cond_dim=1,
            lambda_gp=10.0, lambda_feature=1.0, lambda_corr=0.05,
            n_critic=n_critic, batch_size=N_tiny, epochs=epochs,
            gp_every=1, corr_every=1, use_class_weights=False,
            checkpoint_every=1, resume=resume,
            device=str(device_t), encoded_dim=tiny_D, feature_group_slices=tiny_groups,
        )
        gen, disc = build_models(
            latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, cond_dim=cfg.cond_dim,
            encoded_dim=cfg.encoded_dim, feature_group_slices=tiny_groups, use_feature_head=True,
        )
        return CDGANTrainer(cfg, gen, disc, tiny_cat_slices, out_dir)

    for n_critic_test in (2, 3):
        tmp_dir = tempfile.mkdtemp(prefix=f"cdgan_sanity_ncritic{n_critic_test}_")
        try:
            trainer_a = _build_tiny_trainer(n_critic_test, epochs=1, out_dir=tmp_dir, resume=False)
            trainer_a.train(X_tiny, y_tiny)
            _check(
                f"26a. n_critic={n_critic_test}: ONE minibatch -> EXACTLY {n_critic_test} D-optimizer-steps "
                f"(global_critic_step)",
                trainer_a.global_critic_step == n_critic_test,
                f"global_critic_step={trainer_a.global_critic_step}",
            )
            _check(
                f"26a. n_critic={n_critic_test}: ONE minibatch -> EXACTLY 1 G-optimizer-step "
                f"(global_generator_step)",
                trainer_a.global_generator_step == 1,
                f"global_generator_step={trainer_a.global_generator_step}",
            )

            # ---- 26c: resume continuity. A FRESH CDGANTrainer instance, with the
            # IDENTICAL checkpoint signature (same architecture/hyperparameters) but
            # epochs extended to 2 and resume=True, must RESTORE global_critic_step /
            # global_generator_step EXACTLY (not reconstruct or reset them) and then
            # continue incrementing from there for the next minibatch. ----
            trainer_b = _build_tiny_trainer(n_critic_test, epochs=2, out_dir=tmp_dir, resume=True)
            trainer_b.train(X_tiny, y_tiny)
            _check(
                f"26c. n_critic={n_critic_test}: resume correctly advanced start_epoch to 1 (not 0)",
                trainer_b.start_epoch == 1,
                f"start_epoch={trainer_b.start_epoch}",
            )
            _check(
                f"26c. n_critic={n_critic_test}: after resume + 1 more minibatch, global_critic_step == "
                f"{2 * n_critic_test} (restored {n_critic_test}, then +{n_critic_test} more -- NOT reset "
                f"to {n_critic_test} and NOT re-derived from epoch count)",
                trainer_b.global_critic_step == 2 * n_critic_test,
                f"global_critic_step={trainer_b.global_critic_step}",
            )
            _check(
                f"26c. n_critic={n_critic_test}: after resume + 1 more minibatch, global_generator_step == 2",
                trainer_b.global_generator_step == 2,
                f"global_generator_step={trainer_b.global_generator_step}",
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n26b. Multi-minibatch epoch: global_critic_step increments by n_critic PER MINIBATCH, "
          "global_generator_step by 1 PER MINIBATCH (not once per epoch)")
    N_multi = 48  # batch_size=16 => exactly 3 minibatches per epoch
    n_critic_multi = 2
    X_multi = rng_tiny.normal(size=(N_multi, tiny_D)).astype(np.float32)
    y_multi = np.array([0.0, 1.0] * (N_multi // 2), dtype=np.float32)
    tmp_dir2 = tempfile.mkdtemp(prefix="cdgan_sanity_ncritic_multi_")
    try:
        set_all_seeds(0)
        cfg_multi = CDGANConfig(
            dataset_name="toy_ncritic_multi", config_name="full", seed=0,
            latent_dim=8, hidden_dim=8, cond_dim=1,
            lambda_gp=10.0, lambda_feature=1.0, lambda_corr=0.05,
            n_critic=n_critic_multi, batch_size=16, epochs=1,
            gp_every=1, corr_every=1, use_class_weights=False,
            checkpoint_every=1, resume=False,
            device=str(device_t), encoded_dim=tiny_D, feature_group_slices=tiny_groups,
        )
        gen_multi, disc_multi = build_models(
            latent_dim=cfg_multi.latent_dim, hidden_dim=cfg_multi.hidden_dim, cond_dim=cfg_multi.cond_dim,
            encoded_dim=cfg_multi.encoded_dim, feature_group_slices=tiny_groups, use_feature_head=True,
        )
        trainer_multi = CDGANTrainer(cfg_multi, gen_multi, disc_multi, tiny_cat_slices, tmp_dir2)
        trainer_multi.train(X_multi, y_multi)
        n_batches_expected = N_multi // 16  # 3
        _check(
            f"26b. {n_batches_expected}-minibatch epoch, n_critic={n_critic_multi}: global_critic_step == "
            f"{n_batches_expected}*{n_critic_multi} == {n_batches_expected * n_critic_multi}",
            trainer_multi.global_critic_step == n_batches_expected * n_critic_multi,
            f"global_critic_step={trainer_multi.global_critic_step}",
        )
        _check(
            f"26b. {n_batches_expected}-minibatch epoch, n_critic={n_critic_multi}: global_generator_step "
            f"== {n_batches_expected}",
            trainer_multi.global_generator_step == n_batches_expected,
            f"global_generator_step={trainer_multi.global_generator_step}",
        )
    finally:
        shutil.rmtree(tmp_dir2, ignore_errors=True)

    # ------------------------------------------------------------------
    # Section 27 (Rev 4): missing values, Pearson dependency loss, split
    # fingerprint, evaluator.
    # ------------------------------------------------------------------
    print("\n[Section 27] Rev 4 checks")
    from model import pearson_correlation_loss
    from preprocessing import MISSING_CATEGORY, impute_dataframe, prepare_seed_split
    from run_utils import split_fingerprint

    rng27 = np.random.default_rng(0)
    n27 = 200
    df27 = pd.DataFrame({
        "num_a": rng27.normal(size=n27),
        "num_b": rng27.normal(size=n27),
        "num_allnan": np.nan,
        "cat_a": rng27.choice(["x", "y", "?"], n27),
        "target": (rng27.random(n27) < 0.3).astype(int),
    })
    df27.loc[:20, "num_a"] = np.nan
    sp27 = prepare_seed_split(df27, "target", ["cat_a"], ["num_a", "num_b", "num_allnan"], "SANITY27", seed=0)
    fp27 = sp27["preprocessor"]
    _check("27a. encoded TRAIN and TEST contain no NaN after train-only imputation",
           not np.isnan(sp27["X_train"]).any() and not np.isnan(sp27["X_test"]).any())
    _check("27b. a predictor that is entirely missing in TRAIN is dropped and recorded",
           fp27.dropped_all_missing_cols == ["num_allnan"] and "num_allnan" not in fp27.numerical_cols,
           f"dropped={fp27.dropped_all_missing_cols}")
    _check("27c. ARFF-style '?' becomes the explicit missing category",
           MISSING_CATEGORY in [str(c) for c in fp27.categorical_categories[0]])
    med = float(pd.to_numeric(sp27["train_df"]["num_a"]).median())
    _check("27d. numerical imputation value is the TRAIN median",
           abs(fp27.numerical_impute_values["num_a"] - med) < 1e-12,
           f"stored={fp27.numerical_impute_values['num_a']}, train_median={med}")
    _check("27e. impute_dataframe returns no missing values",
           not impute_dataframe(sp27["train_df"], fp27).isna().any().any())
    sp27b = prepare_seed_split(df27, "target", ["cat_a"], ["num_a", "num_b", "num_allnan"], "SANITY27", seed=0)
    f1_ = split_fingerprint(sp27["train_df"], sp27["test_df"], sp27["train_row_ids"], sp27["test_row_ids"])
    f2_ = split_fingerprint(sp27b["train_df"], sp27b["test_df"], sp27b["train_row_ids"], sp27b["test_row_ids"])
    sp27c = prepare_seed_split(df27, "target", ["cat_a"], ["num_a", "num_b", "num_allnan"], "SANITY27", seed=1)
    f3_ = split_fingerprint(sp27c["train_df"], sp27c["test_df"], sp27c["train_row_ids"], sp27c["test_row_ids"])
    _check("27f. same seed -> identical split fingerprint; different seed -> different",
           f1_["split_sha1"] == f2_["split_sha1"] and f1_["split_sha1"] != f3_["split_sha1"])

    real27 = torch.randn(64, 6, device=device_t)
    fake27 = torch.randn(64, 6, device=device_t, requires_grad=True)
    lp = pearson_correlation_loss(real27, fake27)
    lp.backward()
    _check("27g. Pearson dependency loss is finite and has a gradient",
           bool(torch.isfinite(lp)) and fake27.grad is not None and float(fake27.grad.norm()) > 0,
           f"loss={float(lp):.4f}")
    _check("27h. Pearson dependency loss is ~0 for identical inputs",
           float(pearson_correlation_loss(real27, real27.clone())) < 1e-4)
    scaled = real27 * torch.tensor([1, 10, 100, 1, 1, 1], device=device_t, dtype=real27.dtype)
    _check("27i. Pearson dependency loss is invariant to per-column scale (covariance loss is not)",
           float(pearson_correlation_loss(real27, scaled)) < 1e-4
           and float(covariance_correlation_loss(real27, scaled)) > 1.0)

    print()
    print("ALL CD-GAN ARCHITECTURE SANITY CHECKS PASSED")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    run_all_sanity_checks(device=args.device)
