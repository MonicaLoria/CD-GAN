"""
trainer.py -- CD-GAN training loop, checkpointing/resume, CSV logging,
synthetic data generation.

Implements EXACTLY the losses in Sections 4-7 of the reviewer-revision spec:

    L_D      = E[Dr(x~,c) + lam_f*Dfeat(x~,c)]
             - E[Dr(x ,c) + lam_f*Dfeat(x ,c)]
             + lam_gp * (GP_r + GP_f)

    L_G,adv  = - E[Dr(x~,c) + lam_f*Dfeat(x~,c)]

    L_G      = L_G,adv + lam_corr * L_corr

No PCGrad, no gradient-conflict resolution, no HSIC, no importance
weighting, no extra discriminators -- a controlled reviewer revision, not
a new architecture (Section 25).

FINAL PRE-TRAINING REVISION (Section 1): the training loop below now
implements the LITERAL manuscript n_critic semantics -- for every
minibatch: n_critic discriminator updates (fresh z each), THEN exactly
ONE generator update (fresh z) -- instead of the previous version's
interleaved global-counter scheme, which alternated D/D->G across
successive batches rather than doing n_critic inner D-steps per batch.
``gp_every``/``corr_every`` now gate on two GLOBAL counters
(``global_critic_step``, ``global_generator_step``) that are never reset
per epoch and are saved/restored exactly on checkpoint/resume (Section 2).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from model import (
    DualHeadDiscriminator,
    Generator,
    covariance_correlation_loss,
    pearson_correlation_loss,
    count_parameters,
    dual_gradient_penalty,
)
from preprocessing import project_categorical_blocks_torch


# ======================================================================
# Config
# ======================================================================
# Bumped whenever the FEATURE-HEAD MECHANISM changes (not just its
# hyperparameters), so a checkpoint from the earlier encoded-dimension
# shared-MLP feature head (Rev 1) can never be silently resumed into the
# current original-column, column-specific-ModuleList feature head (Rev 2).
FEATURE_HEAD_VERSION = "original_column_v2_no_embedding"

# Bumped whenever the TRAINING-LOOP SCHEDULING SEMANTICS change (not just
# hyperparameters). Rev 3 replaced the interleaved global-critic-counter
# scheme (in which gp_every/corr_every counted D-steps and G-steps that
# were NOT literally "n_critic D-steps then 1 G-step per minibatch") with
# the exact manuscript scheduling described in the module docstring. A
# checkpoint saved under the OLD scheduling semantics must never be
# silently resumed under the NEW one, because global_critic_step /
# global_generator_step mean something different in each -- resuming
# across the boundary would desynchronize gp_every/corr_every gating from
# what actually happened during earlier training.
TRAINING_LOOP_VERSION = "per_minibatch_ncritic_then_one_g_global_counters_v3"


@dataclass
class CDGANConfig:
    dataset_name: str
    config_name: str = "full"           # one of: full, wo_gp, wo_corr, wo_dual  (or custom)
    seed: int = 0

    latent_dim: int = 100
    hidden_dim: int = 128
    cond_dim: int = 1

    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    adam_betas: Tuple[float, float] = (0.5, 0.9)

    lambda_gp: float = 10.0
    lambda_feature: float = 1.0
    lambda_corr: float = 0.05   # NOTE: manuscript reports 1.0, current code used 0.05.
                                 # Not silently resolved -- see README "Known
                                 # manuscript/code mismatches". Set explicitly
                                 # per experiment via --lambda-corr.
    n_critic: int = 2

    batch_size: int = 128        # NOTE: manuscript reports 64. Not silently
                                  # resolved -- set explicitly via --batch-size.
    epochs: int = 1000

    # gp_every/corr_every gate on the GLOBAL (never epoch-reset)
    # global_critic_step / global_generator_step counters (Section 1-2).
    # Production default restored to 5 (see Rev 2 README notes: 1 is the
    # mathematically "purest" choice but roughly doubles/triples wall-clock
    # cost per epoch for a regularizer that changes slowly relative to a
    # single mini-batch -- a cost/staleness trade-off, not a correctness
    # one). Both remain fully configurable via the CLI.
    gp_every: int = 5            # compute GP every N GLOBAL critic steps
    corr_every: int = 5          # compute L_corr every N GLOBAL generator steps

    # --- ablation switches (Section 14 / 20) ---
    use_gp: bool = True
    use_corr: bool = True
    use_feature_head: bool = True   # False => "w/o Dual": single row critic only

    # --- Rev 4 (reviewer point 10): which dependency regularizer L_dep is
    # used when use_corr=True. "covariance" = ||Cov(real)-Cov(fake)||_F
    # (the manuscript's loss, unchanged default); "pearson" =
    # ||Corr(real)-Corr(fake)||_F, scale-free. use_corr=False = no L_dep.
    dependency_loss: str = "covariance"

    # --- Section 7: class weighting is OPTIONAL and OFF by default ---
    use_class_weights: bool = False

    checkpoint_every: int = 50
    resume: bool = False

    device: str = "cuda"
    amp: bool = False   # kept OFF by default: GP is AMP-sensitive

    encoded_dim: int = -1  # filled in once the preprocessor is fit

    # ORIGINAL-COLUMN feature-group boundaries (start, end) pairs, taken
    # VERBATIM from FittedPreprocessor.feature_group_slices -- never guessed.
    # Required whenever use_feature_head=True.
    feature_group_slices: List[Tuple[int, int]] = field(default_factory=list)
    feature_head_version: str = FEATURE_HEAD_VERSION
    training_loop_version: str = TRAINING_LOOP_VERSION

    # ---- Section 22 (follow-up revision): OPTIONAL, diagnostic-only
    # cosine-similarity measurement between the generator gradients induced
    # by the row-adversarial term alone vs the feature-adversarial term
    # alone. Purely observational: computed with torch.autograd.grad, which
    # never populates .grad and therefore cannot influence the actual
    # optimizer step. NO PCGrad, NO gradient surgery, NO change to which
    # gradients the optimizer applies -- OFF by default because it costs
    # two extra backward-shaped passes through the generator every time it
    # is evaluated. ----
    measure_grad_cosine: bool = False
    grad_cosine_every: int = 50   # in units of global_generator_step

    def __post_init__(self):
        if self.dependency_loss not in {"covariance", "pearson"}:
            raise ValueError(f"dependency_loss must be 'covariance' or 'pearson', got {self.dependency_loss!r}")

    def as_dict(self) -> Dict:
        d = asdict(self)
        d["adam_betas"] = list(self.adam_betas)
        d["feature_group_slices"] = [list(t) for t in self.feature_group_slices]
        return d


ABLATION_FLAGS: Dict[str, Dict[str, bool]] = {
    # Section 14
    "full":    dict(use_feature_head=True,  use_gp=True,  use_corr=True),
    "wo_gp":   dict(use_feature_head=True,  use_gp=False, use_corr=True),
    "wo_corr": dict(use_feature_head=True,  use_gp=True,  use_corr=False),
    "wo_dual": dict(use_feature_head=False, use_gp=True,  use_corr=True),
    # Rev 4 (reviewer point 10): same as full, Pearson-correlation L_dep
    "pearson_corr": dict(use_feature_head=True, use_gp=True, use_corr=True, dependency_loss="pearson"),
}


def apply_ablation_flags(config: CDGANConfig) -> CDGANConfig:
    """Force config.use_gp / use_corr / use_feature_head to the canonical
    combination for config.config_name, if it names one of the four
    registered ablation variants. No-op for custom config names."""
    if config.config_name in ABLATION_FLAGS:
        for k, v in ABLATION_FLAGS[config.config_name].items():
            setattr(config, k, v)
    return config


# ======================================================================
# Reproducibility (Section 16)
# ======================================================================
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_gpu_info(device: torch.device) -> None:
    print(f"PyTorch version   : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available    : {cuda_ok}")
    if cuda_ok:
        print(f"CUDA version      : {torch.version.cuda}")
        idx = torch.cuda.current_device() if device.type == "cuda" else 0
        print(f"GPU name          : {torch.cuda.get_device_name(idx)}")
        total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"GPU memory        : {total_mem_gb:.2f} GB")
    else:
        print("GPU memory        : n/a (CPU)")
    print(f"selected device   : {device}")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda requested but CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


# ======================================================================
# Class weights (Section 7 -- optional, off by default)
# ======================================================================
def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency weights, normalized to mean 1.0 over the batch
    they are applied to. Only used if config.use_class_weights=True."""
    classes, counts = np.unique(y_train.astype(int), return_counts=True)
    freq = counts / counts.sum()
    inv = 1.0 / np.clip(freq, 1e-8, None)
    inv = inv / inv.mean()
    return {int(c): float(w) for c, w in zip(classes, inv)}


# ======================================================================
# Checkpointing (Section 13 of the prior revision, STRICTER signature;
# Section 2 of this revision adds the two global scheduling counters)
# ======================================================================
def _feature_group_slices_hash(feature_group_slices: List[Tuple[int, int]]) -> str:
    """Stable short hash of the exact original-column grouping, so two
    datasets/preprocessing runs that happen to share encoded_dim but group
    it differently (e.g. different category counts) are still detected as
    incompatible."""
    payload = json.dumps([list(t) for t in feature_group_slices], sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _model_signature(config: CDGANConfig) -> Dict:
    """Everything that MUST match for a checkpoint to be safely resumable.

    Any SCIENTIFICALLY RELEVANT training setting participates in the
    signature, not just architecture shape -- changing lambda_gp,
    lambda_feature, lambda_corr, learning rates, batch_size, n_critic,
    gp_every, corr_every, use_gp, use_corr, use_class_weights, or the
    training-loop scheduling semantics themselves, and passing --resume,
    must refuse to load the incompatible checkpoint rather than silently
    continuing training under different hyperparameters/semantics.

    ``measure_grad_cosine``/``grad_cosine_every`` are intentionally NOT
    part of the signature: they are diagnostic-only (computed via
    ``torch.autograd.grad``, never populating ``.grad``) and cannot change
    what is actually trained, so toggling them does not make a checkpoint
    incompatible.
    """
    return {
        # -- identity / architecture shape --
        "dataset_name": config.dataset_name,
        "config_name": config.config_name,
        "seed": config.seed,
        "encoded_dim": config.encoded_dim,
        "cond_dim": config.cond_dim,
        "latent_dim": config.latent_dim,
        "hidden_dim": config.hidden_dim,
        "use_feature_head": config.use_feature_head,
        "feature_head_version": config.feature_head_version,
        "training_loop_version": config.training_loop_version,
        "feature_group_slices_hash": _feature_group_slices_hash(config.feature_group_slices),
        # -- scientifically relevant training hyperparameters --
        "lambda_gp": config.lambda_gp,
        "lambda_feature": config.lambda_feature,
        "lambda_corr": config.lambda_corr,
        "learning_rate_g": config.learning_rate_g,
        "learning_rate_d": config.learning_rate_d,
        "adam_betas": list(config.adam_betas),
        "batch_size": config.batch_size,
        "n_critic": config.n_critic,
        "gp_every": config.gp_every,
        "corr_every": config.corr_every,
        "use_gp": config.use_gp,
        "use_corr": config.use_corr,
        "dependency_loss": config.dependency_loss,
        "use_class_weights": config.use_class_weights,
    }


def checkpoint_dir(root: str, config: CDGANConfig) -> str:
    return os.path.join(root, config.dataset_name, config.config_name, f"seed{config.seed}", "checkpoints")


def save_checkpoint(
    path: str,
    epoch: int,
    global_critic_step: int,
    global_generator_step: int,
    generator: Generator,
    discriminator: DualHeadDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config: CDGANConfig,
    metrics: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        # Section 2: the two GLOBAL scheduling counters, saved EXACTLY (not
        # reconstructed from epoch * n_batches on resume, which would drift
        # if batch_size or dataset size ever changed between runs).
        "global_critic_step": global_critic_step,
        "global_generator_step": global_generator_step,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "config": config.as_dict(),
        "signature": _model_signature(config),
        "metrics": metrics or {},
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("checkpoint_epoch") and f.endswith(".pt")]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f.replace("checkpoint_epoch", "").replace(".pt", "")))
    return os.path.join(ckpt_dir, candidates[-1])


def load_checkpoint(
    path: str,
    generator: Generator,
    discriminator: DualHeadDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    config: CDGANConfig,
    device: torch.device,
) -> Tuple[int, int, int]:
    """Loads a checkpoint IN PLACE. Raises if its signature does not match
    the current run's architecture/dataset/seed/ablation/scheduling
    configuration (Section 18 of the prior revision: "do not accidentally
    load a checkpoint from another architecture or ablation
    configuration"; Section 2 of this revision extends that to the
    training-loop scheduling semantics via ``training_loop_version``).

    Returns (epoch, global_critic_step, global_generator_step) -- the two
    global counters are restored EXACTLY as saved, never re-derived.
    """
    # Rev 4: our own checkpoints carry numpy RNG state; torch>=2.6 defaults to
    # weights_only=True, which rejects them. These files are written by this
    # package only, so full unpickling is safe here.
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch<1.13 has no weights_only argument
        payload = torch.load(path, map_location=device)
    expected = _model_signature(config)
    found = payload.get("signature", {})
    if found != expected:
        raise RuntimeError(
            "Checkpoint signature mismatch -- refusing to resume from a "
            f"checkpoint belonging to a different run.\n  expected={expected}\n  found   ={found}\n"
            f"  checkpoint path: {path}"
        )
    generator.load_state_dict(payload["generator_state_dict"])
    discriminator.load_state_dict(payload["discriminator_state_dict"])
    opt_g.load_state_dict(payload["optimizer_g_state_dict"])
    opt_d.load_state_dict(payload["optimizer_d_state_dict"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and payload.get("torch_cuda_random_state") is not None:
        torch.cuda.set_rng_state_all(payload["torch_cuda_random_state"])
    epoch = int(payload["epoch"])
    global_critic_step = int(payload.get("global_critic_step", 0))
    global_generator_step = int(payload.get("global_generator_step", 0))
    return epoch, global_critic_step, global_generator_step


# ======================================================================
# CSV logging (Section 19 of the prior revision; Sections 3/20/21/22/26 of
# this revision add the skipped-vs-applied / diagnostic / counter columns)
# ======================================================================
TRAINING_LOG_COLUMNS = [
    "epoch",
    "d_loss_total",
    "d_row_wasserstein",
    "d_feature_wasserstein",
    "gp_row",
    "gp_feature",
    "gp_total",
    "g_adv_loss",
    "g_corr_loss",
    "g_loss_total",
    "lambda_gp",
    "lambda_feature",
    "lambda_corr",
    "use_class_weights",
    "epoch_time_sec",
    "peak_cuda_memory_mb",
    # ---- Section 3: skipped-vs-applied distinction. The columns above
    # (gp_row/gp_feature/gp_total/g_corr_loss) average over EVERY D-step /
    # G-step in the epoch, counting a skipped (gp_every/corr_every) step as
    # a literal 0.0 -- this is kept for backward compatibility with Rev 2
    # log consumers. The columns below average ONLY over steps where the
    # term was actually evaluated, and separately report how many such
    # evaluations occurred, so "skipped" and "applied-and-genuinely-zero"
    # can never be confused. ----
    "gp_row_applied_mean",
    "gp_feature_applied_mean",
    "gp_total_applied_mean",
    "n_gp_evaluations",
    "corr_loss_applied_mean",
    "n_corr_evaluations",
    # ---- Section 20: adversarial-loss-vs-covariance-loss balance ----
    "corr_weighted_loss",      # mean(lambda_corr * L_corr), evaluated steps only
    "corr_to_adv_ratio",       # corr_weighted_loss / (mean|g_adv_loss| + eps), evaluated steps only
    # ---- Section 21: generator-side row/feature/combined adversarial
    # decomposition (diagnostic only -- NOT a separate loss term; g_adv_loss
    # above already equals g_combined_adv) ----
    "g_row_adv",
    "g_feature_adv",
    "g_combined_adv",
    # ---- Section 22: OPTIONAL diagnostic-only gradient-cosine-similarity
    # between the generator gradients from the row-adversarial term alone
    # vs the feature-adversarial term alone. NaN / 0 evaluations whenever
    # measure_grad_cosine=False (the default) or use_feature_head=False. ----
    "grad_cosine_row_feature",
    "n_grad_cosine_evaluations",
    # ---- Section 1/2: GLOBAL (never epoch-reset) scheduling counters, and
    # how many D-steps/G-steps this particular epoch actually ran (useful
    # to confirm n_d_steps_this_epoch == n_critic * n_g_steps_this_epoch) ----
    "global_critic_step",
    "global_generator_step",
    "n_d_steps_this_epoch",
    "n_g_steps_this_epoch",
]


class TrainingHistoryLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(TRAINING_LOG_COLUMNS)

    def log(self, row: Dict) -> None:
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in TRAINING_LOG_COLUMNS])


# ======================================================================
# Trainer
# ======================================================================
class CDGANTrainer:
    # Only these keys are checked for NaN/Inf (Section 3): the "_applied_mean"
    # /ratio/cosine columns are legitimately NaN whenever an epoch happens to
    # contain zero evaluated steps of that kind (e.g. gp_every > steps this
    # epoch, or measure_grad_cosine=False), and that is not a training
    # failure.
    _CORE_LOSS_KEYS_FOR_NAN_CHECK = {
        "d_loss_total", "d_row_wasserstein", "d_feature_wasserstein",
        "gp_row", "gp_feature", "gp_total",
        "g_adv_loss", "g_corr_loss", "g_loss_total",
        "g_row_adv", "g_feature_adv", "g_combined_adv",
    }

    def __init__(
        self,
        config: CDGANConfig,
        generator: Generator,
        discriminator: DualHeadDiscriminator,
        categorical_slices: List[Tuple[int, int]],
        out_dir: str,
    ):
        self.config = apply_ablation_flags(config)
        self.device = resolve_device(config.device)
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.categorical_slices = categorical_slices
        self.out_dir = out_dir

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=config.learning_rate_g, betas=config.adam_betas
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=config.learning_rate_d, betas=config.adam_betas
        )

        self.history_logger = TrainingHistoryLogger(os.path.join(out_dir, "training_history.csv"))
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # ---- Section 1/2: GLOBAL scheduling counters. NEVER reset per
        # epoch. gp_every gates on global_critic_step; corr_every and
        # grad_cosine_every gate on global_generator_step. Restored EXACTLY
        # on resume (see maybe_resume()), never reconstructed. ----
        self.global_critic_step = 0
        self.global_generator_step = 0
        self.start_epoch = 0
        self.class_weights: Optional[Dict[int, float]] = None

        print("=" * 70)
        print_gpu_info(self.device)
        pc = count_parameters(self.generator, self.discriminator)
        print(f"Generator parameters   : {pc.generator:,}")
        print(f"Row-head parameters    : {pc.row_head:,}")
        print(f"Feature-head parameters: {pc.feature_head:,}")
        print(f"Total parameters       : {pc.total:,}")
        print(f"config_name = {config.config_name} | use_feature_head={config.use_feature_head} "
              f"| use_gp={config.use_gp} | use_corr={config.use_corr}")
        print(f"lambda_gp={config.lambda_gp}  lambda_feature={config.lambda_feature}  "
              f"lambda_corr={config.lambda_corr}  use_class_weights={config.use_class_weights}")
        print(f"n_critic={config.n_critic}  gp_every={config.gp_every} (on global_critic_step)  "
              f"corr_every={config.corr_every} (on global_generator_step)  "
              f"measure_grad_cosine={config.measure_grad_cosine}")
        print(f"training_loop_version={config.training_loop_version}")
        print("=" * 70)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config.as_dict(), f, indent=2)

    # ------------------------------------------------------------------
    def set_class_weights(self, y_train: np.ndarray) -> None:
        if self.config.use_class_weights:
            self.class_weights = compute_class_weights(y_train)
            print(f"[class weighting ENABLED] weights={self.class_weights}")
        else:
            self.class_weights = None
            print("[class weighting DISABLED] (default; manuscript formulation is unweighted)")

    def _weighted_mean(self, values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Only used when use_class_weights=True. Applies inverse-frequency
        per-sample weights to the GENERATOR adversarial loss ONLY, matching
        where the previous implementation applied class weighting."""
        if self.class_weights is None:
            return values.mean()
        w = torch.ones_like(values)
        for cls, weight in self.class_weights.items():
            w = torch.where(labels.view(-1) == cls, torch.full_like(w, weight), w)
        return (values.view(-1) * w).sum() / w.sum()

    # ------------------------------------------------------------------
    def maybe_resume(self) -> None:
        if not self.config.resume:
            return
        latest = find_latest_checkpoint(self.ckpt_dir)
        if latest is None:
            print("[resume] no compatible checkpoint found; starting from epoch 0.")
            return
        epoch, gcs, ggs = load_checkpoint(
            latest, self.generator, self.discriminator, self.opt_g, self.opt_d, self.config, self.device
        )
        self.start_epoch = epoch + 1
        self.global_critic_step = gcs
        self.global_generator_step = ggs
        print(
            f"[resume] loaded {latest}; resuming from epoch {self.start_epoch}, "
            f"global_critic_step={self.global_critic_step}, global_generator_step={self.global_generator_step}"
        )

    # ------------------------------------------------------------------
    def _discriminator_step(self, x_real: torch.Tensor, cond: torch.Tensor, compute_gp: bool) -> Dict:
        """ONE discriminator update on a fresh z (Section 1). ``compute_gp``
        is decided by the caller from the GLOBAL critic-step counter."""
        B = x_real.size(0)
        z = self.generator.sample_latent(B, self.device)
        with torch.no_grad():
            x_fake = self.generator(z, cond)

        row_out, feat_out, combined_out = self.discriminator(x_real, cond, self.config.lambda_feature)
        row_fake, feat_fake, combined_fake = self.discriminator(x_fake, cond, self.config.lambda_feature)

        d_row_wasserstein = row_fake.mean() - row_out.mean()
        d_feature_wasserstein = (
            (feat_fake.mean() - feat_out.mean()) if self.discriminator.use_feature_head
            else torch.zeros((), device=self.device)
        )
        d_adv = combined_fake.mean() - combined_out.mean()

        gp_applied = bool(self.config.use_gp and compute_gp)
        gp_row = torch.zeros((), device=self.device)
        gp_feature = torch.zeros((), device=self.device)
        gp_total = torch.zeros((), device=self.device)
        if gp_applied:
            gp_row, gp_feature, gp_total = dual_gradient_penalty(
                self.discriminator, x_real, x_fake.detach(), cond
            )

        d_loss = d_adv + self.config.lambda_gp * gp_total

        self.opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        self.opt_d.step()

        return {
            "d_loss_total": float(d_loss.detach().cpu()),
            "d_row_wasserstein": float(d_row_wasserstein.detach().cpu()),
            "d_feature_wasserstein": float(d_feature_wasserstein.detach().cpu()),
            "gp_row": float(gp_row.detach().cpu()),
            "gp_feature": float(gp_feature.detach().cpu()),
            "gp_total": float(gp_total.detach().cpu()),
            "gp_computed": gp_applied,
        }

    def _generator_step(
        self,
        x_real: torch.Tensor,
        cond: torch.Tensor,
        compute_corr: bool,
        compute_grad_cosine: bool = False,
    ) -> Dict:
        """ONE generator update on a fresh z (Section 1). ``compute_corr``
        is decided by the caller from the GLOBAL generator-step counter."""
        B = x_real.size(0)
        z = self.generator.sample_latent(B, self.device)
        x_fake = self.generator(z, cond)

        row_fake, feat_fake, combined_fake = self.discriminator(x_fake, cond, self.config.lambda_feature)

        # ---- Section 21: diagnostic-only row/feature/combined decomposition
        # of the generator's adversarial signal. Logged only -- g_adv (the
        # actual loss term) and g_combined_adv below are the SAME quantity;
        # g_row_adv/g_feature_adv are never separately weighted into the loss. ----
        g_row_adv = -row_fake.mean()
        g_feature_adv = (
            (-feat_fake.mean()) if self.discriminator.use_feature_head
            else torch.zeros((), device=self.device)
        )

        if self.config.use_class_weights:
            g_adv = -self._weighted_mean(combined_fake, cond)
        else:
            g_adv = -combined_fake.mean()
        g_combined_adv = g_adv

        # ---- Section 22: OPTIONAL diagnostic-only gradient-cosine-similarity
        # between the generator gradients induced by the row-adversarial term
        # alone vs the feature-adversarial term alone. Computed with
        # torch.autograd.grad, which does NOT populate .grad and therefore
        # cannot influence opt_g.step() below. No PCGrad, no gradient
        # surgery -- purely observational. ----
        grad_cosine = float("nan")
        grad_cosine_computed = bool(compute_grad_cosine and self.discriminator.use_feature_head)
        if grad_cosine_computed:
            row_only = -row_fake.mean()
            feat_only = -feat_fake.mean()
            params = [p for p in self.generator.parameters() if p.requires_grad]
            row_grads = torch.autograd.grad(row_only, params, retain_graph=True, allow_unused=True)
            feat_grads = torch.autograd.grad(feat_only, params, retain_graph=True, allow_unused=True)
            row_parts = [g.reshape(-1) for g in row_grads if g is not None]
            feat_parts = [g.reshape(-1) for g in feat_grads if g is not None]
            if row_parts and feat_parts:
                row_flat = torch.cat(row_parts)
                feat_flat = torch.cat(feat_parts)
                denom = (row_flat.norm() * feat_flat.norm()).clamp_min(1e-12)
                grad_cosine = float((torch.dot(row_flat, feat_flat) / denom).detach().cpu())

        corr_applied = bool(self.config.use_corr and compute_corr)
        g_corr = torch.zeros((), device=self.device)
        if corr_applied:
            if self.config.dependency_loss == "pearson":
                g_corr = pearson_correlation_loss(x_real, x_fake)
            else:
                g_corr = covariance_correlation_loss(x_real, x_fake)

        g_loss = g_adv + self.config.lambda_corr * g_corr

        self.opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        self.opt_g.step()

        eps = 1e-8
        corr_weighted_loss = float("nan")
        corr_to_adv_ratio = float("nan")
        if corr_applied:
            corr_weighted_loss = float((self.config.lambda_corr * g_corr).detach().cpu())
            corr_to_adv_ratio = corr_weighted_loss / (abs(float(g_adv.detach().cpu())) + eps)

        return {
            "g_adv_loss": float(g_adv.detach().cpu()),
            "g_corr_loss": float(g_corr.detach().cpu()),
            "g_loss_total": float(g_loss.detach().cpu()),
            "corr_computed": corr_applied,
            "corr_weighted_loss": corr_weighted_loss,
            "corr_to_adv_ratio": corr_to_adv_ratio,
            "g_row_adv": float(g_row_adv.detach().cpu()),
            "g_feature_adv": float(g_feature_adv.detach().cpu()),
            "g_combined_adv": float(g_combined_adv.detach().cpu()),
            "grad_cosine_row_feature": grad_cosine,
            "grad_cosine_computed": grad_cosine_computed,
        }

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_epochs: Optional[List[int]] = None,
    ) -> str:
        """Runs self.config.epochs epochs (or resumes and finishes them).
        Returns the path to training_history.csv.

        Section 1: for EVERY minibatch, this runs exactly ``n_critic``
        discriminator updates (fresh z each) and THEN exactly ONE generator
        update (fresh z) -- never an interleaved global-counter alternation
        across successive batches. ``gp_every``/``corr_every`` gate on the
        GLOBAL, never-epoch-reset ``global_critic_step``/
        ``global_generator_step`` counters.
        """
        self.set_class_weights(y_train)
        self.maybe_resume()

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).view(-1, 1)
        n = X_t.size(0)
        batch_size = self.config.batch_size
        n_critic = self.config.n_critic

        for epoch in range(self.start_epoch, self.config.epochs):
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            perm = torch.randperm(n, device=self.device)
            n_batches = max(1, n // batch_size)

            epoch_d_metrics: List[Dict] = []
            epoch_g_metrics: List[Dict] = []

            for b in range(n_batches):
                idx = perm[b * batch_size: (b + 1) * batch_size]
                if idx.numel() < 2:
                    continue
                x_real = X_t[idx]
                cond = y_t[idx]

                # ---- Section 1: n_critic D-updates (fresh z each) ----
                for _ in range(n_critic):
                    compute_gp = (self.global_critic_step % self.config.gp_every == 0)
                    d_metrics = self._discriminator_step(x_real, cond, compute_gp=compute_gp)
                    epoch_d_metrics.append(d_metrics)
                    self.global_critic_step += 1

                # ---- Section 1: THEN exactly ONE G-update (fresh z) ----
                compute_corr = (self.global_generator_step % self.config.corr_every == 0)
                compute_grad_cosine = (
                    self.config.measure_grad_cosine
                    and (self.global_generator_step % self.config.grad_cosine_every == 0)
                )
                g_metrics = self._generator_step(
                    x_real, cond, compute_corr=compute_corr, compute_grad_cosine=compute_grad_cosine
                )
                epoch_g_metrics.append(g_metrics)
                self.global_generator_step += 1

            epoch_time = time.time() - t0
            peak_mem_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if torch.cuda.is_available() else 0.0
            )

            def _avg(dicts, key):
                vals = [d[key] for d in dicts if key in d]
                return float(np.mean(vals)) if vals else 0.0

            def _avg_where(dicts, key, flag_key):
                vals = [d[key] for d in dicts if d.get(flag_key)]
                return float(np.mean(vals)) if vals else float("nan")

            def _count_where(dicts, flag_key):
                return int(sum(1 for d in dicts if d.get(flag_key)))

            n_gp_evaluations = _count_where(epoch_d_metrics, "gp_computed")
            n_corr_evaluations = _count_where(epoch_g_metrics, "corr_computed")
            n_grad_cosine_evaluations = _count_where(epoch_g_metrics, "grad_cosine_computed")

            row = {
                "epoch": epoch,
                "d_loss_total": _avg(epoch_d_metrics, "d_loss_total"),
                "d_row_wasserstein": _avg(epoch_d_metrics, "d_row_wasserstein"),
                "d_feature_wasserstein": _avg(epoch_d_metrics, "d_feature_wasserstein"),
                "gp_row": _avg(epoch_d_metrics, "gp_row"),
                "gp_feature": _avg(epoch_d_metrics, "gp_feature"),
                "gp_total": _avg(epoch_d_metrics, "gp_total"),
                "g_adv_loss": _avg(epoch_g_metrics, "g_adv_loss"),
                "g_corr_loss": _avg(epoch_g_metrics, "g_corr_loss"),
                "g_loss_total": _avg(epoch_g_metrics, "g_loss_total"),
                "lambda_gp": self.config.lambda_gp,
                "lambda_feature": self.config.lambda_feature,
                "lambda_corr": self.config.lambda_corr,
                "use_class_weights": self.config.use_class_weights,
                "epoch_time_sec": epoch_time,
                "peak_cuda_memory_mb": peak_mem_mb,
                "gp_row_applied_mean": _avg_where(epoch_d_metrics, "gp_row", "gp_computed"),
                "gp_feature_applied_mean": _avg_where(epoch_d_metrics, "gp_feature", "gp_computed"),
                "gp_total_applied_mean": _avg_where(epoch_d_metrics, "gp_total", "gp_computed"),
                "n_gp_evaluations": n_gp_evaluations,
                "corr_loss_applied_mean": _avg_where(epoch_g_metrics, "g_corr_loss", "corr_computed"),
                "n_corr_evaluations": n_corr_evaluations,
                "corr_weighted_loss": _avg_where(epoch_g_metrics, "corr_weighted_loss", "corr_computed"),
                "corr_to_adv_ratio": _avg_where(epoch_g_metrics, "corr_to_adv_ratio", "corr_computed"),
                "g_row_adv": _avg(epoch_g_metrics, "g_row_adv"),
                "g_feature_adv": _avg(epoch_g_metrics, "g_feature_adv"),
                "g_combined_adv": _avg(epoch_g_metrics, "g_combined_adv"),
                "grad_cosine_row_feature": _avg_where(epoch_g_metrics, "grad_cosine_row_feature", "grad_cosine_computed"),
                "n_grad_cosine_evaluations": n_grad_cosine_evaluations,
                "global_critic_step": self.global_critic_step,
                "global_generator_step": self.global_generator_step,
                "n_d_steps_this_epoch": len(epoch_d_metrics),
                "n_g_steps_this_epoch": len(epoch_g_metrics),
            }
            self.history_logger.log(row)

            if self._nan_check(row):
                raise FloatingPointError(f"NaN/Inf detected in training losses at epoch {epoch}: {row}")

            if epoch % max(1, self.config.epochs // 20 or 1) == 0 or epoch == self.config.epochs - 1:
                print(
                    f"[epoch {epoch:5d}] D={row['d_loss_total']:.4f} "
                    f"(row_W={row['d_row_wasserstein']:.4f} feat_W={row['d_feature_wasserstein']:.4f} "
                    f"GP_r={row['gp_row']:.4f} GP_f={row['gp_feature']:.4f} n_gp_eval={row['n_gp_evaluations']}) | "
                    f"G={row['g_loss_total']:.4f} (adv={row['g_adv_loss']:.4f} corr={row['g_corr_loss']:.4f} "
                    f"n_corr_eval={row['n_corr_evaluations']}) | "
                    f"gcs={row['global_critic_step']} ggs={row['global_generator_step']} | {epoch_time:.2f}s"
                )

            do_checkpoint = (
                (epoch + 1) % self.config.checkpoint_every == 0
                or epoch == self.config.epochs - 1
                or (checkpoint_epochs is not None and epoch in checkpoint_epochs)
            )
            if do_checkpoint:
                ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_epoch{epoch:04d}.pt")
                save_checkpoint(
                    ckpt_path, epoch, self.global_critic_step, self.global_generator_step,
                    self.generator, self.discriminator,
                    self.opt_g, self.opt_d, self.config, metrics=row,
                )

        return self.history_logger.csv_path

    @classmethod
    def _nan_check(cls, row: Dict) -> bool:
        for k in cls._CORE_LOSS_KEYS_FOR_NAN_CHECK:
            v = row.get(k)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return True
        return False

    # ------------------------------------------------------------------
    def generate_synthetic(
        self,
        n_samples: int,
        class_proportions: Dict[int, float],
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Section 10: generate conditional on training-set class labels,
        preserving training-set class proportions.

        Returns (X_syn_raw_encoded, X_syn_projected_encoded, y_syn).
        """
        if seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(seed)
        else:
            gen = None

        classes = sorted(class_proportions.keys())
        counts = np.array([class_proportions[c] for c in classes]) * n_samples
        counts = np.round(counts).astype(int)
        counts[-1] += n_samples - counts.sum()  # fix rounding drift

        y_syn = np.concatenate([np.full(c, cls, dtype=np.float32) for cls, c in zip(classes, counts)])
        rng = np.random.default_rng(seed)
        rng.shuffle(y_syn)

        self.generator.eval()
        with torch.no_grad():
            y_t = torch.tensor(y_syn, dtype=torch.float32, device=self.device).view(-1, 1)
            if gen is not None:
                z = torch.randn(y_t.size(0), self.generator.latent_dim, device=self.device, generator=gen)
            else:
                z = torch.randn(y_t.size(0), self.generator.latent_dim, device=self.device)
            x_syn_raw = self.generator(z, y_t)
            x_syn_proj = project_categorical_blocks_torch(x_syn_raw, self.categorical_slices)
        self.generator.train()

        return (
            x_syn_raw.cpu().numpy(),
            x_syn_proj.cpu().numpy(),
            y_syn,
        )
