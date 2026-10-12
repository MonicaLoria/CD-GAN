"""
model.py -- CD-GAN generator and TRUE dual-head discriminator with an
ORIGINAL-COLUMN feature critic.

Revision history
-----------------
* Rev 1 (previous pass): fixed the reviewer-flagged bug where "feature"
  and "row" were both full-record critics, by giving the feature head a
  SHARED MLP applied to every ENCODED dimension independently.
* Rev 2 (THIS file): Rev 1 still let a high-cardinality one-hot-encoded
  categorical column dominate the feature-level average, because it
  counted once per ENCODED dimension rather than once per ORIGINAL
  column. This revision replaces the encoded-dimension feature head with
  an ORIGINAL-COLUMN feature head: one score per original predictor
  column (numerical columns: width 1; categorical columns: their
  COMPLETE one-hot block treated as a single unit), so a 20-category
  column contributes exactly ONE term to the aggregate, same as a
  numerical column.

FEATURE-HEAD PARAMETER SHARING -- explicitly answered per your question:
    Feature-head parameters are COLUMN-SPECIFIC, not shared.
    ``DualHeadDiscriminator.feature_heads`` is an ``nn.ModuleList`` with
    ONE small critic sub-network per ORIGINAL column (``D_f,j`` for
    j = 1..P), because different original columns have different input
    widths (a numerical column is width 1, a categorical column's block
    is as wide as its category count) and therefore cannot literally
    share one set of weights the way the previous encoded-dimension
    design could. This mirrors the "one head per original-column group"
    mechanism already validated in your earlier
    ``cd_gan_step3_validatedclaude.py`` (``feature_level='original_column'``),
    reused here for the grouping/ModuleList structure -- but WITHOUT that
    file's learned per-column embedding (``nn.Embedding``), which Section
    27 of your instructions rules out as unnecessary added complexity for
    this manuscript-strict revision. Each ``D_f,j`` here takes exactly
    ``[x_i^(j), c_i]`` (the column's own encoded block concatenated with
    the condition) -- nothing else.

Two heads, two different inductive roles
-----------------------------------------
* ROW HEAD   D_r(x_i, c_i):     sees the WHOLE encoded record [x_i, c_i] in R^(D+1).
* FEATURE HEAD D_f,j(x_i^(j), c_i): one column-specific critic per ORIGINAL
  column j = 1..P, each applied to that column's own encoded block
  (width 1 for numerical, width = #categories for categorical) plus the
  condition. Aggregated by an unweighted mean over the P ORIGINAL columns
  (NOT over the D encoded dimensions):

      D_feat(x_i, c_i) = (1/P) * sum_{j=1}^{P} D_f,j(x_i^(j), c_i)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ======================================================================
# 1. Generator :  G(z, c) = x_tilde
# ======================================================================
class Generator(nn.Module):
    """Conditional MLP generator (UNCHANGED from the previous revision).

        h1      = ReLU( BN( W1 [z ; c] + b1 ) )   in R^hidden_dim
        h2      = ReLU( BN( W2 h1       + b2 ) )   in R^(2*hidden_dim)
        x_tilde =            W3 h2       + b3      in R^output_dim   (LINEAR)
        G(z, c) = x_tilde

    Categorical one-hot blocks are NOT specially handled inside the
    generator. Projection of a generated block onto a valid one-hot
    vector happens OUTSIDE the model
    (see ``preprocessing.project_categorical_blocks``).
    """

    def __init__(self, latent_dim: int, cond_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.cond_dim = int(cond_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(self.latent_dim + self.cond_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.BatchNorm1d(self.hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim * 2, self.output_dim),  # linear output layer
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z: [B, latent_dim], cond: [B, cond_dim] -> x_tilde: [B, output_dim]."""
        return self.net(torch.cat([z, cond], dim=1))

    def sample_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)


# ======================================================================
# 2. Row-level critic :  D_r(x_i, c_i)
# ======================================================================
class RowCriticMLP(nn.Module):
    """The row-level critic. Sees the COMPLETE encoded record + condition.
    UNCHANGED by this revision -- see Section 2 of your instructions."""

    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_and_cond: torch.Tensor) -> torch.Tensor:
        """x_and_cond: [B, D + cond_dim] -> [B, 1]."""
        return self.net(x_and_cond)


# ======================================================================
# 3. Original-column feature critic :  D_f,j(x_i^(j), c_i)
# ======================================================================
def _make_column_critic(group_width: int, cond_dim: int, hidden_dim: int) -> nn.Sequential:
    """One small critic sub-network for a single ORIGINAL column's encoded
    block (width = 1 for numerical, width = #categories for categorical)."""
    return nn.Sequential(
        nn.Linear(group_width + cond_dim, hidden_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(hidden_dim, 1),
    )


# ======================================================================
# 4. Dual-head discriminator wrapper
# ======================================================================
class DualHeadDiscriminator(nn.Module):
    """TRUE dual-head critic with two genuinely different input granularities.

    use_feature_head=True  (default, "Full CD-GAN" / manuscript model):
        - ROW head:      Dr(x, c)    on [x, c]                          in R^(D+1)
        - FEATURE head:  Dfeat(x, c) on ONE ORIGINAL-COLUMN score per
          j = 1..P (P = number of ORIGINAL predictor columns, NOT the
          encoded dimension D), unweighted mean over the P scores.

    use_feature_head=False ("w/o Dual" ablation):
        - ONLY the row critic exists. No feature-head parameters are
          instantiated at all (not merely zeroed out).

    Parameters
    ----------
    feature_group_slices : list of (start, end) index ranges into the
        encoded [B, D] vector, one per ORIGINAL predictor column, in
        encoded-column order. MUST come from the fitted preprocessor
        (``preprocessing.FittedPreprocessor.original_feature_groups``) --
        this class never guesses or re-derives them.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        use_feature_head: bool = True,
        feature_group_slices: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)     # D: encoded feature dimension
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_feature_head = bool(use_feature_head)

        # ---- ROW HEAD: complete record + condition -> [B, D + cond_dim] ----
        self.row_head = RowCriticMLP(self.input_dim, self.cond_dim, self.hidden_dim)

        # ---- FEATURE HEAD: one column-specific critic per ORIGINAL column ----
        self.feature_group_slices: List[Tuple[int, int]] = []
        self.feature_heads: Optional[nn.ModuleList] = None
        if self.use_feature_head:
            if not feature_group_slices:
                raise ValueError(
                    "feature_group_slices is required when use_feature_head=True. "
                    "Pass FittedPreprocessor.original_feature_groups (start, end pairs) "
                    "-- this class does not guess column boundaries."
                )
            normalized = []
            covered = 0
            for start, end in feature_group_slices:
                start, end = int(start), int(end)
                if not (0 <= start < end <= self.input_dim):
                    raise ValueError(f"Invalid feature group slice {(start, end)} for input_dim={self.input_dim}")
                normalized.append((start, end))
                covered += end - start
            if covered != self.input_dim:
                raise ValueError(
                    f"feature_group_slices cover {covered} encoded dims but input_dim={self.input_dim}. "
                    "Every encoded dimension must belong to exactly one original-column group."
                )
            self.feature_group_slices = normalized
            self.feature_heads = nn.ModuleList(
                [_make_column_critic(end - start, self.cond_dim, self.hidden_dim) for start, end in normalized]
            )

    @property
    def n_original_feature_groups(self) -> int:
        """P: number of ORIGINAL predictor columns the feature head scores."""
        return len(self.feature_group_slices)

    # ------------------------------------------------------------------
    def feature_scores(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return the UN-aggregated per-ORIGINAL-COLUMN critic scores.

        For every original column j = 1..P (P = ``n_original_feature_groups``,
        independent of how many one-hot dimensions that column expands to):

            x^(j)   = x[:, start_j:end_j]              (that column's encoded block)
            score_j = D_f,j( [x^(j), c] )               column-specific head j

        Returns
        -------
        torch.Tensor of shape [B, P] -- ONE score per ORIGINAL column, not
        one per encoded dimension D. A 20-category column contributes
        exactly one entry to this tensor, identical in kind to a
        numerical column's single entry.
        """
        if not self.use_feature_head:
            raise RuntimeError(
                "feature_scores() called but use_feature_head=False "
                "(this discriminator has no feature head, as in the 'w/o Dual' ablation)."
            )
        per_column_scores = []
        for (start, end), head in zip(self.feature_group_slices, self.feature_heads):
            block = x[:, start:end]                          # [B, width_j]  (width_j = 1 or #categories)
            block_and_cond = torch.cat([block, cond], dim=1)  # [B, width_j + cond_dim]
            per_column_scores.append(head(block_and_cond))    # [B, 1]
        return torch.cat(per_column_scores, dim=1)  # [B, P]

    def aggregated_feature_score(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """D_feat(x_i, c_i) = (1/P) * sum_j D_f,j(x_i^(j), c_i)  ->  [B, 1].

        Equal (unweighted) averaging across the P ORIGINAL predictor
        columns -- explicitly NOT over the D encoded dimensions. This is
        what prevents a high-cardinality one-hot categorical column from
        dominating the feature-level signal merely by occupying more
        encoded dimensions.
        """
        if not self.use_feature_head:
            B = x.size(0)
            return torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        per_column_scores = self.feature_scores(x, cond)            # [B, P]
        return per_column_scores.mean(dim=1, keepdim=True)          # [B, 1]  <- mean over P original columns

    def row_score(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """D_r(x_i, c_i) on the complete record [x_i, c_i] -> [B, 1]. Unchanged."""
        return self.row_head(torch.cat([x, cond], dim=1))

    def combined_score(self, x: torch.Tensor, cond: torch.Tensor, lambda_feature: float = 1.0) -> torch.Tensor:
        """S(x, c) = D_r(x, c) + lambda_f * D_feat(x, c)  ->  [B, 1]."""
        row_out = self.row_score(x, cond)
        feat_out = self.aggregated_feature_score(x, cond)
        return row_out + float(lambda_feature) * feat_out

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, lambda_feature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (row_out, feat_out, combined_out), each [B,1]. Row and
        feature outputs are each computed exactly ONCE -- no path here can
        return the same tensor under both names."""
        row_out = self.row_score(x, cond)
        feat_out = self.aggregated_feature_score(x, cond)
        combined = row_out + float(lambda_feature) * feat_out
        return row_out, feat_out, combined


# ======================================================================
# 5. Gradient penalties -- SEPARATE for the two heads
# ======================================================================
def _interpolate(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """x_hat = alpha * x + (1 - alpha) * x_tilde, alpha ~ U(0,1), per-row."""
    B = real.size(0)
    alpha = torch.rand(B, 1, device=real.device, dtype=real.dtype)
    x_hat = alpha * real + (1.0 - alpha) * fake
    x_hat.requires_grad_(True)
    return x_hat


def row_gradient_penalty(
    disc: DualHeadDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
    x_hat: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GP_r = E[(||grad_xhat Dr(x_hat, c)||_2 - 1)^2].

    Returns (gp_row, x_hat) so the SAME interpolated sample can be reused
    for ``feature_gradient_penalty``.
    """
    if x_hat is None:
        x_hat = _interpolate(real, fake)
    row_out = disc.row_score(x_hat, cond)  # [B, 1]
    grad = torch.autograd.grad(
        outputs=row_out, inputs=x_hat, grad_outputs=torch.ones_like(row_out),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = grad.norm(2, dim=1)  # [B]
    gp_row = ((grad_norm - 1.0) ** 2).mean()
    return gp_row, x_hat


def feature_gradient_penalty(
    disc: DualHeadDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
    x_hat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GP_f = E[(||grad_xhat D_feat(x_hat, c)||_2 - 1)^2], where D_feat is
    the ORIGINAL-COLUMN AGGREGATED feature score (this function is
    unchanged in form from the previous revision -- what changed is what
    ``aggregated_feature_score`` computes underneath it).

    Returns an exact zero (not merely a near-zero tensor) when
    use_feature_head=False -- the 'w/o Dual' ablation."""
    if not disc.use_feature_head:
        return torch.zeros((), device=real.device, dtype=real.dtype)
    if x_hat is None:
        x_hat = _interpolate(real, fake)
    feat_out = disc.aggregated_feature_score(x_hat, cond)  # [B, 1]
    grad = torch.autograd.grad(
        outputs=feat_out, inputs=x_hat, grad_outputs=torch.ones_like(feat_out),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grad_norm = grad.norm(2, dim=1)  # [B]
    gp_feature = ((grad_norm - 1.0) ** 2).mean()
    return gp_feature


def dual_gradient_penalty(
    disc: DualHeadDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    cond: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GP_row and GP_feature at a SHARED interpolated sample x_hat.
    L_gp = GP_r + GP_f. When use_feature_head=False, GP_feature is exactly
    zero and GP_total == GP_row."""
    gp_row, x_hat = row_gradient_penalty(disc, real, fake, cond)
    gp_feature = feature_gradient_penalty(disc, real, fake, cond, x_hat=x_hat)
    gp_total = gp_row + gp_feature
    return gp_row, gp_feature, gp_total


# ======================================================================
# 6. Covariance-based correlation regularizer (UNCHANGED)
# ======================================================================
def covariance_correlation_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L_corr = || Cov(X_real) - Cov(X_fake) ||_F, computed in encoded space.

    Second-order, linear dependency regularizer only. No HSIC, mutual
    information, distance correlation, or PCGrad have been added (Section
    27) -- this function is byte-for-byte the same as the previous
    revision.
    """
    if real.size(0) < 2 or fake.size(0) < 2:
        return fake.sum() * 0.0
    real_cov = torch.cov(real.T)
    fake_cov = torch.cov(fake.T)
    return torch.norm(real_cov - fake_cov, p="fro")


def pearson_correlation_loss(real: torch.Tensor, fake: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Rev 4 (reviewer point 10): L_pearson = || Corr(X_real) - Corr(X_fake) ||_F,
    computed in the same encoded space as the covariance loss. Unlike the
    covariance loss it is invariant to per-column scale, so wide one-hot
    blocks and standardized numerical columns contribute on the same
    footing. Columns that are constant in the REAL minibatch (zero
    variance, e.g. an absent category) are excluded from both matrices,
    because their correlation is undefined. Still a second-order, linear
    dependency measure."""
    if real.size(0) < 3 or fake.size(0) < 3:
        return fake.sum() * 0.0
    keep = real.std(dim=0) > eps
    if int(keep.sum()) < 2:
        return fake.sum() * 0.0
    r, f = real[:, keep], fake[:, keep]

    def _corr(x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(dim=0, keepdim=True)
        cov = xc.T @ xc / (x.size(0) - 1)
        sd = torch.sqrt(torch.diag(cov).clamp_min(eps))
        return cov / (sd[:, None] * sd[None, :])

    return torch.norm(_corr(r) - _corr(f), p="fro")


# ======================================================================
# 7. Parameter counting
# ======================================================================
@dataclass
class ParameterCounts:
    generator: int
    row_head: int
    feature_head: int
    total: int

    def as_dict(self) -> dict:
        return {
            "generator_parameters": self.generator,
            "row_head_parameters": self.row_head,
            "feature_head_parameters": self.feature_head,
            "total_parameters": self.total,
        }


def _count_params(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_parameters(generator: Generator, disc: DualHeadDiscriminator) -> ParameterCounts:
    g = _count_params(generator)
    r = _count_params(disc.row_head)
    f = _count_params(disc.feature_heads)
    return ParameterCounts(generator=g, row_head=r, feature_head=f, total=g + r + f)


# ======================================================================
# 8. Factory helper
# ======================================================================
def build_models(
    latent_dim: int,
    hidden_dim: int,
    cond_dim: int,
    encoded_dim: int,
    feature_group_slices: Optional[List[Tuple[int, int]]] = None,
    use_feature_head: bool = True,
) -> Tuple[Generator, DualHeadDiscriminator]:
    """Build a (Generator, DualHeadDiscriminator) pair with matching dims.

    ``feature_group_slices`` MUST be supplied (from
    ``FittedPreprocessor.original_feature_groups``) whenever
    ``use_feature_head=True`` -- see Section 9 of your instructions:
    "the model must NOT guess these slices."
    """
    generator = Generator(latent_dim=latent_dim, cond_dim=cond_dim, output_dim=encoded_dim, hidden_dim=hidden_dim)
    discriminator = DualHeadDiscriminator(
        input_dim=encoded_dim, cond_dim=cond_dim, hidden_dim=hidden_dim,
        use_feature_head=use_feature_head, feature_group_slices=feature_group_slices,
    )
    return generator, discriminator
