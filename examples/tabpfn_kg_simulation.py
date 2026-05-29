"""End-to-End Simulation of Knowledge Graph-Aware TabPFN.

This prototype validates the integration of structural Knowledge Graph (KG) prior
with a pre-trained tabular foundation model (TabPFN). It addresses target-leakage 
issues by employing Stratified K-Fold Cross-Validation for feature extraction.

It compares 6 settings:
1. Pure TabPFN
2. BaselineHead (Standard MLP on TabPFN tokens)
3. LargeBaselineHead (Parameter-matched MLP baseline)
4. KGHead_real (Cross-attention with valid KG structure)
5. KGHead_permuted (Cross-attention with shuffled KG, testing structural utility)
6. KGHead_random (Cross-attention with random Gaussian KG embeddings)
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig


@dataclasses.dataclass(frozen=True)
class KGSimulationData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    kg_embeddings: torch.Tensor
    true_feature_weights: torch.Tensor


@dataclasses.dataclass(frozen=True)
class RepresentationBatch:
    feature_tokens: torch.Tensor
    target_tokens: torch.Tensor
    feature_group_kg: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ExperimentResult:
    seed: int
    repr_extract_time: float
    tabpfn_accuracy: float
    baseline_head_accuracy: float
    large_baseline_head_accuracy: float
    kg_head_accuracy: float
    kg_permuted_head_accuracy: float
    kg_random_head_accuracy: float
    tabpfn_time: float
    baseline_head_time: float
    large_baseline_head_time: float
    kg_head_time: float
    kg_permuted_head_time: float
    kg_random_head_time: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def tensor_to_numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    """Convert tensors without relying on torch's NumPy C bridge."""

    return np.asarray(tensor.detach().cpu().tolist(), dtype=dtype)


def numpy_to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert ndarrays through Python lists for NumPy/PyTorch ABI robustness."""

    return torch.tensor(array.tolist(), dtype=dtype)


def generate_kg_smooth_classification_data(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    kg_dim: int,
    n_concepts: int,
    concept_noise: float,
    feature_noise: float,
    label_noise: float,
    seed: int,
) -> KGSimulationData:
    """Generate a dataset where the KG embeddings explain feature relevance.

    Features are assigned to latent concepts. KG embeddings cluster by concept,
    and the true linear classifier is a smooth function of those KG embeddings.
    The small-data regime makes estimating all feature weights directly hard,
    while the KG gives the head a compact hint about which feature groups should
    be attended to together.
    """

    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    concept_embeddings = normalize_rows(
        torch.randn(n_concepts, kg_dim, generator=torch_gen),
    )
    feature_concepts = torch.arange(n_features) % n_concepts
    feature_concepts = feature_concepts[torch.randperm(n_features, generator=torch_gen)]

    kg_embeddings = concept_embeddings[feature_concepts] + concept_noise * torch.randn(
        n_features,
        kg_dim,
        generator=torch_gen,
    )
    kg_embeddings = normalize_rows(kg_embeddings)

    active_concepts = torch.randperm(n_concepts, generator=torch_gen)[
        : max(2, n_concepts // 3)
    ]
    concept_direction = torch.randn(kg_dim, generator=torch_gen)
    concept_direction = normalize_rows(concept_direction.unsqueeze(0)).squeeze(0)

    concept_sign = torch.zeros(n_concepts)
    concept_sign[active_concepts] = torch.sign(
        torch.randn(active_concepts.numel(), generator=torch_gen),
    ).clamp(min=-1, max=1)
    concept_sign[concept_sign == 0] = 1

    smooth_weight = kg_embeddings @ concept_direction
    sparse_concept_weight = concept_sign[feature_concepts]
    true_feature_weights = smooth_weight + 1.25 * sparse_concept_weight
    true_feature_weights = true_feature_weights / true_feature_weights.norm().clamp_min(
        1e-8,
    )

    def make_X(n: int) -> torch.Tensor:
        concept_latents = torch.randn(n, n_concepts, generator=torch_gen)
        X = concept_latents[:, feature_concepts]
        X = X + feature_noise * torch.randn(n, n_features, generator=torch_gen)
        return X

    X_all = make_X(n_train + n_test)
    logits = math.sqrt(n_features) * (X_all @ true_feature_weights)
    logits += 0.4 * (X_all[:, feature_concepts == active_concepts[0]].mean(dim=1) ** 2)
    logits += label_noise * torch.randn(n_train + n_test, generator=torch_gen)

    threshold = torch.quantile(logits, 0.5)
    y_all = (logits > threshold).long()

    perm = rng.permutation(n_train + n_test)
    X_all_np = tensor_to_numpy(X_all, np.float32)[perm]
    y_all_np = tensor_to_numpy(y_all, np.int64)[perm]

    return KGSimulationData(
        X_train=X_all_np[:n_train],
        y_train=y_all_np[:n_train],
        X_test=X_all_np[n_train:],
        y_test=y_all_np[n_train:],
        kg_embeddings=kg_embeddings.float(),
        true_feature_weights=true_feature_weights.float(),
    )


def group_kg_embeddings(
    kg_embeddings: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Pool original feature KG embeddings to match TabPFN feature-token groups."""

    n_features, kg_dim = kg_embeddings.shape
    if n_groups == n_features:
        return kg_embeddings

    padded_features = int(math.ceil(n_features / n_groups) * n_groups)
    pad = padded_features - n_features
    if pad:
        kg_embeddings = torch.cat(
            [kg_embeddings, torch.zeros(pad, kg_dim, dtype=kg_embeddings.dtype)],
            dim=0,
        )

    features_per_group = padded_features // n_groups
    grouped = kg_embeddings.view(n_groups, features_per_group, kg_dim).mean(dim=1)
    return normalize_rows(grouped)


def _simple_tabpfn_inference_config() -> dict:
    """Keep feature order aligned with KG embeddings for the prototype."""

    return {
        "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none")],
        "FINGERPRINT_FEATURE": False,
        "FEATURE_SHIFT_METHOD": None,
        "CLASS_SHIFT_METHOD": None,
        "POLYNOMIAL_FEATURES": "no",
        "OUTLIER_REMOVAL_STD": None,
        "ENABLE_GPU_PREPROCESSING": False,
    }


def fit_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    model_path: str,
    device: str,
    random_state: int,
) -> TabPFNClassifier:
    clf = TabPFNClassifier(
        n_estimators=1,
        model_path=model_path,
        device=device,
        fit_mode="fit_preprocessors",
        random_state=random_state,
        ignore_pretraining_limits=True,
        inference_config=_simple_tabpfn_inference_config(),
    )
    clf.fit(X_train, y_train)
    return clf


def _find_feature_token_module(model: nn.Module) -> nn.Module:
    """Find a stable module whose output still contains feature tokens."""

    if hasattr(model, "blocks") and len(model.blocks) > 0:  # v2.5/v2.6 path.
        return model.blocks[-1]
    if hasattr(model, "transformer_decoder") and model.transformer_decoder is not None:
        return model.transformer_decoder
    if hasattr(model, "transformer_encoder"):
        return model.transformer_encoder
    raise RuntimeError(
        "Could not find a hookable feature-token module for this TabPFN architecture.",
    )


def _feature_tokens_from_hook_output(
    hook_output: torch.Tensor,
    *,
    n_query_rows: int,
    n_features: int,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """Convert common TabPFN final-block outputs to [N, groups, d]."""

    # TabPFN inference runs under torch.inference_mode(). Clone after leaving
    # that context so the lightweight heads can use these tensors with autograd.
    out = hook_output.detach().float().cpu().clone()

    # v2.5/v2.6 final block: [batch, rows, feature_groups + target, dim].
    if out.ndim == 4 and out.shape[-2] > 1:
        query = out[:, -n_query_rows:, :-1, :]
        return query.squeeze(0)

    # Base PerFeatureTransformer encoder/decoder: [batch, rows, groups + target, dim].
    if out.ndim == 4:
        query = out[:, -n_query_rows:, :-1, :]
        return query.squeeze(0)

    # v3 exposes row embeddings rather than final per-feature tokens. Fall back to
    # broadcasting the target token across features so the script still runs, while
    # making the KG head equivalent to a KG-conditioned pooling head.
    if out.ndim == 3:
        return target_tokens.unsqueeze(1).expand(-1, n_features, -1).contiguous()

    raise RuntimeError(f"Unsupported hook output shape: {tuple(out.shape)}")


def extract_tabpfn_representations(
    clf: TabPFNClassifier,
    X_query: np.ndarray,
    kg_embeddings: torch.Tensor,
) -> RepresentationBatch:
    """Extract target-token and feature-token representations via a forward hook."""

    if len(clf.models_) != 1:
        raise ValueError("This prototype expects n_estimators=1 for KG alignment.")

    model = clf.models_[0]
    hook_outputs: list[torch.Tensor] = []

    def hook_fn(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        hook_outputs.append(output.detach().cpu())

    handle = _find_feature_token_module(model).register_forward_hook(hook_fn)
    try:
        outputs = list(
            clf.executor_.iter_outputs(
                X_query,
                autocast=clf.use_autocast_,
                task_type="multiclass",
                only_return_standard_out=False,
            ),
        )
    finally:
        handle.remove()

    if not outputs:
        raise RuntimeError("TabPFN produced no outputs.")
    if not hook_outputs:
        raise RuntimeError("The representation hook did not capture any tensor.")

    output_dict = outputs[0][0]
    if not isinstance(output_dict, dict) or "test_embeddings" not in output_dict:
        raise RuntimeError("TabPFN did not return test embeddings.")

    target_tokens = (
        output_dict["test_embeddings"].squeeze(1).detach().float().cpu().clone()
    )
    feature_tokens = _feature_tokens_from_hook_output(
        hook_outputs[-1],
        n_query_rows=len(X_query),
        n_features=kg_embeddings.shape[0],
        target_tokens=target_tokens,
    )
    feature_group_kg = group_kg_embeddings(kg_embeddings, feature_tokens.shape[1])

    if feature_tokens.shape[0] != target_tokens.shape[0]:
        raise RuntimeError(
            "Feature-token and target-token row counts disagree: "
            f"{feature_tokens.shape=} {target_tokens.shape=}",
        )

    return RepresentationBatch(
        feature_tokens=feature_tokens,
        target_tokens=target_tokens,
        feature_group_kg=feature_group_kg,
    )



class BaselineHead(nn.Module):
    """Baseline MLP using only the target-token representation."""

    def __init__(self, d_pfn: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(32, d_pfn // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_pfn),
            nn.Linear(d_pfn, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(target_tokens)


class LargeBaselineHead(nn.Module):
    """Large Baseline MLP using only the target-token representation."""

    def __init__(self, d_pfn: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        # KGAwareAttentionHead param size is roughly ~1.5 d_pfn^2
        # To match parameter count, we set the hidden dimension to 1.5 * d_pfn
        hidden = hidden_dim or int(d_pfn * 1.5)
        self.net = nn.Sequential(
            nn.LayerNorm(d_pfn),
            nn.Linear(d_pfn, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(target_tokens)


class KGAwareAttentionHead(nn.Module):
    """Low-parameter KG attention head inspired by the 4.3 formula."""

    def __init__(self, d_pfn: int, d_kg: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(32, d_pfn // 2)
        
        # Project target row to KG space for dot-product attention
        self.query = nn.Linear(d_pfn, d_kg, bias=False)
        self.feature_value = nn.Linear(d_pfn, d_pfn, bias=False)
        
        # Learnable gating parameter initialized to 0
        self.beta = nn.Parameter(torch.zeros(1))
        
        self.net = nn.Sequential(
            nn.LayerNorm(d_pfn),
            nn.Linear(d_pfn, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        target_tokens: torch.Tensor,
        feature_tokens: torch.Tensor,
        feature_group_kg: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Query: target_tokens -> KG space
        query = self.query(target_tokens)
        
        # 2. Key: feature_group_kg (external KG embeddings directly)
        scores = query @ feature_group_kg.T / math.sqrt(feature_group_kg.shape[-1])
        alpha = F.softmax(scores, dim=-1)

        # 3. Value: feature_tokens
        values = self.feature_value(feature_tokens)
        pooled = torch.einsum("ng,ngd->nd", alpha, values)
        
        # 4. Residual Connection
        combined = target_tokens + self.beta * pooled
        
        return self.net(combined)


def train_head(
    head: nn.Module,
    train_repr: RepresentationBatch,
    y_train: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> nn.Module:
    y = numpy_to_tensor(y_train, torch.long)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        if isinstance(head, KGAwareAttentionHead):
            logits = head(
                train_repr.target_tokens,
                train_repr.feature_tokens,
                train_repr.feature_group_kg,
            )
        else:
            logits = head(train_repr.target_tokens)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

    return head.eval()


@torch.no_grad()
def evaluate_head(
    head: nn.Module,
    repr_batch: RepresentationBatch,
    y_true: np.ndarray,
) -> float:
    if isinstance(head, KGAwareAttentionHead):
        logits = head(
            repr_batch.target_tokens,
            repr_batch.feature_tokens,
            repr_batch.feature_group_kg,
        )
    else:
        logits = head(repr_batch.target_tokens)
    preds = tensor_to_numpy(logits.argmax(dim=-1), np.int64)
    return float((preds == y_true).mean())


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def extract_train_representations_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kg_embeddings: torch.Tensor,
    *,
    model_path: str,
    device: str,
    random_state: int,
    n_splits: int = 5,
) -> RepresentationBatch:
    """Extract training representations using cross-validation to avoid leakage."""

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_samples = len(X_train)
    feature_tokens_list = [None] * n_samples
    target_tokens_list = [None] * n_samples
    feature_group_kg = None

    for train_idx, test_idx in kf.split(X_train, y_train):
        X_sup, y_sup = X_train[train_idx], y_train[train_idx]
        X_qry = X_train[test_idx]

        clf_fold = fit_tabpfn(
            X_sup,
            y_sup,
            model_path=model_path,
            device=device,
            random_state=random_state,
        )

        rep = extract_tabpfn_representations(clf_fold, X_qry, kg_embeddings)

        if feature_group_kg is None:
            feature_group_kg = rep.feature_group_kg

        for i, idx in enumerate(test_idx):
            feature_tokens_list[idx] = rep.feature_tokens[i : i + 1]
            target_tokens_list[idx] = rep.target_tokens[i : i + 1]

    return RepresentationBatch(
        feature_tokens=torch.cat(feature_tokens_list, dim=0),
        target_tokens=torch.cat(target_tokens_list, dim=0),
        feature_group_kg=feature_group_kg,
    )


def run_one_seed(
    *,
    seed: int,
    model_path: str,
    device: str,
    n_train: int,
    n_test: int,
    n_features: int,
    kg_dim: int,
    n_concepts: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    n_splits: int = 5,
) -> ExperimentResult:
    set_seed(seed)
    data = generate_kg_smooth_classification_data(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        kg_dim=kg_dim,
        n_concepts=n_concepts,
        concept_noise=0.08,
        feature_noise=0.85,
        label_noise=0.65,
        seed=seed,
    )

    import time
    
    start_time_tabpfn = time.time()
    clf = fit_tabpfn(
        data.X_train,
        data.y_train,
        model_path=model_path,
        device=device,
        random_state=seed,
    )
    tabpfn_accuracy = float(clf.score(data.X_test, data.y_test))
    tabpfn_time = time.time() - start_time_tabpfn

    start_time_repr = time.time()
    train_repr = extract_train_representations_cv(
        data.X_train,
        data.y_train,
        data.kg_embeddings,
        model_path=model_path,
        device=device,
        random_state=seed,
        n_splits=n_splits,
    )
    test_repr = extract_tabpfn_representations(
        clf,
        data.X_test,
        data.kg_embeddings,
    )
    repr_time = time.time() - start_time_repr

    d_pfn = train_repr.target_tokens.shape[-1]
    d_kg = train_repr.feature_group_kg.shape[-1]
    
    # Generate Permuted and Random KG representations
    kg_permuted = data.kg_embeddings[torch.randperm(data.kg_embeddings.shape[0], generator=torch.Generator().manual_seed(seed))]
    feature_group_kg_perm = group_kg_embeddings(kg_permuted, train_repr.feature_tokens.shape[1])
    train_repr_perm = dataclasses.replace(train_repr, feature_group_kg=feature_group_kg_perm)
    test_repr_perm = dataclasses.replace(test_repr, feature_group_kg=feature_group_kg_perm)

    gen_rand = torch.Generator().manual_seed(seed)
    kg_random = normalize_rows(torch.randn(data.kg_embeddings.shape, generator=gen_rand))
    feature_group_kg_rand = group_kg_embeddings(kg_random, train_repr.feature_tokens.shape[1])
    train_repr_rand = dataclasses.replace(train_repr, feature_group_kg=feature_group_kg_rand)
    test_repr_rand = dataclasses.replace(test_repr, feature_group_kg=feature_group_kg_rand)

    baseline_head = BaselineHead(d_pfn)
    start_time_baseline = time.time()
    train_head(
        baseline_head,
        train_repr,
        data.y_train,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    baseline_head_accuracy = evaluate_head(baseline_head, test_repr, data.y_test)
    baseline_head_time = time.time() - start_time_baseline

    large_baseline_head = LargeBaselineHead(d_pfn)
    start_time_large = time.time()
    train_head(
        large_baseline_head,
        train_repr,
        data.y_train,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    large_baseline_head_accuracy = evaluate_head(large_baseline_head, test_repr, data.y_test)
    large_baseline_head_time = time.time() - start_time_large

    kg_head = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_kg = time.time()
    train_head(
        kg_head,
        train_repr,
        data.y_train,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    kg_head_accuracy = evaluate_head(kg_head, test_repr, data.y_test)
    kg_head_time = time.time() - start_time_kg

    kg_head_perm = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_perm = time.time()
    train_head(
        kg_head_perm,
        train_repr_perm,
        data.y_train,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    kg_permuted_head_accuracy = evaluate_head(kg_head_perm, test_repr_perm, data.y_test)
    kg_permuted_head_time = time.time() - start_time_perm

    kg_head_rand = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_rand = time.time()
    train_head(
        kg_head_rand,
        train_repr_rand,
        data.y_train,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
    )
    kg_random_head_accuracy = evaluate_head(kg_head_rand, test_repr_rand, data.y_test)
    kg_random_head_time = time.time() - start_time_rand

    print(
        f"seed={seed} d_pfn={d_pfn} "
        f"kg_params={count_parameters(kg_head):,} "
        f"large_baseline_params={count_parameters(large_baseline_head):,}",
    )

    return ExperimentResult(
        seed=seed,
        repr_extract_time=repr_time,
        tabpfn_accuracy=tabpfn_accuracy,
        baseline_head_accuracy=baseline_head_accuracy,
        large_baseline_head_accuracy=large_baseline_head_accuracy,
        kg_head_accuracy=kg_head_accuracy,
        kg_permuted_head_accuracy=kg_permuted_head_accuracy,
        kg_random_head_accuracy=kg_random_head_accuracy,
        tabpfn_time=tabpfn_time,
        baseline_head_time=baseline_head_time,
        large_baseline_head_time=large_baseline_head_time,
        kg_head_time=kg_head_time,
        kg_permuted_head_time=kg_permuted_head_time,
        kg_random_head_time=kg_random_head_time,
    )


def summarize(results: Sequence[ExperimentResult]) -> None:
    def mean_std(values: Sequence[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    tabpfn_values = [r.tabpfn_accuracy for r in results]
    baseline_values = [r.baseline_head_accuracy for r in results]
    large_baseline_values = [r.large_baseline_head_accuracy for r in results]
    kg_values = [r.kg_head_accuracy for r in results]
    kg_perm_values = [r.kg_permuted_head_accuracy for r in results]
    kg_rand_values = [r.kg_random_head_accuracy for r in results]
    
    tabpfn_times = [r.tabpfn_time for r in results]
    baseline_times = [r.baseline_head_time for r in results]
    large_baseline_times = [r.large_baseline_head_time for r in results]
    kg_times = [r.kg_head_time for r in results]
    kg_perm_times = [r.kg_permuted_head_time for r in results]
    kg_rand_times = [r.kg_random_head_time for r in results]
    repr_times = [r.repr_extract_time for r in results]

    print("\nPer-seed results")
    for r in results:
        print(
            f"  seed={r.seed:3d} "
            f"TabPFN={100 * r.tabpfn_accuracy:5.1f}% ({r.tabpfn_time:.2f}s) "
            f"Base={100 * r.baseline_head_accuracy:5.1f}% ({r.baseline_head_time:.2f}s) "
            f"L-Base={100 * r.large_baseline_head_accuracy:5.1f}% ({r.large_baseline_head_time:.2f}s) "
            f"KG-Real={100 * r.kg_head_accuracy:5.1f}% ({r.kg_head_time:.2f}s) "
            f"KG-Perm={100 * r.kg_permuted_head_accuracy:5.1f}% ({r.kg_permuted_head_time:.2f}s) "
            f"KG-Rand={100 * r.kg_random_head_accuracy:5.1f}% ({r.kg_random_head_time:.2f}s)",
        )

    print("\nSummary")
    mean_repr_t, std_repr_t = mean_std(repr_times)
    print(f"  Repr Extraction:   ---    (Time: {mean_repr_t:.2f}s +/- {std_repr_t:.2f}s)")
    
    mean_tab, std_tab = mean_std(tabpfn_values)
    mean_tab_t, std_tab_t = mean_std(tabpfn_times)
    print(f"  Pure TabPFN:       {100 * mean_tab:5.1f}% +/- {100 * std_tab:4.1f}%  (Time: {mean_tab_t:.2f}s +/- {std_tab_t:.2f}s)")
    
    mean_base, std_base = mean_std(baseline_values)
    mean_base_t, std_base_t = mean_std(baseline_times)
    print(f"  Baseline head:     {100 * mean_base:5.1f}% +/- {100 * std_base:4.1f}%  (Time: {mean_base_t:.2f}s +/- {std_base_t:.2f}s)")

    mean_lbase, std_lbase = mean_std(large_baseline_values)
    mean_lbase_t, std_lbase_t = mean_std(large_baseline_times)
    print(f"  L-Baseline head:   {100 * mean_lbase:5.1f}% +/- {100 * std_lbase:4.1f}%  (Time: {mean_lbase_t:.2f}s +/- {std_lbase_t:.2f}s)")

    mean_kg, std_kg = mean_std(kg_values)
    mean_kg_t, std_kg_t = mean_std(kg_times)
    print(f"  KG-aware head:     {100 * mean_kg:5.1f}% +/- {100 * std_kg:4.1f}%  (Time: {mean_kg_t:.2f}s +/- {std_kg_t:.2f}s)")

    mean_kgp, std_kgp = mean_std(kg_perm_values)
    mean_kgp_t, std_kgp_t = mean_std(kg_perm_times)
    print(f"  KG-permuted head:  {100 * mean_kgp:5.1f}% +/- {100 * std_kgp:4.1f}%  (Time: {mean_kgp_t:.2f}s +/- {std_kgp_t:.2f}s)")

    mean_kgr, std_kgr = mean_std(kg_rand_values)
    mean_kgr_t, std_kgr_t = mean_std(kg_rand_times)
    print(f"  KG-random head:    {100 * mean_kgr:5.1f}% +/- {100 * std_kgr:4.1f}%  (Time: {mean_kgr_t:.2f}s +/- {std_kgr_t:.2f}s)")

    kg_minus_base = (mean_kg - mean_lbase) * 100
    kg_minus_perm = (mean_kg - mean_kgp) * 100
    print(f"\n  [Deltas]")
    print(f"  KG_real - LargeBaseline: {kg_minus_base:+.2f}%")
    print(f"  KG_real - KG_permuted:   {kg_minus_perm:+.2f}%")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--kg-dim", type=int, default=16)
    parser.add_argument("--n-concepts", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        run_one_seed(
            seed=seed,
            model_path=args.model_path,
            device=args.device,
            n_train=args.n_train,
            n_test=args.n_test,
            n_features=args.n_features,
            kg_dim=args.kg_dim,
            n_concepts=args.n_concepts,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        for seed in args.seeds
    ]
    summarize(results)


if __name__ == "__main__":
    main()
