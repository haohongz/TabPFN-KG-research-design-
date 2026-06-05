"""End-to-End Simulation of Knowledge Graph-Aware TabPFN — WITH Y EMBEDDING.

基于孙傲的原始代码，加入 y 的 embedding。
改动用 ★ 标记，其余代码完全不动。

改动说明：
1. KGSimulationData 新增 y_embedding 字段
2. generate_kg_smooth_classification_data 里计算 y_embedding = active concepts 的平均
3. RepresentationBatch 新增 y_kg_embedding 字段
4. KGAwareAttentionHead 的 forward 里用 y_embedding 调整注意力权重
5. 相应的训练/评估/调用代码同步更新

运行方式和原来一样：
  python tabpfn_kg_with_y.py --device cpu --n-features 200 --n-train 100 --n-concepts 6
  python tabpfn_kg_with_y.py --device cuda --n-features 1000 --n-train 40 --n-concepts 20
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
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


# ★ 新增 y_embedding 字段
@dataclasses.dataclass(frozen=True)
class KGSimulationData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    kg_embeddings: torch.Tensor
    true_feature_weights: torch.Tensor
    y_embedding: torch.Tensor           # ★ 新增


# ★ 新增 y_kg_embedding 字段
@dataclasses.dataclass(frozen=True)
class RepresentationBatch:
    feature_tokens: torch.Tensor
    target_tokens: torch.Tensor
    feature_group_kg: torch.Tensor
    y_kg_embedding: torch.Tensor        # ★ 新增


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
    return np.asarray(tensor.detach().cpu().tolist(), dtype=dtype)


def numpy_to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
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
    """Generate dataset with KG embeddings + y embedding.
    
    ★ 改动：
    1. 计算 y_embedding = active concept embeddings 的平均
    2. 用 y_embedding 参与特征权重计算（特征和 y 的相似度越高 → 权重越大）
    """
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator().manual_seed(seed)

    concept_embeddings = normalize_rows(
        torch.randn(n_concepts, kg_dim, generator=torch_gen),
    )
    feature_concepts = torch.arange(n_features) % n_concepts
    feature_concepts = feature_concepts[torch.randperm(n_features, generator=torch_gen)]

    kg_embeddings = concept_embeddings[feature_concepts] + concept_noise * torch.randn(
        n_features, kg_dim, generator=torch_gen,
    )
    kg_embeddings = normalize_rows(kg_embeddings)

    active_concepts = torch.randperm(n_concepts, generator=torch_gen)[
        : max(2, n_concepts // 3)
    ]

    # ★ 新增：y 的 embedding = active concept 的平均，归一化
    y_embedding = concept_embeddings[active_concepts].mean(dim=0)
    y_embedding = normalize_rows(y_embedding.unsqueeze(0)).squeeze(0)

    # ★ 改动：特征权重用 KG embedding 和 y_embedding 的相似度
    # 原来的方式（保留一部分作为 smooth component）
    concept_direction = torch.randn(kg_dim, generator=torch_gen)
    concept_direction = normalize_rows(concept_direction.unsqueeze(0)).squeeze(0)
    smooth_weight = kg_embeddings @ concept_direction

    # ★ 新增：y-aware weight = 特征 KG embedding 和 y_embedding 的余弦相似度
    y_aware_weight = kg_embeddings @ y_embedding  # (n_features,)

    # ★ 改动：结合 smooth weight 和 y-aware weight
    # y_aware_weight 让 active concept 的特征权重大，其他的小
    concept_sign = torch.zeros(n_concepts)
    concept_sign[active_concepts] = torch.sign(
        torch.randn(active_concepts.numel(), generator=torch_gen),
    ).clamp(min=-1, max=1)
    concept_sign[concept_sign == 0] = 1
    sparse_concept_weight = concept_sign[feature_concepts]

    true_feature_weights = smooth_weight + 1.25 * sparse_concept_weight + 0.75 * y_aware_weight  # ★ 加了 y 的贡献
    true_feature_weights = true_feature_weights / true_feature_weights.norm().clamp_min(1e-8)

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
        y_embedding=y_embedding.float(),    # ★ 新增
    )


def group_kg_embeddings(
    kg_embeddings: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
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
    if hasattr(model, "blocks") and len(model.blocks) > 0:
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
    out = hook_output.detach().float().cpu().clone()
    if out.ndim == 4 and out.shape[-2] > 1:
        query = out[:, -n_query_rows:, :-1, :]
        return query.squeeze(0)
    if out.ndim == 4:
        query = out[:, -n_query_rows:, :-1, :]
        return query.squeeze(0)
    if out.ndim == 3:
        return target_tokens.unsqueeze(1).expand(-1, n_features, -1).contiguous()
    raise RuntimeError(f"Unsupported hook output shape: {tuple(out.shape)}")


# ★ 改动：返回值包含 y_kg_embedding
def extract_tabpfn_representations(
    clf: TabPFNClassifier,
    X_query: np.ndarray,
    kg_embeddings: torch.Tensor,
    y_embedding: torch.Tensor,          # ★ 新增参数
) -> RepresentationBatch:
    if len(clf.models_) != 1:
        raise ValueError("This prototype expects n_estimators=1 for KG alignment.")

    model = clf.models_[0]
    hook_outputs: list[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
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
            f"Row count mismatch: {feature_tokens.shape=} {target_tokens.shape=}",
        )

    return RepresentationBatch(
        feature_tokens=feature_tokens,
        target_tokens=target_tokens,
        feature_group_kg=feature_group_kg,
        y_kg_embedding=y_embedding,     # ★ 新增
    )


class BaselineHead(nn.Module):
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

    def forward(self, target_tokens, **kwargs):
        return self.net(target_tokens)


class LargeBaselineHead(nn.Module):
    def __init__(self, d_pfn: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or int(d_pfn * 1.5)
        self.net = nn.Sequential(
            nn.LayerNorm(d_pfn),
            nn.Linear(d_pfn, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 2),
        )

    def forward(self, target_tokens, **kwargs):
        return self.net(target_tokens)


# ★ 改动：KGHead 加入 y_embedding
class KGAwareAttentionHead(nn.Module):
    """KG attention head with y-embedding awareness.
    
    ★ 改动：
    在计算 attention scores 时，用 y_embedding 调整每个特征组的重要性。
    y_relevance_j = cos_sim(feature_group_kg_j, y_embedding)
    最终 scores = query @ keys + lambda * y_relevance
    
    这样和 y 关系近的特征组得到更高的 attention 权重。
    """

    def __init__(self, d_pfn: int, d_kg: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(32, d_pfn // 2)
        self.query = nn.Linear(d_pfn, d_kg, bias=False)
        self.feature_value = nn.Linear(d_pfn, d_pfn, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))
        # ★ 新增：y_relevance 的可学习权重
        self.y_lambda = nn.Parameter(torch.tensor(1.0))
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
        y_kg_embedding: torch.Tensor,   # ★ 新增参数
    ) -> torch.Tensor:
        # 1. Query: target → KG space
        query = self.query(target_tokens)

        # 2. Key: feature_group_kg
        scores = query @ feature_group_kg.T / math.sqrt(feature_group_kg.shape[-1])

        # ★ 3. Y-aware bias: 和 y 关系近的特征组得到更高的分数
        y_relevance = feature_group_kg @ y_kg_embedding  # (n_groups,)
        scores = scores + self.y_lambda * y_relevance.unsqueeze(0)  # broadcast to (N, n_groups)

        alpha = F.softmax(scores, dim=-1)

        # 4. Value: feature_tokens
        values = self.feature_value(feature_tokens)
        pooled = torch.einsum("ng,ngd->nd", alpha, values)

        # 5. Residual
        combined = target_tokens + self.beta * pooled

        return self.net(combined)


# ★ 改动：统一 train_head 接口
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
        logits = head(
            target_tokens=train_repr.target_tokens,
            feature_tokens=train_repr.feature_tokens,
            feature_group_kg=train_repr.feature_group_kg,
            y_kg_embedding=train_repr.y_kg_embedding,   # ★ 传 y
        )
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
    logits = head(
        target_tokens=repr_batch.target_tokens,
        feature_tokens=repr_batch.feature_tokens,
        feature_group_kg=repr_batch.feature_group_kg,
        y_kg_embedding=repr_batch.y_kg_embedding,       # ★ 传 y
    )
    preds = tensor_to_numpy(logits.argmax(dim=-1), np.int64)
    return float((preds == y_true).mean())


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ★ 改动：传 y_embedding
def extract_train_representations_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kg_embeddings: torch.Tensor,
    y_embedding: torch.Tensor,          # ★ 新增
    *,
    model_path: str,
    device: str,
    random_state: int,
    n_splits: int = 5,
) -> RepresentationBatch:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    n_samples = len(X_train)
    feature_tokens_list = [None] * n_samples
    target_tokens_list = [None] * n_samples
    feature_group_kg = None

    for train_idx, test_idx in kf.split(X_train, y_train):
        X_sup, y_sup = X_train[train_idx], y_train[train_idx]
        X_qry = X_train[test_idx]

        clf_fold = fit_tabpfn(
            X_sup, y_sup,
            model_path=model_path,
            device=device,
            random_state=random_state,
        )

        rep = extract_tabpfn_representations(clf_fold, X_qry, kg_embeddings, y_embedding)  # ★

        if feature_group_kg is None:
            feature_group_kg = rep.feature_group_kg

        for i, idx in enumerate(test_idx):
            feature_tokens_list[idx] = rep.feature_tokens[i : i + 1]
            target_tokens_list[idx] = rep.target_tokens[i : i + 1]

    return RepresentationBatch(
        feature_tokens=torch.cat(feature_tokens_list, dim=0),
        target_tokens=torch.cat(target_tokens_list, dim=0),
        feature_group_kg=feature_group_kg,
        y_kg_embedding=y_embedding,     # ★
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

    start_time_tabpfn = time.time()
    clf = fit_tabpfn(
        data.X_train, data.y_train,
        model_path=model_path,
        device=device,
        random_state=seed,
    )
    tabpfn_accuracy = float(clf.score(data.X_test, data.y_test))
    tabpfn_time = time.time() - start_time_tabpfn

    start_time_repr = time.time()
    train_repr = extract_train_representations_cv(
        data.X_train, data.y_train,
        data.kg_embeddings,
        data.y_embedding,               # ★ 传 y
        model_path=model_path,
        device=device,
        random_state=seed,
        n_splits=n_splits,
    )
    test_repr = extract_tabpfn_representations(
        clf, data.X_test,
        data.kg_embeddings,
        data.y_embedding,               # ★ 传 y
    )
    repr_time = time.time() - start_time_repr

    d_pfn = train_repr.target_tokens.shape[-1]
    d_kg = train_repr.feature_group_kg.shape[-1]

    # Permuted KG（打乱特征对应关系，但 y_embedding 不变）
    gen_perm = torch.Generator().manual_seed(seed)
    kg_permuted = data.kg_embeddings[torch.randperm(data.kg_embeddings.shape[0], generator=gen_perm)]
    fg_kg_perm = group_kg_embeddings(kg_permuted, train_repr.feature_tokens.shape[1])
    train_repr_perm = dataclasses.replace(train_repr, feature_group_kg=fg_kg_perm)
    test_repr_perm = dataclasses.replace(test_repr, feature_group_kg=fg_kg_perm)

    # Random KG（随机向量，y_embedding 也随机）
    gen_rand = torch.Generator().manual_seed(seed)
    kg_random = normalize_rows(torch.randn(data.kg_embeddings.shape, generator=gen_rand))
    fg_kg_rand = group_kg_embeddings(kg_random, train_repr.feature_tokens.shape[1])
    y_random = normalize_rows(torch.randn(1, kg_dim, generator=gen_rand)).squeeze(0)
    train_repr_rand = dataclasses.replace(train_repr, feature_group_kg=fg_kg_rand, y_kg_embedding=y_random)
    test_repr_rand = dataclasses.replace(test_repr, feature_group_kg=fg_kg_rand, y_kg_embedding=y_random)

    # ---- Baseline Head ----
    baseline_head = BaselineHead(d_pfn)
    start_time_baseline = time.time()
    train_head(baseline_head, train_repr, data.y_train,
               epochs=epochs, lr=lr, weight_decay=weight_decay)
    baseline_head_accuracy = evaluate_head(baseline_head, test_repr, data.y_test)
    baseline_head_time = time.time() - start_time_baseline

    # ---- Large Baseline Head ----
    large_baseline_head = LargeBaselineHead(d_pfn)
    start_time_large = time.time()
    train_head(large_baseline_head, train_repr, data.y_train,
               epochs=epochs, lr=lr, weight_decay=weight_decay)
    large_baseline_head_accuracy = evaluate_head(large_baseline_head, test_repr, data.y_test)
    large_baseline_head_time = time.time() - start_time_large

    # ---- KG Head (Real KG + Real Y) ----
    kg_head = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_kg = time.time()
    train_head(kg_head, train_repr, data.y_train,
               epochs=epochs, lr=lr, weight_decay=weight_decay)
    kg_head_accuracy = evaluate_head(kg_head, test_repr, data.y_test)
    kg_head_time = time.time() - start_time_kg

    # ---- KG Head (Permuted KG, Real Y) ----
    kg_head_perm = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_perm = time.time()
    train_head(kg_head_perm, train_repr_perm, data.y_train,
               epochs=epochs, lr=lr, weight_decay=weight_decay)
    kg_permuted_head_accuracy = evaluate_head(kg_head_perm, test_repr_perm, data.y_test)
    kg_permuted_head_time = time.time() - start_time_perm

    # ---- KG Head (Random KG + Random Y) ----
    kg_head_rand = KGAwareAttentionHead(d_pfn, d_kg)
    start_time_rand = time.time()
    train_head(kg_head_rand, train_repr_rand, data.y_train,
               epochs=epochs, lr=lr, weight_decay=weight_decay)
    kg_random_head_accuracy = evaluate_head(kg_head_rand, test_repr_rand, data.y_test)
    kg_random_head_time = time.time() - start_time_rand

    print(
        f"seed={seed} d_pfn={d_pfn} "
        f"kg_params={count_parameters(kg_head):,} "
        f"large_baseline_params={count_parameters(large_baseline_head):,} "
        f"TabPFN={100*tabpfn_accuracy:.1f}% "
        f"Base={100*baseline_head_accuracy:.1f}% "
        f"KG_real={100*kg_head_accuracy:.1f}% "
        f"KG_perm={100*kg_permuted_head_accuracy:.1f}% "
        f"KG_rand={100*kg_random_head_accuracy:.1f}%"
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
    def mean_std(values):
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    print("\n" + "=" * 80)
    print("  Results (with Y embedding)")
    print("=" * 80)

    print("\nPer-seed results")
    for r in results:
        print(
            f"  seed={r.seed:3d} "
            f"TabPFN={100*r.tabpfn_accuracy:5.1f}% "
            f"Base={100*r.baseline_head_accuracy:5.1f}% "
            f"L-Base={100*r.large_baseline_head_accuracy:5.1f}% "
            f"KG-Real={100*r.kg_head_accuracy:5.1f}% "
            f"KG-Perm={100*r.kg_permuted_head_accuracy:5.1f}% "
            f"KG-Rand={100*r.kg_random_head_accuracy:5.1f}%"
        )

    print("\nSummary")
    for name, vals in [
        ("Pure TabPFN",      [r.tabpfn_accuracy for r in results]),
        ("Baseline head",    [r.baseline_head_accuracy for r in results]),
        ("L-Baseline head",  [r.large_baseline_head_accuracy for r in results]),
        ("KG-real+Y head",   [r.kg_head_accuracy for r in results]),
        ("KG-permuted head", [r.kg_permuted_head_accuracy for r in results]),
        ("KG-random head",   [r.kg_random_head_accuracy for r in results]),
    ]:
        m, s = mean_std(vals)
        print(f"  {name:<20s} {100*m:5.1f}% +/- {100*s:4.1f}%")

    kg_vals = [r.kg_head_accuracy for r in results]
    lbase_vals = [r.large_baseline_head_accuracy for r in results]
    perm_vals = [r.kg_permuted_head_accuracy for r in results]
    print(f"\n  [Deltas]")
    print(f"  KG_real+Y - LargeBaseline: {100*(np.mean(kg_vals)-np.mean(lbase_vals)):+.2f}%")
    print(f"  KG_real+Y - KG_permuted:   {100*(np.mean(kg_vals)-np.mean(perm_vals)):+.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--kg-dim", type=int, default=16)
    parser.add_argument("--n-concepts", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Config: n_features={args.n_features}, n_train={args.n_train}, "
          f"n_concepts={args.n_concepts}, kg_dim={args.kg_dim}")
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
