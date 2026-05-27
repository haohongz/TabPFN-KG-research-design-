#!/usr/bin/env python3
"""
TabICL + KG-Adapter (Variant A) — Real KG Complete Experiment
==============================================================
Front injection at col_embedder output: E in R^{B x T x (F+C) x 128}
KG: Hetionet TransE (128d) + Sentence-BERT (384d) = 512d
Datasets: BreastCancer (sklearn) + Leukemia (Golub 1999)
"""
import warnings
warnings.filterwarnings("ignore")
import os, sys, json, time, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tabicl import TabICLClassifier
from tabicl._model.tabicl import TabICL

def load_breast_cancer_data():
    data = load_breast_cancer()
    return {"name": "BreastCancer", "X": data.data, "y": data.target,
            "feature_names": list(data.feature_names), "desc": "n=569, d=30"}

def load_leukemia_data(max_features=1000):
    import pandas as pd
    import urllib.request, tempfile
    url = "http://hastie.su.domains/CASI_files/DATA/leukemia_big.csv"
    print(f"    Downloading Golub Leukemia (72 patients, 7128 genes)...")
    tmp = os.path.join(tempfile.gettempdir(), "leukemia_big.csv")
    if not os.path.exists(tmp):
        urllib.request.urlretrieve(url, tmp)
    df = pd.read_csv(tmp)
    y_raw = df.columns.values
    y = np.array([0 if "ALL" in str(v) else 1 for v in y_raw])
    X = df.values.T.astype(np.float64)
    feature_names = [f"gene_{i}" for i in range(X.shape[1])]
    print(f"    Raw: n={X.shape[0]}, d={X.shape[1]}")
    if X.shape[1] > max_features:
        variances = np.var(X, axis=0)
        top_idx = np.sort(np.argsort(variances)[-max_features:])
        X = X[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]
        print(f"    Top variance: d={X.shape[1]}")
    return {"name": "Leukemia", "X": X, "y": y,
            "feature_names": feature_names, "desc": f"n={X.shape[0]}, d={X.shape[1]}"}

class HetionetKGEncoder:
    def __init__(self, device="cpu", kg_dim=128, epochs=30, cache_path="hetionet_cache.pkl"):
        self.device = device
        self.kg_dim = kg_dim
        self.epochs = epochs
        self.cache_path = cache_path
        self.entity_embeddings = None
        self.entity_names = None
    def train_or_load(self):
        if os.path.exists(self.cache_path):
            print("    [Hetionet] Loading from cache...")
            with open(self.cache_path, "rb") as f:
                cache = pickle.load(f)
            self.entity_embeddings = cache["embeddings"]
            self.entity_names = cache["names"]
            print(f"    [Hetionet] {len(self.entity_names)} entities, {self.entity_embeddings.shape[1]}d")
            return
        from pykeen.datasets import Hetionet
        from pykeen.pipeline import pipeline
        print("    [Hetionet] Downloading dataset...")
        dataset = Hetionet()
        print(f"    [Hetionet] {dataset.num_entities} entities, {dataset.num_relations} relations")
        print(f"    [Hetionet] Training TransE ({self.epochs} epochs, dim={self.kg_dim})...")
        t0 = time.time()
        result = pipeline(dataset=dataset, model="TransE",
            model_kwargs=dict(embedding_dim=self.kg_dim),
            training_kwargs=dict(num_epochs=self.epochs, batch_size=512),
            device=self.device, random_seed=42)
        print(f"    [Hetionet] Done ({time.time()-t0:.0f}s)")
        self.entity_embeddings = result.model.entity_representations[0]().detach().cpu().numpy()
        self.entity_names = list(dataset.entity_to_id.keys())
        with open(self.cache_path, "wb") as f:
            pickle.dump({"embeddings": self.entity_embeddings, "names": self.entity_names}, f)
        print(f"    [Hetionet] Cached: {self.cache_path}")
    def map_features(self, feature_names, sbert_device="cpu"):
        from sentence_transformers import SentenceTransformer
        print(f"    [Hetionet] Mapping {len(feature_names)} features...")
        sbert = SentenceTransformer("all-MiniLM-L6-v2", device=sbert_device)
        feat_embs = sbert.encode(feature_names, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        print(f"    [Hetionet] Encoding {len(self.entity_names)} entity names...")
        entity_embs = sbert.encode(self.entity_names, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True, batch_size=512)
        sims = feat_embs @ entity_embs.T
        nearest_idx = sims.argmax(axis=1)
        nearest_sims = sims[np.arange(len(feature_names)), nearest_idx]
        print("    [Hetionet] Mapping examples:")
        for i in range(min(5, len(feature_names))):
            print(f"      {feature_names[i]:30s} -> {self.entity_names[nearest_idx[i]]:30s} (sim={nearest_sims[i]:.3f})")
        print(f"    [Hetionet] Mean similarity: {nearest_sims.mean():.3f}")
        return self.entity_embeddings[nearest_idx]

def build_combined_kg(feature_names, hetionet_encoder, device="cpu"):
    from sentence_transformers import SentenceTransformer
    print("    [SBERT] Encoding features...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    sbert_emb = sbert.encode([f"biomedical feature {n}" for n in feature_names],
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    kg_emb = hetionet_encoder.map_features(feature_names, sbert_device=device)
    norms = np.linalg.norm(kg_emb, axis=1, keepdims=True) + 1e-8
    kg_emb = kg_emb / norms
    combined = np.concatenate([sbert_emb, kg_emb], axis=1)
    print(f"    [KG] Combined: SBERT({sbert_emb.shape[1]}) + Hetionet({kg_emb.shape[1]}) = {combined.shape[1]}d")
    return combined.astype(np.float32)

def build_sbert_only_kg(feature_names, device="cpu"):
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    emb = sbert.encode([f"biomedical feature {n}" for n in feature_names],
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    print(f"    [KG] SBERT-only: {emb.shape}")
    return emb.astype(np.float32)

class ScalarGateAdapter(nn.Module):
    """Variant A: front injection at col_embedder, orthogonal alignment."""
    def __init__(self, kg_embeddings, embed_dim=128, init_gate=3.0):
        super().__init__()
        n_features, kg_dim = kg_embeddings.shape
        self.n_features = n_features
        self.gate_logits = nn.Parameter(torch.full((n_features,), init_gate))
        self.align = nn.Linear(kg_dim, embed_dim, bias=False)
        nn.init.orthogonal_(self.align.weight)
        self.register_buffer("kg_emb", torch.from_numpy(kg_embeddings).float())
    def forward(self, E):
        B, T, FC, D = E.shape
        F = self.n_features
        C = FC - F
        z = self.align(self.kg_emb)
        z = z / (z.norm(dim=1, keepdim=True) + 1e-8) * E[:,:,:F,:].detach().norm(dim=3, keepdim=True).mean(dim=(0,1))
        alpha = torch.sigmoid(self.gate_logits)
        e_feat = E[:, :, :F, :]
        e_cls = E[:, :, F:, :]
        a = alpha.view(1, 1, F, 1)
        z_exp = z.unsqueeze(0).unsqueeze(0).expand(B, T, F, D)
        e_new = a * e_feat + (1 - a) * z_exp
        return torch.cat([e_new, e_cls], dim=2)
    def set_scale(self, value):
        pass

def make_random_kg(n, d, seed=42):
    rng = np.random.RandomState(seed)
    e = rng.randn(n, d).astype(np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

def make_permuted_kg(emb, seed=42):
    rng = np.random.RandomState(seed)
    return emb[rng.permutation(len(emb))].copy()

class TabICLWithKG:
    def __init__(self, n_estimators=4, device="cpu", random_state=42,
                 kg_embeddings=None, inject_scale=0.3):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.kg_embeddings = kg_embeddings
        self.inject_scale = inject_scale
        self._hook = None
    def fit(self, X, y):
        self.clf = TabICLClassifier(n_estimators=self.n_estimators, device=self.device,
            random_state=self.random_state, verbose=False)
        self.clf.fit(X, y)
        if self.kg_embeddings is not None:
            model = self.clf.model_
            embed_dim = model.embed_dim
            adapter = ScalarGateAdapter(self.kg_embeddings, embed_dim, init_gate=self.inject_scale)
            adapter.eval().to(self.device)
            self._adapter = adapter
            def hook_fn(module, input, output):
                with torch.no_grad():
                    return adapter(output)
            self._hook = model.col_embedder.register_forward_hook(hook_fn)
        return self
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    def cleanup(self):
        if self._hook:
            self._hook.remove()
            self._hook = None

def run_experiment(args):
    print("="*65)
    print("  TabICL + KG-Adapter — Real KG (Hetionet + SBERT)")
    print("  Front injection at col_embedder output")
    print("  Datasets: BreastCancer + Leukemia (d >> n)")
    print("="*65)
    print(f"  scale={args.scale}, n_estimators={args.n_estimators}, device={args.device}")
    print("\n[1] Training Hetionet TransE...")
    hetionet = HetionetKGEncoder(device=args.device, kg_dim=128, epochs=args.kg_epochs, cache_path="hetionet_cache.pkl")
    try:
        hetionet.train_or_load()
        has_hetionet = True
    except Exception as e:
        print(f"    Warning: Hetionet failed: {e}")
        print(f"    Falling back to SBERT-only")
        has_hetionet = False
    print("\n[2] Loading datasets...")
    datasets = [load_breast_cancer_data()]
    if not args.skip_leukemia:
        try:
            datasets.append(load_leukemia_data(max_features=args.leukemia_d))
        except Exception as e:
            print(f"    Warning: Leukemia failed: {e}")
    all_results = {}
    for ds in datasets:
        name = ds["name"]
        X, y = ds["X"], ds["y"]
        feature_names = ds["feature_names"]
        print(f"\n{'='*65}")
        print(f"  Dataset: {name} ({ds['desc']})")
        print(f"{'='*65}")
        print("\n  [KG] Building embeddings...")
        if has_hetionet:
            real_kg = build_combined_kg(feature_names, hetionet, device=args.device)
            kg_source = "SBERT+Hetionet"
        else:
            real_kg = build_sbert_only_kg(feature_names, device=args.device)
            kg_source = "SBERT-only"
        kg_dim = real_kg.shape[1]
        random_kg = make_random_kg(len(feature_names), kg_dim, seed=args.seed)
        permuted_kg = make_permuted_kg(real_kg, seed=args.seed)
        methods = [("Baseline",None),("+Real KG",real_kg),("+Random KG",random_kg),("+Permuted KG",permuted_kg)]
        results = {m:{"accuracy":[],"f1":[],"auc":[]} for m,_ in methods}
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for fold,(tr,te) in enumerate(skf.split(X,y)):
            print(f"\n  Fold {fold+1}/{args.n_splits}")
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            for method_name, kg_emb in methods:
                t0 = time.time()
                clf = TabICLWithKG(n_estimators=args.n_estimators, device=args.device,
                    random_state=args.seed, kg_embeddings=kg_emb, inject_scale=args.scale)
                clf.fit(X_tr, y_tr)
                proba = clf.predict_proba(X_te)
                clf.cleanup()
                pred = proba.argmax(axis=1)
                acc = accuracy_score(y_te, pred)
                f1 = f1_score(y_te, pred, average="weighted")
                try:
                    auc = roc_auc_score(y_te, proba[:,1])
                except:
                    auc = float("nan")
                results[method_name]["accuracy"].append(acc)
                results[method_name]["f1"].append(f1)
                results[method_name]["auc"].append(auc)
                dt = time.time()-t0
                print(f"    {method_name:<18s} Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f} ({dt:.1f}s)")
        all_results[name] = {"results":results,"desc":ds["desc"],"kg_source":kg_source}
    print("\n\n"+"="*75)
    print("  RESULTS SUMMARY")
    print("="*75)
    for ds_name, ds_data in all_results.items():
        res = ds_data["results"]
        print(f"\n  {ds_name} ({ds_data['desc']}) - KG: {ds_data['kg_source']}")
        print(f"  {'Method':<20s} {'Accuracy':<16s} {'F1':<16s} {'AUC':<16s}")
        print(f"  {'─'*68}")
        for mn, metrics in res.items():
            row = f"  {mn:<20s}"
            for m in ["accuracy","f1","auc"]:
                v = np.array(metrics[m])
                row += f" {v.mean():.4f}+/-{v.std():.4f} "
            print(row)
    json_out = {}
    for ds_name, ds_data in all_results.items():
        json_out[ds_name] = {"kg_source":ds_data["kg_source"]}
        for mn, metrics in ds_data["results"].items():
            json_out[ds_name][mn] = {m:{"values":metrics[m],"mean":float(np.mean(metrics[m])),"std":float(np.std(metrics[m]))} for m in ["accuracy","f1","auc"]}
    json_out["config"] = vars(args)
    json_path = args.output.replace(".png",".json")
    with open(json_path,"w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n  JSON: {json_path}")
    plot_results(all_results, args.n_splits, args.output)
    return all_results

def plot_results(all_results, n_splits, save_path):
    n_ds = len(all_results)
    fig, axes = plt.subplots(n_ds, 3, figsize=(18, 5.5*n_ds))
    if n_ds == 1:
        axes = axes.reshape(1,-1)
    fig.suptitle("TabICL + KG-Adapter (Variant A)\nFront Injection at col_embedder — Real KG: Hetionet TransE + SBERT",
        fontsize=14, fontweight="bold", y=1.02)
    folds = list(range(1, n_splits+1))
    mtitles = [("Accuracy","accuracy"),("F1 Score","f1"),("ROC AUC","auc")]
    styles = {"Baseline":{"c":"#1565C0","m":"o","ls":"-","lw":2.5},"+Real KG":{"c":"#D32F2F","m":"s","ls":"-","lw":2.5},
        "+Random KG":{"c":"#9E9E9E","m":"^","ls":"--","lw":1.5},"+Permuted KG":{"c":"#388E3C","m":"D","ls":"--","lw":1.5}}
    for row,(ds_name,ds_data) in enumerate(all_results.items()):
        res = ds_data["results"]
        for col,(title,key) in enumerate(mtitles):
            ax = axes[row,col]
            for mn,metrics in res.items():
                vals = metrics[key]
                s = styles.get(mn,{"c":"gray","m":"x","ls":":","lw":1})
                mean = np.mean(vals)
                ax.plot(folds, vals, marker=s["m"], linewidth=s["lw"], linestyle=s["ls"],
                    color=s["c"], markersize=7, label=f"{mn} ({mean:.4f})", zorder=3)
            ax.set_xlabel("Fold"); ax.set_ylabel(title)
            ax.set_title(f"{ds_name} - {title}\n(KG: {ds_data['kg_source']})", fontweight="bold", fontsize=10)
            ax.set_xticks(folds)
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle="--")
            all_v = [v for m in res.values() for v in m[key] if not np.isnan(v)]
            if all_v:
                ax.set_ylim(max(0,min(all_v)-0.03), min(1,max(all_v)+0.03))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {save_path}")

def demo_hook(device="cpu"):
    print("\n  Hook dimension check")
    print("  "+"─"*40)
    X, y = load_breast_cancer(return_X_y=True)
    clf = TabICLClassifier(n_estimators=1, device=device, verbose=False)
    clf.fit(X[:100], y[:100])
    model = clf.model_
    captured = {}
    def hook_col(mod, inp, out):
        captured["col"] = out.detach().clone()
    def hook_row(mod, inp, out):
        captured["row"] = out.detach().clone()
    h1 = model.col_embedder.register_forward_hook(hook_col)
    h2 = model.row_interactor.register_forward_hook(hook_row)
    _ = clf.predict_proba(X[100:110])
    h1.remove()
    h2.remove()
    col = captured["col"]
    row = captured["row"]
    print(f"\n  col_embedder output: {col.shape}")
    print(f"    (B={col.shape[0]}, T={col.shape[1]}, F+C={col.shape[2]}, D={col.shape[3]})")
    print(f"    F+C = {col.shape[2]-4} features + 4 CLS tokens")
    print(f"    >>> THIS is the injection point")
    print(f"\n  row_interactor output: {row.shape}")
    print(f"    (B={row.shape[0]}, T={row.shape[1]}, D={row.shape[2]})")
    print(f"    >>> Features already fused, cannot inject per-feature KG here")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TabICL + Real KG Experiment")
    p.add_argument("--n_estimators", type=int, default=4)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--scale", type=float, default=-2.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="kg_real_comparison.png")
    p.add_argument("--kg_epochs", type=int, default=30)
    p.add_argument("--leukemia_d", type=int, default=1000)
    p.add_argument("--skip_leukemia", action="store_true")
    p.add_argument("--demo_hook", action="store_true")
    args = p.parse_args()
    if args.demo_hook:
        demo_hook(args.device)
    else:
        run_experiment(args)
