# sourcescripts/test_only.py
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import dgl
from dgl import load_graphs

from utils.preprocessdata import Dataset
import utils.utills as imp
from model import LitGNN

from sklearn.metrics import (
    f1_score, precision_score, accuracy_score, recall_score,
    precision_recall_curve, auc
)

torch.set_float32_matmul_precision("medium")


class VulnerabilityTester:
    def __init__(self, checkpoint_path, output_dir, gtype="pdg+raw", splits="default", root=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.gtype = gtype
        self.splits = splits
        self.root = root

        os.makedirs(self.output_dir, exist_ok=True)

        self.function_data = []
        self.statement_data = []
        self.cwe_data = {}

    # ---------- MODEL ----------
    def load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint non trovato: {self.checkpoint_path}")
        print(f">>> Loading model from: {self.checkpoint_path}")
        self.model = LitGNN.load_from_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    # ---------- DATA ----------
    def prepare_test_data(self):
        print(">>> Preparing test graphs...")

        if self.root:
            # load splits from archive
            pq = os.path.join(self.root, "minimal_datasets", "minimal_Dataset_False.pq")
            if not os.path.isfile(pq):
                raise FileNotFoundError(f"Split arquet not found: {pq}")
            df = pd.read_parquet(pq, engine="fastparquet")

            # graphs from archive
            gpath = os.path.join(self.root, f"Dataset_Vvuldet_codebert_{self.gtype}")
        else:
            # fallback: standard mechanism
            df = Dataset(splits=self.splits)
            gpath = f"{imp.cache_dir()}/Dataset_Vvuldet_codebert_{self.gtype}"

        # dataset signature
        ds_name = None
        for col in ("dataset", "source", "corpus", "project", "repo"):
            if col in df.columns and df[col].nunique() > 0:
                ds_name = df[col].mode().iat[0]
                break
        if ds_name is None:
            ds_name = os.path.basename(str(imp.processed_dir())) or "unknown"
        print(f">>> Dataset used: {ds_name}")

        dftest = df[df["label"] == "test"].copy()
        print(f">>> Test samples: {len(dftest)}")

        if not os.path.isdir(gpath):
            raise FileNotFoundError(f"Graph directory not found:{gpath}")

        test_ids = set(dftest["id"].tolist())
        self.test_graphs = []
        self.dataset_df = dftest.set_index("id")

        for fname in tqdm(os.listdir(gpath), desc="Load test graphs"):
            try:
                _id = int(fname)
            except ValueError:
                continue
            if _id in test_ids:
                try:
                    g = load_graphs(os.path.join(gpath, fname))[0][0]
                    g.path = os.path.join(gpath, fname)
                    self.test_graphs.append(g)
                except Exception as e:
                    print(f"Error loading graph {_id}: {e}")

        print(f">>> Loaded test graphs: {len(self.test_graphs)}")


    # ---------- INFERENCE ----------
    @torch.no_grad()
    def analyze(self):
        print(">>> Start analysis (test only)...")
        for g in tqdm(self.test_graphs, desc="Analize"):
            self._process_graph(g)
        self._save_outputs()

    def _process_graph(self, graph):
        graph_id = int(os.path.basename(graph.path))
        # retrieve dataset row for graph_id
        if graph_id not in self.dataset_df.index:
            return
        row = self.dataset_df.loc[graph_id]

        graph = graph.to(self.device)
        logits, _, _ = self.model.shared_step(graph, test=True)

        # statement-level
        node_probs = torch.softmax(logits[0], dim=1).detach().cpu().numpy()
        node_preds = np.argmax(node_probs, axis=1)

        # function-level
        func_prob = torch.softmax(logits[1], dim=1).detach().cpu().numpy()[0]
        func_pred = int(np.argmax(func_prob))

        node_labels = graph.ndata["_VULN"].detach().cpu().numpy()
        func_label = int(graph.ndata["_FVULN"][0].item())
        line_numbers = graph.ndata["_LINE"].detach().cpu().numpy()

        # collectors
        self.function_data.append({
            "CWE_ID": row.get("CWE_ID", ""),
            "func_id": graph_id,
            "func_label": func_label,
            "func_pred": func_pred,
            "func_prob_0": float(func_prob[0]),
            "func_prob_1": float(func_prob[1]),
        })

        for i in range(len(node_labels)):
            self.statement_data.append({
                "CWE_ID": row.get("CWE_ID", ""),
                "func_id": graph_id,
                "line_number": int(line_numbers[i]),
                "node_label": int(node_labels[i]),
                "node_pred": int(node_preds[i]),
                "node_prob_0": float(node_probs[i][0]),
                "node_prob_1": float(node_probs[i][1]),
            })

        cwe = row.get("CWE_ID", "")
        store = self.cwe_data.setdefault(cwe, {
            "func_labels": [], "func_preds": [], "func_probs": [],
            "node_labels": [], "node_preds": [], "node_probs": []
        })
        store["func_labels"].append(func_label)
        store["func_preds"].append(func_pred)
        store["func_probs"].append(float(func_prob[1]))
        store["node_labels"].extend(node_labels.tolist())
        store["node_preds"].extend(node_preds.tolist())
        store["node_probs"].extend(node_probs[:, 1].tolist())

    # ---------- SAVES ----------
    def _save_outputs(self):
        os.makedirs(self.output_dir, exist_ok=True)

        # function-level CSV + lines for statement
        func_df = pd.DataFrame(self.function_data)
        stmt_df = pd.DataFrame(self.statement_data)

        if not stmt_df.empty:
            stmt_grouped = stmt_df.groupby(["CWE_ID", "func_id"]).agg({
                "line_number": list, "node_label": list, "node_pred": list
            }).reset_index()
            merged_df = pd.merge(func_df, stmt_grouped, on=["CWE_ID", "func_id"], how="left")
        else:
            merged_df = func_df.copy()

        out_func_pred = os.path.join(self.output_dir, "function_level_predictions.csv")
        merged_df.to_csv(out_func_pred, index=False)
        print(f">>> Saved: {out_func_pred}")

        # metrics for-CWE
        rows = []
        for cwe, data in self.cwe_data.items():
            rows.append({
                "CWE_ID": cwe,
                "num_functions": len(data["func_labels"]),
                "num_statements": len(data["node_labels"]),
                **{f"func_{k}": v for k, v in self._metrics(data["func_labels"], data["func_preds"]).items()},
                **{f"stmt_{k}": v for k, v in self._metrics(data["node_labels"], data["node_preds"]).items()},
                "func_pr_auc": self._pr_auc(data["func_labels"], data["func_probs"]),
                "stmt_pr_auc": self._pr_auc(data["node_labels"], data["node_probs"]),
            })
        cwe_df = pd.DataFrame(rows)
        out_cwe = os.path.join(self.output_dir, "cwe_level_metrics.csv")
        cwe_df.to_csv(out_cwe, index=False)
        print(f">>> Saved: {out_cwe}")

        # global metrics 
        if not func_df.empty:
            func_metrics = self._metrics(func_df["func_label"], func_df["func_pred"])
            func_metrics["pr_auc"] = self._pr_auc(
                func_df["func_label"].tolist(),
                func_df["func_prob_1"].tolist()
            )
            pd.DataFrame([func_metrics]).to_csv(
                os.path.join(self.output_dir, "function_level_metrics.csv"),
                index=False
            )

        if not stmt_df.empty:
            stmt_metrics = self._metrics(stmt_df["node_label"], stmt_df["node_pred"])
            stmt_metrics["pr_auc"] = self._pr_auc(
                stmt_df["node_label"].tolist(),
                stmt_df["node_prob_1"].tolist()
            )
            pd.DataFrame([stmt_metrics]).to_csv(
                os.path.join(self.output_dir, "statement_level_metrics.csv"),
                index=False
            )

        print(">>> Analysis completed.")

    # ---------- METRICS ----------
    @staticmethod
    def _metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    @staticmethod
    def _pr_auc(y_true, y_score):
        if len(y_true) == 0:
            return float("nan")
        try:
            p, r, _ = precision_recall_curve(y_true, y_score)
            return float(auc(r, p))
        except Exception:
            return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                help="Root of an archived run (will use its splits and graphs)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint .ckpt")
    ap.add_argument("--out", type=str, default=str(imp.outputs_dir()),
                    help="Output folder")
    ap.add_argument("--gtype", type=str, default="pdg+raw",
                    help="Graph type (must match cached graphs)")
    ap.add_argument("--splits", type=str, default="default",
                    help="Split scheme (default, crossproject_*, ecc.)")
    args = ap.parse_args()

    # default checkpoint if not passed
    ckpt = args.checkpoint
    if ckpt is None:
        if args.root:
            ckpt = os.path.join(args.root, "checkpoints", "model-checkpoint.ckpt")
        else:
            ckpt = os.path.join(args.out, "checkpoints", "model-checkpoint.ckpt")


    tester = VulnerabilityTester(ckpt, args.out, gtype=args.gtype, splits=args.splits, root=args.root)
    tester.load_model()
    tester.prepare_test_data()
    tester.analyze()


if __name__ == "__main__":
    main()
