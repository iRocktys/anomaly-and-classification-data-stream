import json
import os
import re
from typing import Dict, Optional, Sequence, Tuple, Union
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

class PlotsBestModels:
    def __init__(
        self,
        output_dir: str = "output/Results/plots",
        metadata_path: str = "data/attack_regions_metadata.json",
    ):
        self.output_dir = output_dir
        self.metadata_path = metadata_path
        self.classifier_colors = ["#0B5FA5", "#C0392B", "#1565C0", "#8E44AD"]
        self.anomaly_colors = ["#1B5E20", "#BF360C", "#4A148C", "#00695C"]
        self.bg_colors = ["#F5B041", "#EC7063", "#AF7AC5", "#48C9B0", "#EB984E"]
        self.scenario_map = {
            "Consistência": "Consistency",
            "Consistencia": "Consistency",
            "Generalização": "Generalization",
            "Generalizacao": "Generalization",
            "Adaptação": "Adaptation",
            "Adaptacao": "Adaptation",
            "Recorrência": "Recurrence",
            "Recorrencia": "Recurrence",
        }
        self.model_aliases = {
            "AdaptiveIsolationForest": "AIF",
            "HalfSpaceTrees": "HST",
            "Autoencoder": "AE",
            "OnlineIsolationForest": "OIF",
            "RobustRandomCutForest": "RRCF",
            "LeveragingBagging": "LB",
            "HoeffdingAdaptiveTree": "HAT",
            "AdaptiveRandomForest": "ARF",
            "AdaptiveRandomForestClassifier": "ARF",
            "HoeffdingTree": "HT",
            "AIF": "AIF",
            "HST": "HST",
            "AE": "AE",
            "OIF": "OIF",
            "RRCF": "RRCF",
            "LB": "LB",
            "HAT": "HAT",
            "ARF": "ARF",
            "HT": "HT",
        }

    def _read_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    def _clean_filename(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
        return text.strip("_") or "plot"

    def _ensure_dir(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def _extract_exec_datetime(self, value):
        text = str(value).strip()
        patterns = [
            (r"(\d{8})[_\-T\s]?(\d{4})$", "%Y%m%d%H%M"),
            (r"(\d{8})[_\-T\s]?(\d{6})$", "%Y%m%d%H%M%S"),
            (r"(\d{4})[-_/](\d{2})[-_/](\d{2})[T_\s-]+(\d{2})[:hH_-](\d{2})(?:[:_-](\d{2}))?", None),
            (r"(\d{2})[-_/](\d{2})[-_/](\d{4})[T_\s-]+(\d{2})[:hH_-](\d{2})(?:[:_-](\d{2}))?", None),
        ]
        match = re.search(patterns[0][0], text)
        if match:
            return pd.to_datetime("".join(match.groups()), format=patterns[0][1], errors="coerce")
        match = re.search(patterns[1][0], text)
        if match:
            return pd.to_datetime("".join(match.groups()), format=patterns[1][1], errors="coerce")
        match = re.search(patterns[2][0], text)
        if match:
            year, month, day, hour, minute, second = match.groups()
            return pd.Timestamp(int(year), int(month), int(day), int(hour), int(minute), int(second or 0))
        match = re.search(patterns[3][0], text)
        if match:
            day, month, year, hour, minute, second = match.groups()
            return pd.Timestamp(int(year), int(month), int(day), int(hour), int(minute), int(second or 0))
        return pd.NaT

    def _filter_latest_rows_by_exec_id(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.reset_index(drop=True)
        if "Exec_ID" not in df.columns:
            return df.drop_duplicates().reset_index(drop=True)
        df = df.copy()
        df["_exec_datetime"] = df["Exec_ID"].apply(self._extract_exec_datetime).fillna(pd.Timestamp.min)
        df["_exec_order"] = np.arange(len(df))
        group_cols = [
            col for col in [
                "Dataset",
                "Category",
                "Contamination_Block",
                "Task_Type",
                "Model",
                "Scenario",
                "Strategy",
                "Threshold_Strategy",
                "Decision_Strategy",
                "Decision_Window",
                "Persistence_K",
                "Persistence_N",
                "Warmup",
            ]
            if col in df.columns
        ]
        if group_cols:
            df = df.sort_values(["_exec_datetime", "_exec_order"]).groupby(group_cols, dropna=False, as_index=False).tail(1)
        else:
            df = df.sort_values(["_exec_datetime", "_exec_order"]).tail(1)
        return df.drop(columns=["_exec_datetime", "_exec_order"], errors="ignore").drop_duplicates().reset_index(drop=True)

    def _scenario_base(self, value) -> str:
        value = str(value).strip()
        return self.scenario_map.get(value, value)

    def _dataset_english_name(self, dataset_name: str) -> str:
        parts = str(dataset_name).split("_")
        if len(parts) < 2:
            return self._scenario_base(dataset_name)
        return f"{self._scenario_base(parts[0])}_{parts[-1]}"

    def _model_alias(self, value) -> str:
        text = str(value).strip()
        return self.model_aliases.get(text, text)

    def _feature_label(self, value) -> str:
        text = str(value).strip().lower()
        if "33features" in text or "33_features" in text or "bestfeatures" in text or "best_features" in text:
            return "Best Features"
        if "fullfeatures" in text or "full_features" in text:
            return "Full Features"
        return "Features N/A"

    def _format_threshold(self, value, task_type: str) -> str:
        if task_type == "classifier":
            return ""
        text = str(value).strip()
        mapping = {
            "z_score": "z-score",
            "dinamic": "optimized fixed",
            "dynamic": "optimized fixed",
            "fixed": "fixed",
            "params": "parameters",
        }
        return mapping.get(text.lower(), text.replace("_", " "))

    def _format_decision(self, value, task_type: str) -> str:
        if task_type == "classifier":
            return ""
        text = str(value).strip()
        mapping = {
            "raw": "raw",
            "moving_average_w3": "moving average w3",
            "moving_average_w5": "moving average w5",
            "persistence_2_of_3": "persistence 2 of 3",
            "persistence_3_of_5": "persistence 3 of 5",
        }
        return mapping.get(text.lower(), text.replace("_", " "))

    def _task_type_label(self, value) -> str:
        text = str(value).strip().lower()
        if text in ["classification", "classifier", "supervised"]:
            return "classifier"
        if text in ["anomaly", "detector", "anomaly_detector"]:
            return "anomaly"
        return text

    def _label_is_normal(self, value) -> bool:
        text = str(value).strip().lower()
        return text in ["benign", "normal", "0", "b'benign'", "b'normal'"]

    def _clean_attack_name(self, value) -> str:
        text = str(value).strip()
        text = re.sub(r"(?i)^drdos[_\-\s]*", "", text)
        text = re.sub(r"(?i)^ddos[_\-\s]*", "", text)
        text = text.replace("_", " ").strip()
        aliases = {
            "Syn": "SYN",
            "LDAP": "LDAP",
            "DNS": "DNS",
        }
        return aliases.get(text, text)

    def _extract_attack_regions_from_dataframe(self, df: pd.DataFrame, label_col: str) -> Tuple[list, list]:
        labels = df[label_col].astype(str).tolist()
        target_names = sorted({self._clean_attack_name(v) for v in labels if not self._label_is_normal(v)})
        regions = []
        start = None
        current_label = None
        current_idx = None
        attack_name_to_idx = {name: idx + 1 for idx, name in enumerate(target_names)}
        for idx, label in enumerate(labels):
            is_attack = not self._label_is_normal(label)
            clean_label = self._clean_attack_name(label) if is_attack else None
            if is_attack and start is None:
                start = idx
                current_label = clean_label
                current_idx = attack_name_to_idx.get(clean_label, 1)
            elif is_attack and clean_label != current_label:
                regions.append({
                    "start": int(start),
                    "end": int(idx - 1),
                    "attack_idx": int(current_idx),
                    "attack_name": str(current_label),
                })
                start = idx
                current_label = clean_label
                current_idx = attack_name_to_idx.get(clean_label, 1)
            elif not is_attack and start is not None:
                regions.append({
                    "start": int(start),
                    "end": int(idx - 1),
                    "attack_idx": int(current_idx),
                    "attack_name": str(current_label),
                })
                start = None
                current_label = None
                current_idx = None
        if start is not None:
            regions.append({
                "start": int(start),
                "end": int(len(labels) - 1),
                "attack_idx": int(current_idx),
                "attack_name": str(current_label),
            })
        return regions, target_names

    def build_attack_metadata(
        self,
        data_root: str = "data/15k",
        label_col: str = "Label",
        output_path: Optional[str] = None,
        csv_sep: str = ",",
    ) -> dict:
        metadata = {}
        for root, _, files in os.walk(data_root):
            for file in files:
                if not file.lower().endswith(".csv"):
                    continue
                path = os.path.join(root, file)
                dataset_name = os.path.splitext(file)[0]
                df = pd.read_csv(path, sep=csv_sep)
                if label_col not in df.columns:
                    raise ValueError(f"Column {label_col} not found in dataset {path}")
                regions, target_names = self._extract_attack_regions_from_dataframe(df, label_col)
                english_name = self._dataset_english_name(dataset_name)
                metadata[dataset_name] = {
                    "dataset": dataset_name,
                    "english_dataset": english_name,
                    "target_names": target_names,
                    "attack_regions": regions,
                    "source_path": path,
                }
                metadata[english_name] = metadata[dataset_name]
        final_path = output_path or self.metadata_path
        self._ensure_dir(os.path.dirname(final_path) or ".")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata

    def load_attack_metadata(self, metadata_path: Optional[str] = None) -> dict:
        final_path = metadata_path or self.metadata_path
        if not os.path.exists(final_path):
            return {}
        with open(final_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _attack_legend_handles(self, attack_regions: Sequence[dict]) -> list:
        handles = []
        seen = set()
        for region in attack_regions:
            attack_idx = int(region.get("attack_idx", 1))
            attack_name = str(region.get("attack_name", f"Attack {attack_idx}"))
            if attack_name in seen:
                continue
            color = self.bg_colors[attack_idx % len(self.bg_colors)]
            handles.append(mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.45, label=attack_name))
            seen.add(attack_name)
        return handles

    def _add_attack_regions(self, ax, attack_regions, alpha=0.28):
        if not attack_regions:
            return
        for region in attack_regions:
            start = int(region["start"])
            end = int(region["end"])
            attack_idx = int(region.get("attack_idx", 1))
            bg_color = self.bg_colors[attack_idx % len(self.bg_colors)]
            ax.axvspan(start, end, facecolor=bg_color, alpha=alpha, zorder=1)
            mid = (start + end) / 2
            ax.axvline(mid, color=bg_color, alpha=0.8, linewidth=1.1, zorder=2)

    def _expand_y_limits(self, ax):
        ymin, ymax = ax.get_ylim()
        if ymin == ymax:
            delta = abs(ymax) * 0.1 if ymax != 0 else 1.0
            ax.set_ylim(ymin - delta, ymax + delta)
            return
        span = ymax - ymin
        ax.set_ylim(ymin - 0.03 * span, ymax + max(0.15 * span, 0.08 * max(abs(ymax), 1.0)))

    def _derive_prequential_path(self, cumulative_path: str, window_size: int) -> str:
        directory = os.path.dirname(cumulative_path)
        filename = os.path.basename(cumulative_path)
        base, ext = os.path.splitext(filename)
        return os.path.join(directory, "prequential", f"{base}_window_{window_size}{ext}")

    def build_candidates_dataframe(
        self,
        csv_dict: Dict[str, Optional[Union[str, int]]],
        window_size: int = 100,
        use_exec_filter: bool = True,
        use_latest_when_exec_id_none: bool = True,
    ) -> pd.DataFrame:
        rows = []
        for path, exec_id in csv_dict.items():
            if not os.path.exists(path):
                print(f"[Warning] CSV not found: {path}")
                continue
            df = self._read_csv(path)
            if use_exec_filter and exec_id is not None and "Exec_ID" in df.columns:
                df = df[df["Exec_ID"].astype(str) == str(exec_id)]
            elif use_exec_filter and exec_id is None and use_latest_when_exec_id_none:
                df = self._filter_latest_rows_by_exec_id(df)
            if df.empty:
                continue
            required_cols = ["Dataset", "Task_Type", "Model", "Scenario", "Threshold_Strategy", "Decision_Strategy", "F1_avg", "Prec_avg", "Rec_avg", "FP_avg", "FN_avg"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"CSV {path} does not contain the expected columns: {missing}")
            for _, row in df.iterrows():
                task_type = self._task_type_label(row.get("Task_Type", "anomaly"))
                model = row.get("Model", "N/A")
                threshold = row.get("Threshold_Strategy", row.get("Strategy", "N/A"))
                decision = row.get("Decision_Strategy", "N/A")
                rows.append({
                    "Dataset": row.get("Dataset", "N/A"),
                    "English_Dataset": self._dataset_english_name(row.get("Dataset", "N/A")),
                    "Scenario_Group": self._dataset_english_name(row.get("Dataset", "N/A")).split("_")[0],
                    "Task_Type": task_type,
                    "Model": model,
                    "Model_Alias": self._model_alias(model),
                    "Features": self._feature_label(row.get("Scenario", "N/A")),
                    "Scenario_Config": row.get("Scenario", "N/A"),
                    "Threshold": threshold,
                    "Decision": decision,
                    "Threshold_Label": self._format_threshold(threshold, task_type),
                    "Decision_Label": self._format_decision(decision, task_type),
                    "F1": row.get("F1_avg", 0.0),
                    "Precision": row.get("Prec_avg", 0.0),
                    "Recall": row.get("Rec_avg", 0.0),
                    "FP": row.get("FP_avg", 0.0),
                    "FN": row.get("FN_avg", 0.0),
                    "Exec_ID": row.get("Exec_ID", None),
                    "Cumulative_Path": path,
                    "Prequential_Path": self._derive_prequential_path(path, window_size),
                })
        if not rows:
            return pd.DataFrame()
        df_final = pd.DataFrame(rows)
        for col in ["F1", "Precision", "Recall", "FP", "FN"]:
            df_final[col] = self._to_numeric(df_final[col])
        return df_final.reset_index(drop=True)

    def select_best_models(self, candidates_df: pd.DataFrame, top_n: int = 2) -> pd.DataFrame:
        if candidates_df.empty:
            return candidates_df
        selected = []
        for _, group_df in candidates_df.groupby(["English_Dataset", "Task_Type"], dropna=False):
            group_df = group_df.copy()
            group_df["_rank_fn"] = group_df["FN"].fillna(np.inf)
            group_df["_rank_fp"] = group_df["FP"].fillna(np.inf)
            group_df["_legend_label"] = group_df.apply(lambda row: f"{row['Model_Alias']} - {row['Features']}", axis=1)
            group_df = group_df.sort_values(
                ["F1", "Recall", "Precision", "_rank_fn", "_rank_fp"],
                ascending=[False, False, False, True, True],
            )
            selected.append(group_df.drop_duplicates(subset=["_legend_label"], keep="first").head(top_n))
        return pd.concat(selected, ignore_index=True).drop(columns=["_rank_fn", "_rank_fp"], errors="ignore")

    def _series_label(self, row: pd.Series) -> str:
        return f"{row['Model_Alias']} - {row['Features']}"

    def _read_prequential_series(self, row: pd.Series) -> Optional[pd.DataFrame]:
        path = row["Prequential_Path"]
        if not os.path.exists(path):
            print(f"[Warning] Prequential CSV not found: {path}")
            return None
        df = self._read_csv(path)
        if "Dataset" in df.columns:
            df = df[df["Dataset"].astype(str) == str(row["Dataset"])]
        if "Exec_ID" in df.columns and pd.notnull(row.get("Exec_ID")):
            df = df[df["Exec_ID"].astype(str) == str(row["Exec_ID"])]
        if df.empty:
            return None
        required_cols = ["Instance", "FP_avg", "FN_avg"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Prequential CSV {path} does not contain the expected columns: {missing}")
        for col in ["Instance", "FP_avg", "FP_std", "FN_avg", "FN_std"]:
            if col in df.columns:
                df[col] = self._to_numeric(df[col])
        return df.sort_values("Instance").reset_index(drop=True)

    def plot_dataset_best_models(
        self,
        dataset_name: str,
        selected_df: pd.DataFrame,
        metadata: dict,
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> Optional[str]:
        dataset_df = selected_df[selected_df["English_Dataset"] == dataset_name].copy()
        if dataset_df.empty:
            return None
        scenario_group = dataset_name.split("_")[0]
        final_dir = self._ensure_dir(os.path.join(output_dir or self.output_dir, scenario_group))
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
        axes = axes.flatten()
        panel_specs = [
            ("classifier", "FP_avg", "FP_std", "Classifiers - False Positives (FP)"),
            ("classifier", "FN_avg", "FN_std", "Classifiers - False Negatives (FN)"),
            ("anomaly", "FP_avg", "FP_std", "Anomaly Detectors - False Positives (FP)"),
            ("anomaly", "FN_avg", "FN_std", "Anomaly Detectors - False Negatives (FN)"),
        ]
        meta = metadata.get(dataset_name, metadata.get(str(dataset_df.iloc[0]["Dataset"]), {}))
        attack_regions = meta.get("attack_regions", [])
        attack_handles = self._attack_legend_handles(attack_regions)
        for ax_idx, (task_type, metric_col, std_col, title) in enumerate(panel_specs):
            ax = axes[ax_idx]
            task_rows = dataset_df[dataset_df["Task_Type"] == task_type].reset_index(drop=True)
            palette = self.classifier_colors if task_type == "classifier" else self.anomaly_colors
            for i, (_, row) in enumerate(task_rows.iterrows()):
                preq = self._read_prequential_series(row)
                if preq is None or preq.empty:
                    continue
                color = palette[i % len(palette)]
                label = self._series_label(row)
                x_axis = preq["Instance"].to_numpy(dtype=float)
                y_axis = np.ceil(preq[metric_col].to_numpy(dtype=float))
                ax.plot(x_axis, y_axis, label=label, color=color, linewidth=2.2, zorder=3, marker="o", markersize=4)
                if std_col in preq.columns:
                    y_std = np.ceil(preq[std_col].fillna(0).to_numpy(dtype=float))
                    if np.sum(y_std) > 0:
                        ax.fill_between(x_axis, y_axis - y_std, y_axis + y_std, color=color, alpha=0.12, zorder=2)
            self._expand_y_limits(ax)
            self._add_attack_regions(ax, attack_regions, alpha=0.28)
            ax.grid(True, alpha=0.3, linestyle=":", zorder=0)
            ax.tick_params(axis="both", which="major", labelsize=11)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(metric_col.split("_")[0], fontsize=12)
        axes[2].set_xlabel("Instances", fontsize=12)
        axes[3].set_xlabel("Instances", fontsize=12)
        line_handles, line_labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in line_labels:
                    line_handles.append(handle)
                    line_labels.append(label)
        final_handles = line_handles + [h for h in attack_handles if h.get_label() not in line_labels]
        final_labels = line_labels + [h.get_label() for h in attack_handles if h.get_label() not in line_labels]
        if final_handles:
            legend = fig.legend(
                final_handles,
                final_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(6, len(final_handles)),
                fontsize=13,
                frameon=False,
                handlelength=2.0,
                columnspacing=1.6,
            )
            for text in legend.get_texts():
                text.set_fontweight("bold")
        fig.subplots_adjust(bottom=0.18, top=0.95, hspace=0.30, wspace=0.15)
        file_path = os.path.join(final_dir, f"{self._clean_filename(dataset_name)}_BestModels_FP_FN.png")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return file_path

    def plot_all_best_models(
        self,
        cumulative_csv_dict: Dict[str, Optional[Union[str, int]]],
        data_root: str = "data/15k",
        label_col: str = "Label",
        window_size: int = 100,
        top_n: int = 2,
        metadata_path: Optional[str] = None,
        rebuild_metadata: bool = True,
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> pd.DataFrame:
        final_metadata_path = metadata_path or self.metadata_path
        if rebuild_metadata or not os.path.exists(final_metadata_path):
            metadata = self.build_attack_metadata(data_root=data_root, label_col=label_col, output_path=final_metadata_path)
        else:
            metadata = self.load_attack_metadata(final_metadata_path)
        candidates = self.build_candidates_dataframe(cumulative_csv_dict, window_size=window_size)
        selected = self.select_best_models(candidates, top_n=top_n)
        if selected.empty:
            print("No selected models found.")
            return selected
        generated = []
        scenario_order = {"Consistency": 0, "Generalization": 1, "Adaptation": 2, "Recurrence": 3}
        contamination_order = {"25": 0, "200": 1, "1000": 2}
        datasets = sorted(
            selected["English_Dataset"].unique(),
            key=lambda name: (
                scenario_order.get(str(name).split("_")[0], 99),
                contamination_order.get(str(name).split("_")[-1], 99),
                str(name),
            ),
        )
        for dataset_name in datasets:
            path = self.plot_dataset_best_models(dataset_name, selected, metadata, output_dir=output_dir, show=show)
            if path:
                generated.append(path)
                print(f"Plot saved at: {path}")
        selected = selected.copy()
        selected["Generated_Plots_Count"] = len(generated)
        return selected
