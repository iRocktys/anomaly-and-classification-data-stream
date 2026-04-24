import csv
import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.*")

class Metrics:
    COMMON_PARAM_NAMES = {
        "threshold",
        "anomaly_threshold",
        "dynamic_threshold",
    }

    AUXILIARY_PARAM_NAMES = {
        "z",
        "u",
        "std",
        "mu",
        "warmup_mean",
        "warmup_std",
        "calculated_threshold",
    }

    MODEL_PARAM_MAP = {
        # Detectores de anomalia
        "HalfSpaceTrees": [
            "window_size",
            "number_of_trees",
            "max_depth",
            "size_limit",
            "anomaly_threshold",
            "threshold",
        ],
        "OnlineIsolationForest": [
            "num_trees",
            "max_leaf_samples",
            "growth_criterion",
            "subsample",
            "window_size",
            "branching_factor",
            "split",
            "n_jobs",
            "threshold",
        ],
        "Autoencoder": [
            "hidden_layer",
            "learning_rate",
            "threshold",
        ],
        "RobustRandomCutForest": [
            "tree_size",
            "n_trees",
            "random_state",
            "threshold",
        ],
        "AdaptiveIsolationForest": [
            "window_size",
            "n_trees",
            "height",
            "seed",
            "m_trees",
            "weights",
            "threshold",
        ],
        # Classificadores
        "LeveragingBagging": [
            "base_learner",
            "ensemble_size",
            "minibatch_size",
            "number_of_jobs",
        ],
        "HoeffdingAdaptiveTree": [
            "grace_period",
            "split_criterion",
            "confidence",
            "tie_threshold",
            "leaf_prediction",
            "nb_threshold",
            "numeric_attribute_observer",
            "binary_split",
            "max_byte_size",
            "memory_estimate_period",
            "stop_mem_management",
            "remove_poor_attrs",
            "disable_prepruning",
        ],
        "AdaptiveRandomForest": [
            "base_learner",
            "ensemble_size",
            "max_features",
            "lambda_param",
            "minibatch_size",
            "number_of_jobs",
            "drift_detection_method",
            "warning_detection_method",
            "disable_weighted_vote",
            "disable_drift_detection",
            "disable_background_learner",
        ],
        "HoeffdingTree": [
            "grace_period",
            "split_criterion",
            "confidence",
            "tie_threshold",
            "leaf_prediction",
            "nb_threshold",
            "numeric_attribute_observer",
            "binary_split",
            "max_byte_size",
            "memory_estimate_period",
            "stop_mem_management",
            "remove_poor_attrs",
            "disable_prepruning",
        ],
    }

    MODEL_ALIASES = {
        "HST": "HalfSpaceTrees",
        "OIF": "OnlineIsolationForest",
        "AE": "Autoencoder",
        "RRCF": "RobustRandomCutForest",
        "AIF": "AdaptiveIsolationForest",
        "LB": "LeveragingBagging",
        "HAT": "HoeffdingAdaptiveTree",
        "ARF": "AdaptiveRandomForest",
        "HT": "HoeffdingTree",
    }

    def __init__(self):
        self.all_possible_params = sorted(
            {
                param
                for params in self.MODEL_PARAM_MAP.values()
                for param in params
            }
            | self.COMMON_PARAM_NAMES
            | self.AUXILIARY_PARAM_NAMES
        )

    def get_metric_classifier(self, metrics_dict, metric_name):
        norm_metrics = {str(k).lower(): v for k, v in metrics_dict.items()}
        metric_name = str(metric_name).lower()
        val = norm_metrics.get(f"{metric_name}_1")
        return float(val) if val is not None and not np.isnan(val) else 0.0

    def calc_sklearn_metrics(self, y_true, y_pred, target_class=None):
        if len(y_true) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=1)
        prec = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=1)
        rec = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=1)
        mcc = matthews_corrcoef(y_true, y_pred)

        y_t_arr = np.array(y_true)
        y_p_arr = np.array(y_pred)
        fp = float(np.sum((y_p_arr == 1) & (y_t_arr == 0)))
        fn = float(np.sum((y_p_arr == 0) & (y_t_arr == 1)))

        return f1 * 100.0, prec * 100.0, rec * 100.0, mcc, fp, fn

    def extract_attack_regions(self, y_true_multi, normal_class_idx=0):
        y_true_array = np.array(y_true_multi)
        attack_indices = np.where(y_true_array != normal_class_idx)[0]

        attack_regions = []
        if len(attack_indices) > 0:
            start_idx = attack_indices[0]
            last_idx = attack_indices[0]
            for idx in attack_indices[1:]:
                if idx - last_idx > 1000:
                    block_labels = y_true_array[start_idx:last_idx + 1]
                    block_attack_labels = block_labels[block_labels != normal_class_idx]
                    block_label = np.bincount(block_attack_labels).argmax() if len(block_attack_labels) > 0 else 1
                    attack_regions.append((start_idx, last_idx, block_label))
                    start_idx = idx
                last_idx = idx

            block_labels = y_true_array[start_idx:last_idx + 1]
            block_attack_labels = block_labels[block_labels != normal_class_idx]
            block_label = np.bincount(block_attack_labels).argmax() if len(block_attack_labels) > 0 else 1
            attack_regions.append((start_idx, last_idx, block_label))

        return attack_regions

    def _normalize_model_name(self, model_name: str) -> str:
        return self.MODEL_ALIASES.get(str(model_name), str(model_name))

    def _is_simple_value(self, value: Any) -> bool:
        return value is None or isinstance(value, (int, float, str, bool, np.integer, np.floating, np.bool_))

    def _format_value(self, value: Any, decimal_comma: bool = True) -> str:
        if value is None:
            text = "N/A"
        elif isinstance(value, (np.integer,)):
            text = str(int(value))
        elif isinstance(value, (np.floating, float)):
            if np.isnan(value):
                text = "N/A"
            else:
                text = f"{float(value):.4f}"
        elif isinstance(value, bool):
            text = str(value)
        else:
            text = str(value)

        return text.replace(".", ",") if decimal_comma else text

    def _safe_ceil_int(self, value: Any) -> int:
        try:
            if value is None or np.isnan(value):
                return 0
        except TypeError:
            pass
        return int(np.ceil(float(value)))

    def _split_experiment_name(self, experiment_name: str) -> Tuple[str, str]:
        parts = str(experiment_name).split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return str(experiment_name), "N/A"

    def _resolve_output_dir(self, model_name: str, strategy_name: str) -> str:
        output_dir = os.path.join("output", self._normalize_model_name(model_name), str(strategy_name))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _clean_params(self, params_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not params_dict:
            return {}
        return {str(k): v for k, v in params_dict.items() if self._is_simple_value(v)}

    def _extract_threshold_metadata(
        self,
        params_dict: Dict[str, Any],
        discretization: Any,
    ) -> Tuple[Any, Any, Any, Any]:
        z_value = params_dict.get("z", "N/A")
        warmup_mean = params_dict.get("u", params_dict.get("mu", params_dict.get("warmup_mean", "N/A")))
        warmup_std = params_dict.get("std", params_dict.get("warmup_std", "N/A"))
        calculated_threshold = params_dict.get("calculated_threshold", discretization)
        return z_value, warmup_mean, warmup_std, calculated_threshold

    def _filter_model_params(self, model_name: str, params_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        clean_params = self._clean_params(params_dict)
        if not clean_params:
            return {}

        normalized_model_name = self._normalize_model_name(model_name)
        allowed = self.MODEL_PARAM_MAP.get(normalized_model_name)

        excluded = set(self.AUXILIARY_PARAM_NAMES)
        if allowed is None:
            return {k: v for k, v in clean_params.items() if k not in excluded}

        allowed_set = set(allowed) | self.COMMON_PARAM_NAMES
        return {
            k: v
            for k, v in clean_params.items()
            if k in allowed_set and k not in excluded
        }

    def _write_csv_row(self, csv_file_path: str, headers: List[str], row: List[Any]) -> None:
        file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not file_exists:
                writer.writerow(headers)
            writer.writerow([self._format_value(x) for x in row])

    def _write_csv_rows(self, csv_file_path: str, headers: List[str], rows: Iterable[List[Any]]) -> None:
        rows = list(rows)
        if not rows:
            return

        file_exists = os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            if not file_exists:
                writer.writerow(headers)
            for row in rows:
                writer.writerow([self._format_value(x) for x in row])

    def _get_metric_pair(self, data: Dict[str, Any], key_mean: str, key_std: str, idx: int) -> Tuple[Any, Any]:
        mean_values = data.get(key_mean, [])
        std_values = data.get(key_std, [])
        mean_value = mean_values[idx] if idx < len(mean_values) else 0.0
        std_value = std_values[idx] if idx < len(std_values) else 0.0
        return mean_value, std_value

    def save_cumulative_metrics_csv(
        self,
        model_name: str,
        data: Dict[str, Any],
        params_dict: Optional[Dict[str, Any]],
        experiment_name: str,
        scenario_name: str,
        discretization: Any,
        window_evaluation: Any,
        exec_id: Any,
        warmup_instances: int,
        strategy_name: str,
        task_type: str,
    ) -> str:
        category, contamination_block = self._split_experiment_name(experiment_name)
        clean_params = self._clean_params(params_dict)
        model_params = self._filter_model_params(model_name, clean_params)
        z_value, warmup_mean, warmup_std, calculated_threshold = self._extract_threshold_metadata(clean_params, discretization)

        output_dir = self._resolve_output_dir(model_name, strategy_name)
        csv_file_path = os.path.join(output_dir, f"{self._normalize_model_name(model_name)}_{scenario_name}.csv")

        base_headers = [
            "Exec_ID",
            "Dataset",
            "Category",
            "Contamination_Block",
            "Task_Type",
            "Model",
            "Scenario",
            "Strategy",
            "Warmup",
            "Win_Evaluation",
            "Discretization",
            "Z",
            "Score_Mean",
            "Score_Std",
            "Calculated_Threshold",
            "F1_avg",
            "F1_std",
            "Prec_avg",
            "Prec_std",
            "Rec_avg",
            "Rec_std",
            "MCC_avg",
            "MCC_std",
            "FP_avg",
            "FP_std",
            "FN_avg",
            "FN_std",
            "Time_avg",
            "Time_std",
        ]
        param_headers = [f"param_{p}" for p in model_params.keys()]
        headers = base_headers + param_headers

        if "cumulative" in data:
            f1_m, f1_s = data["cumulative"].get("f1", (0.0, 0.0))
            prec_m, prec_s = data["cumulative"].get("prec", (0.0, 0.0))
            rec_m, rec_s = data["cumulative"].get("rec", (0.0, 0.0))
            mcc_m, mcc_s = data["cumulative"].get("mcc", (0.0, 0.0))
            fp_m, fp_s = data["cumulative"].get("fp", (0.0, 0.0))
            fn_m, fn_s = data["cumulative"].get("fn", (0.0, 0.0))
            tm_m = data.get("exec_time_mean", data.get("exec_time", 0.0))
            tm_s = data.get("exec_time_std", 0.0)
        else:
            f1_m, prec_m, rec_m, mcc_m, fp_m, fn_m = self.calc_sklearn_metrics(data.get("y_true", []), data.get("y_pred", []))
            f1_s = prec_s = rec_s = mcc_s = fp_s = fn_s = 0.0
            tm_m = data.get("exec_time", 0.0)
            tm_s = 0.0

        row = [
            exec_id,
            experiment_name,
            category,
            contamination_block,
            task_type,
            self._normalize_model_name(model_name),
            scenario_name,
            strategy_name,
            warmup_instances,
            window_evaluation if window_evaluation is not None else "N/A",
            discretization if discretization is not None else "N/A",
            z_value,
            warmup_mean,
            warmup_std,
            calculated_threshold if calculated_threshold is not None else "N/A",
            f1_m,
            f1_s,
            prec_m,
            prec_s,
            rec_m,
            rec_s,
            mcc_m,
            mcc_s,
            self._safe_ceil_int(fp_m),
            self._safe_ceil_int(fp_s),
            self._safe_ceil_int(fn_m),
            self._safe_ceil_int(fn_s),
            tm_m,
            tm_s,
        ] + list(model_params.values())

        self._write_csv_row(csv_file_path, headers, row)
        return csv_file_path

    def save_prequential_metrics_csv(
        self,
        model_name: str,
        data: Dict[str, Any],
        experiment_name: str,
        scenario_name: str,
        window_evaluation: Any,
        exec_id: Any,
        warmup_instances: int,
        strategy_name: str,
        task_type: str,
    ) -> Optional[str]:
        if window_evaluation is None:
            return None

        instances = list(data.get("instances", []))
        if not instances:
            return None

        output_dir = self._resolve_output_dir(model_name, strategy_name)
        prequential_dir = os.path.join(output_dir, "prequential")
        os.makedirs(prequential_dir, exist_ok=True)

        normalized_model_name = self._normalize_model_name(model_name)
        csv_file_path = os.path.join(
            prequential_dir,
            f"{normalized_model_name}_{scenario_name}_window_{window_evaluation}.csv",
        )

        headers = [
            "Exec_ID",
            "Dataset",
            "Task_Type",
            "Model",
            "Scenario",
            "Strategy",
            "Warmup",
            "Window_Size",
            "Window_Index",
            "Instance",
            "F1_avg",
            "F1_std",
            "Prec_avg",
            "Prec_std",
            "Rec_avg",
            "Rec_std",
            "FP_avg",
            "FP_std",
            "FN_avg",
            "FN_std",
        ]

        rows = []
        for idx, instance in enumerate(instances):
            f1_m, f1_s = self._get_metric_pair(data, "f1_mean", "f1_std", idx)
            prec_m, prec_s = self._get_metric_pair(data, "precision_mean", "precision_std", idx)
            rec_m, rec_s = self._get_metric_pair(data, "recall_mean", "recall_std", idx)
            fp_m, fp_s = self._get_metric_pair(data, "fp_mean", "fp_std", idx)
            fn_m, fn_s = self._get_metric_pair(data, "fn_mean", "fn_std", idx)

            rows.append([
                exec_id,
                experiment_name,
                task_type,
                normalized_model_name,
                scenario_name,
                strategy_name,
                warmup_instances,
                window_evaluation,
                idx + 1,
                instance,
                f1_m,
                f1_s,
                prec_m,
                prec_s,
                rec_m,
                rec_s,
                self._safe_ceil_int(fp_m),
                self._safe_ceil_int(fp_s),
                self._safe_ceil_int(fn_m),
                self._safe_ceil_int(fn_s),
            ])

        self._write_csv_rows(csv_file_path, headers, rows)
        return csv_file_path

    def display_cumulative_metrics(
        self,
        predictions_history,
        warmup_instances=0,
        target_class=None,
        n_runs=1,
        params_dict=None,
        experiment_name="General",
        scenario_name="Default_Full",
        discretization=None,
        window_evaluation=None,
        exec_id="N/A",
        discretization_strategy="fixed",
        task_type=None,
        save_prequential=True,
    ):
        if task_type is None:
            task_type = "classification" if str(discretization).upper() == "N/A" else "anomaly"

        if task_type == "classification" and discretization_strategy == "fixed":
            discretization_strategy = "classifier"

        titulo_relatorio = f"METRICS REPORT | {scenario_name.upper()}"
        discretization_str = self._format_value(discretization, decimal_comma=False) if discretization is not None else "N/A"
        window_eval_str = str(window_evaluation) if window_evaluation is not None else "N/A"

        header_base = (
            f"{'Algorithm':<24} | {'F1 (%)':<17} | {'Prec (%)':<17} | {'Rec (%)':<17} | "
            f"{'MCC':<17} | {'FP':<8} | {'FN':<8} | {'Time (s)':<11} | {'Task':<14} | "
            f"{'Strategy':<12} | {'Win_Eval':<10}"
        )
        line_len = max(len(header_base), len(titulo_relatorio) + 4)

        print(f"\n{'=' * line_len}\n{titulo_relatorio:^{line_len}}\n{'=' * line_len}\n{header_base}\n{'-' * line_len}")

        cumulative_paths = []
        prequential_paths = []

        for name, data in predictions_history.items():
            if "cumulative" in data:
                f1_m, f1_s = data["cumulative"].get("f1", (0.0, 0.0))
                prec_m, prec_s = data["cumulative"].get("prec", (0.0, 0.0))
                rec_m, rec_s = data["cumulative"].get("rec", (0.0, 0.0))
                mcc_m, mcc_s = data["cumulative"].get("mcc", (0.0, 0.0))
                fp_m, _ = data["cumulative"].get("fp", (0.0, 0.0))
                fn_m, _ = data["cumulative"].get("fn", (0.0, 0.0))
                tm_m = data.get("exec_time_mean", data.get("exec_time", 0.0))

                print(
                    f"{name:<24} | {f1_m:>7.4f} ± {f1_s:<6.4f} | "
                    f"{prec_m:>7.4f} ± {prec_s:<6.4f} | {rec_m:>7.4f} ± {rec_s:<6.4f} | "
                    f"{mcc_m:>7.4f} ± {mcc_s:<6.4f} | {self._safe_ceil_int(fp_m):>4} | "
                    f"{self._safe_ceil_int(fn_m):>4} | {float(tm_m):>9.4f} | {task_type:<14} | "
                    f"{str(discretization_strategy):<12} | {window_eval_str:<10}"
                )
            else:
                f1, pr, re, mcc, fp, fn = self.calc_sklearn_metrics(data.get("y_true", []), data.get("y_pred", []))
                tm = data.get("exec_time", 0.0)
                print(
                    f"{name:<24} | {f1:<17.4f} | {pr:<17.4f} | {re:<17.4f} | "
                    f"{mcc:<17.4f} | {self._safe_ceil_int(fp):<8} | {self._safe_ceil_int(fn):<8} | "
                    f"{float(tm):<11.4f} | {task_type:<14} | {str(discretization_strategy):<12} | {window_eval_str:<10}"
                )

            cumulative_path = self.save_cumulative_metrics_csv(
                model_name=name,
                data=data,
                params_dict=params_dict,
                experiment_name=experiment_name,
                scenario_name=scenario_name,
                discretization=discretization_str,
                window_evaluation=window_evaluation,
                exec_id=exec_id,
                warmup_instances=warmup_instances,
                strategy_name=discretization_strategy,
                task_type=task_type,
            )
            cumulative_paths.append(cumulative_path)

            if save_prequential:
                prequential_path = self.save_prequential_metrics_csv(
                    model_name=name,
                    data=data,
                    experiment_name=experiment_name,
                    scenario_name=scenario_name,
                    window_evaluation=window_evaluation,
                    exec_id=exec_id,
                    warmup_instances=warmup_instances,
                    strategy_name=discretization_strategy,
                    task_type=task_type,
                )
                if prequential_path:
                    prequential_paths.append(prequential_path)

        print(f"{'-' * line_len}")
        if cumulative_paths:
            print("Cumulative CSV:")
            for path in cumulative_paths:
                print(f" - {path}")
        if prequential_paths:
            print("Prequential CSV:")
            for path in prequential_paths:
                print(f" - {path}")
        print()
