import os
import re
from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class BestResultsTable:
    def __init__(self, output_dir: str = "output/Results"):
        self.output_dir = output_dir
        self.header_color = "#3f456c"
        self.even_row_color = "#f7f7f7"
        self.grid_color = "#777777"
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
        self.poisoning_percentages = {
            "Consistency_25": "0.48",
            "Consistency_200": "3.34",
            "Consistency_1000": "16.08",
            "Generalization_25": "0.49",
            "Generalization_200": "3.81",
            "Generalization_1000": "16.65",
            "Adaptation_25": "0.47",
            "Adaptation_200": "3.79",
            "Adaptation_1000": "16.63",
            "Recurrence_25": "0.47",
            "Recurrence_200": "3.83",
            "Recurrence_1000": "17.13",
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
        return text.strip("_") or "best_results_table"

    def _ensure_output_dir(self, output_dir: Optional[str] = None) -> str:
        final_dir = output_dir or self.output_dir
        os.makedirs(final_dir, exist_ok=True)
        return final_dir

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

    def _normalize_text(self, value) -> str:
        return str(value).strip()

    def _scenario_base(self, value: str) -> str:
        value = self._normalize_text(value)
        return self.scenario_map.get(value, value)

    def _scenario_name(self, category, block) -> str:
        base = self._scenario_base(category)
        block_text = self._normalize_text(block).replace(",", ".")
        try:
            block_number = float(block_text)
            if block_number.is_integer():
                block_text = str(int(block_number))
            else:
                block_text = f"{block_number:g}"
        except ValueError:
            pass
        return f"{base}_{block_text}"

    def _task_type_label(self, value: str) -> str:
        text = str(value).strip().lower()
        if text in ["classification", "classifier", "supervised"]:
            return "classifier"
        if text in ["anomaly", "detector", "anomaly_detector"]:
            return "anomaly"
        return text

    def _model_alias(self, value) -> str:
        text = str(value).strip()
        return self.model_aliases.get(text, text)

    def _feature_label(self, value) -> str:
        text = str(value).strip().lower()
        if "33features" in text or "33_features" in text or "bestfeatures" in text or "best_features" in text:
            return "Best"
        if "fullfeatures" in text or "full_features" in text:
            return "Full"
        return "-"

    def _format_threshold(self, value, task_type: str) -> str:
        if task_type == "classifier":
            return "-"
        text = str(value).strip()
        if text.lower() in ["classifier", "classification", "n/a", "nan", "none", ""]:
            return "-"
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
            return "-"
        text = str(value).strip()
        if text.lower() in ["classifier", "classification", "n/a", "nan", "none", ""]:
            return "-"
        mapping = {
            "raw": "raw",
            "moving_average_w3": "moving average w3",
            "moving_average_w5": "moving average w5",
            "persistence_2_of_3": "persistence 2 of 3",
            "persistence_3_of_5": "persistence 3 of 5",
        }
        return mapping.get(text.lower(), text.replace("_", " "))

    def build_all_results_dataframe(
        self,
        csv_dict: Dict[str, Optional[Union[str, int]]],
        include_blocks: Optional[Sequence[str]] = None,
        exclude_blocks: Optional[Sequence[str]] = None,
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
            if "Dataset" in df.columns:
                if include_blocks:
                    df = df[df["Dataset"].astype(str).apply(lambda value: any(str(block) in value for block in include_blocks))]
                if exclude_blocks:
                    df = df[~df["Dataset"].astype(str).apply(lambda value: any(str(block) in value for block in exclude_blocks))]
            if df.empty:
                continue
            required_cols = [
                "Dataset",
                "Model",
                "Scenario",
                "Threshold_Strategy",
                "Decision_Strategy",
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
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"CSV {path} does not contain the expected columns: {missing}")
            for _, row in df.iterrows():
                dataset_text = str(row.get("Dataset", "N/A"))
                parts = dataset_text.split("_")
                category = row.get("Category", parts[0] if parts else "N/A")
                block = row.get("Contamination_Block", parts[-1] if parts else "N/A")
                task_type = self._task_type_label(row.get("Task_Type", "anomaly"))
                rows.append({
                    "Scenario": self._scenario_name(category, block),
                    "Task_Type_Internal": task_type,
                    "Best Model": self._model_alias(row.get("Model", "N/A")),
                    "Features": self._feature_label(row.get("Scenario", "N/A")),
                    "F1_avg": row.get("F1_avg", 0.0),
                    "F1_std": row.get("F1_std", 0.0),
                    "Prec_avg": row.get("Prec_avg", 0.0),
                    "Prec_std": row.get("Prec_std", 0.0),
                    "Rec_avg": row.get("Rec_avg", 0.0),
                    "Rec_std": row.get("Rec_std", 0.0),
                    "FN_avg": row.get("FN_avg", 0.0),
                    "FN_std": row.get("FN_std", 0.0),
                    "FP_avg": row.get("FP_avg", 0.0),
                    "FP_std": row.get("FP_std", 0.0),
                    "Threshold": self._format_threshold(row.get("Threshold_Strategy", row.get("Strategy", "N/A")), task_type),
                    "Decision": self._format_decision(row.get("Decision_Strategy", "N/A"), task_type),
                    "Scenario_Config": row.get("Scenario", "N/A"),
                    "Threshold_Internal": row.get("Threshold_Strategy", row.get("Strategy", "N/A")),
                    "Decision_Internal": row.get("Decision_Strategy", "N/A"),
                })
        if not rows:
            return pd.DataFrame()
        df_final = pd.DataFrame(rows)
        numeric_cols = ["F1_avg", "F1_std", "Prec_avg", "Prec_std", "Rec_avg", "Rec_std", "FN_avg", "FN_std", "FP_avg", "FP_std"]
        for col in numeric_cols:
            df_final[col] = self._to_numeric(df_final[col])
        dedup_cols = ["Scenario", "Task_Type_Internal", "Best Model", "Features", "Scenario_Config", "Threshold_Internal", "Decision_Internal"]
        df_final = df_final.drop_duplicates(subset=dedup_cols, keep="last")
        return df_final.reset_index(drop=True)

    def _select_best_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        selected = []
        for _, group_df in df.groupby(["Scenario", "Task_Type_Internal"], dropna=False):
            group_df = group_df.copy()
            group_df["_rank_fn"] = group_df["FN_avg"].fillna(np.inf)
            group_df["_rank_fp"] = group_df["FP_avg"].fillna(np.inf)
            group_df = group_df.sort_values(
                ["F1_avg", "Rec_avg", "Prec_avg", "_rank_fn", "_rank_fp"],
                ascending=[False, False, False, True, True],
            )
            selected.append(group_df.iloc[0])
        best = pd.DataFrame(selected).drop(columns=["_rank_fn", "_rank_fp"], errors="ignore")
        scenario_order = {
            "Consistency_25": 0,
            "Consistency_200": 1,
            "Consistency_1000": 2,
            "Generalization_25": 3,
            "Generalization_200": 4,
            "Generalization_1000": 5,
            "Adaptation_25": 6,
            "Adaptation_200": 7,
            "Adaptation_1000": 8,
            "Recurrence_25": 9,
            "Recurrence_200": 10,
            "Recurrence_1000": 11,
        }
        type_order = {"classifier": 0, "anomaly": 1}
        best["_scenario_order"] = best["Scenario"].map(scenario_order).fillna(99)
        best["_type_order"] = best["Task_Type_Internal"].map(type_order).fillna(99)
        return best.sort_values(["_scenario_order", "_type_order", "Best Model"]).drop(columns=["_scenario_order", "_type_order"], errors="ignore").reset_index(drop=True)

    def _metric_pm(self, mean_value, std_value, integer: bool = False) -> str:
        if pd.isnull(mean_value):
            return "-"
        if integer:
            mean_text = str(int(np.ceil(float(mean_value))))
            std_text = str(int(np.ceil(float(std_value)))) if pd.notnull(std_value) else "0"
            return f"{mean_text} ± {std_text}"
        std = float(std_value) if pd.notnull(std_value) else 0.0
        return f"{float(mean_value):.2f} ± {std:.2f}"

    def build_best_dataframe(
        self,
        csv_dict: Dict[str, Optional[Union[str, int]]],
        include_blocks: Optional[Sequence[str]] = None,
        exclude_blocks: Optional[Sequence[str]] = None,
        use_exec_filter: bool = True,
        use_latest_when_exec_id_none: bool = True,
    ) -> pd.DataFrame:
        df = self.build_all_results_dataframe(
            csv_dict=csv_dict,
            include_blocks=include_blocks,
            exclude_blocks=exclude_blocks,
            use_exec_filter=use_exec_filter,
            use_latest_when_exec_id_none=use_latest_when_exec_id_none,
        )
        best = self._select_best_rows(df)
        if best.empty:
            return best
        output = pd.DataFrame({
            "Scenario": best["Scenario"],
            "Poisoning (%)": best["Scenario"].map(self.poisoning_percentages).fillna("-"),
            "Best Model": best["Best Model"],
            "Features": best["Features"],
            "F1-Score": [self._metric_pm(m, s) for m, s in zip(best["F1_avg"], best["F1_std"])],
            "Precision": [self._metric_pm(m, s) for m, s in zip(best["Prec_avg"], best["Prec_std"])],
            "Recall": [self._metric_pm(m, s) for m, s in zip(best["Rec_avg"], best["Rec_std"])],
            "FN": [self._metric_pm(m, s, integer=True) for m, s in zip(best["FN_avg"], best["FN_std"])],
            "FP": [self._metric_pm(m, s, integer=True) for m, s in zip(best["FP_avg"], best["FP_std"])],
            "Threshold": best["Threshold"],
            "Decision": best["Decision"],
        })
        return output.reset_index(drop=True)

    def save_png(
        self,
        df: pd.DataFrame,
        title: str = "Best Results by Scenario",
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        col_widths: Optional[Sequence[float]] = None,
    ) -> str:
        if df.empty:
            raise ValueError("Empty DataFrame. There is no table to save as PNG.")
        final_dir = self._ensure_output_dir(output_dir)
        png_path = os.path.join(final_dir, filename or f"{self._clean_filename(title)}.png")
        widths = list(col_widths) if col_widths is not None else [0.13, 0.07, 0.07, 0.07, 0.105, 0.105, 0.105, 0.06, 0.06, 0.09, 0.13]
        fig_height = max(3.0, 0.42 * len(df) + 1.0)
        fig, ax = plt.subplots(figsize=(19, fig_height))
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center", colWidths=widths[:len(df.columns)])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#777777")
            cell.set_linewidth(0.45)
            if row_idx == 0:
                cell.set_facecolor("#3f456c")
                cell.set_text_props(weight="bold", color="white")
            elif row_idx % 2 == 0:
                cell.set_facecolor("#f7f7f7")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        plt.savefig(png_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return png_path

    def save_pdf(
        self,
        df: pd.DataFrame,
        title: str = "Best Results by Scenario",
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        col_widths: Optional[Sequence[float]] = None,
    ) -> str:
        if df.empty:
            raise ValueError("Empty DataFrame. There is no table to save as PDF.")
        final_dir = self._ensure_output_dir(output_dir)
        pdf_path = os.path.join(final_dir, filename or f"{self._clean_filename(title)}.pdf")
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=14, leading=16)
        cell_style = ParagraphStyle("Cell", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=6.8, leading=8)
        header_style = ParagraphStyle("Header", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=7, leading=8, textColor=colors.white)
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4), rightMargin=0.6 * cm, leftMargin=0.6 * cm, topMargin=0.7 * cm, bottomMargin=0.7 * cm)
        widths = list(col_widths) if col_widths is not None else [2.65, 1.45, 1.55, 1.35, 2.15, 2.15, 2.15, 1.35, 1.35, 2.0, 2.75]
        widths = [value * cm for value in widths]
        story = [Paragraph(title, title_style), Spacer(1, 0.25 * cm)]
        table_data = [[Paragraph(str(col), header_style) for col in df.columns]]
        for _, row in df.iterrows():
            table_data.append([Paragraph(str(row[col]), cell_style) for col in df.columns])
        table = Table(table_data, colWidths=widths[:len(df.columns)], repeatRows=1)
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self.header_color)),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(self.grid_color)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]
        for local_idx in range(1, len(df) + 1):
            if local_idx % 2 == 0:
                table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.even_row_color)))
        table.setStyle(TableStyle(table_style))
        story.append(table)
        doc.build(story)
        return pdf_path

    def _latex_escape(self, value) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            "±": r"$\pm$",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _latex_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        latex_df = df.copy()
        if "Poisoning (%)" not in latex_df.columns and "Scenario" in latex_df.columns:
            latex_df.insert(1, "Poisoning (%)", latex_df["Scenario"].map(self.poisoning_percentages).fillna("-"))
        return latex_df

    def save_latex(
        self,
        df: pd.DataFrame,
        caption: str = "Best results by scenario.",
        label: str = "tab:best_results_by_scenario",
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        if df.empty:
            raise ValueError("Empty DataFrame. There is no table to save as LaTeX.")
        final_dir = self._ensure_output_dir(output_dir)
        latex_path = os.path.join(final_dir, filename or "best_results_table.tex")
        latex_df = self._latex_dataframe(df)
        headers = list(latex_df.columns)
        col_spec = "llccccccccc"
        lines = [
            r"\begin{table*}[ht]",
            r"\centering",
            rf"\caption{{{self._latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.8pt}",
            r"\renewcommand{\arraystretch}{1.05}",
            r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\hline",
            " & ".join([rf"\textbf{{{self._latex_escape(h)}}}" for h in headers]) + r" \\ \hline",
        ]
        previous_scenario = None
        for _, row in latex_df.iterrows():
            scenario = str(row["Scenario"])
            values = []
            for col in headers:
                value = row[col]
                if col == "Scenario" and scenario == previous_scenario:
                    values.append("")
                else:
                    values.append(self._latex_escape(value))
            if previous_scenario is not None and scenario != previous_scenario:
                lines.append(r"\hline")
            lines.append(" & ".join(values) + r" \\")
            previous_scenario = scenario
        lines.extend([
            r"\hline",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ])
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return latex_path

    def create_outputs(
        self,
        csv_dict: Dict[str, Optional[Union[str, int]]],
        title: str = "Best Results by Scenario",
        include_blocks: Optional[Sequence[str]] = None,
        exclude_blocks: Optional[Sequence[str]] = None,
        use_exec_filter: bool = True,
        use_latest_when_exec_id_none: bool = True,
        output_dir: Optional[str] = None,
        png_filename: Optional[str] = None,
        pdf_filename: Optional[str] = None,
        latex_filename: Optional[str] = None,
        save_png: bool = True,
        save_pdf: bool = True,
        save_latex: bool = True,
        png_col_widths: Optional[Sequence[float]] = None,
        pdf_col_widths: Optional[Sequence[float]] = None,
        latex_caption: str = "Best results by scenario.",
        latex_label: str = "tab:best_results_by_scenario",
    ) -> pd.DataFrame:
        df = self.build_best_dataframe(
            csv_dict=csv_dict,
            include_blocks=include_blocks,
            exclude_blocks=exclude_blocks,
            use_exec_filter=use_exec_filter,
            use_latest_when_exec_id_none=use_latest_when_exec_id_none,
        )
        if df.empty:
            print("No data found to generate the best results table.")
            return df
        if save_png:
            png_path = self.save_png(df, title=title, output_dir=output_dir, filename=png_filename, col_widths=png_col_widths)
            print(f"PNG saved at: {png_path}")
        if save_pdf:
            pdf_path = self.save_pdf(df, title=title, output_dir=output_dir, filename=pdf_filename, col_widths=pdf_col_widths)
            print(f"PDF saved at: {pdf_path}")
        if save_latex:
            latex_path = self.save_latex(df, caption=latex_caption, label=latex_label, output_dir=output_dir, filename=latex_filename)
            print(f"LaTeX saved at: {latex_path}")
        return df
