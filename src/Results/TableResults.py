import os
import re
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


class TableResults:
    def __init__(self, output_dir: str = "output/Results"):
        self.output_dir = output_dir
        self.header_color = "#3f456c"
        self.global_best_color = "#d9ead3"
        self.local_best_color = "#fff2cc"
        self.global_worst_color = "#f4cccc"
        self.zero_result_color = "#e6e6e6"
        self.even_row_color = "#f7f7f7"
        self.grid_color = "#777777"

    def _read_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

    def _to_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    def _clean_filename(self, text: str) -> str:
        text = str(text).strip()
        text = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE)
        return text.strip("_") or "table_results"

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
                "Dataset", "Category", "Contamination_Block", "Task_Type", "Model", "Scenario",
                "Strategy", "Threshold_Strategy", "Decision_Strategy", "Decision_Window",
                "Persistence_K", "Persistence_N", "Warmup"
            ]
            if col in df.columns
        ]
        if group_cols:
            df = df.sort_values(["_exec_datetime", "_exec_order"]).groupby(group_cols, dropna=False, as_index=False).tail(1)
        else:
            df = df.sort_values(["_exec_datetime", "_exec_order"]).tail(1)
        return df.drop(columns=["_exec_datetime", "_exec_order"], errors="ignore").drop_duplicates().reset_index(drop=True)

    def build_dataframe(self, csv_dict: Dict[str, Optional[Union[str, int]]], include_blocks: Optional[Sequence[str]] = None, exclude_blocks: Optional[Sequence[str]] = None, use_exec_filter: bool = True, use_latest_when_exec_id_none: bool = True) -> pd.DataFrame:
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
            required_cols = ["Dataset", "Model", "Scenario", "Threshold_Strategy", "Decision_Strategy", "F1_avg", "Prec_avg", "Rec_avg", "FP_avg", "FN_avg"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"CSV {path} does not contain the expected columns: {missing}")
            for _, row in df.iterrows():
                rows.append({
                    "Dataset": row.get("Dataset", "N/A"),
                    "Model": row.get("Model", "N/A"),
                    "Scenario": row.get("Scenario", "N/A"),
                    "Threshold": row.get("Threshold_Strategy", row.get("Strategy", "N/A")),
                    "Decision": row.get("Decision_Strategy", "N/A"),
                    "F1-Score": row.get("F1_avg", 0.0),
                    "Precision": row.get("Prec_avg", 0.0),
                    "Recall": row.get("Rec_avg", 0.0),
                    "FP": row.get("FP_avg", 0.0),
                    "FN": row.get("FN_avg", 0.0),
                })
        if not rows:
            return pd.DataFrame()
        df_final = pd.DataFrame(rows)
        for col in ["F1-Score", "Precision", "Recall", "FP", "FN"]:
            df_final[col] = self._to_numeric(df_final[col])
        df_final = df_final.drop_duplicates(subset=["Dataset", "Model", "Scenario", "Threshold", "Decision"], keep="last")
        return df_final.sort_values(["Dataset", "Model", "Scenario", "Threshold", "Decision"]).reset_index(drop=True)

    def _highlight_indices(self, df: pd.DataFrame) -> dict:
        highlights = {"global_best": set(), "local_best": set(), "global_worst": set(), "zero_result": set()}
        if df.empty or "F1-Score" not in df.columns:
            return highlights
        metric_cols = [col for col in ["F1-Score", "Precision", "Recall"] if col in df.columns]
        if metric_cols:
            zero_mask = df[metric_cols].fillna(0).eq(0).all(axis=1)
            highlights["zero_result"].update(df[zero_mask].index.tolist())
        non_zero_df = df.drop(index=list(highlights["zero_result"]), errors="ignore")
        if not non_zero_df.empty:
            best_value = non_zero_df["F1-Score"].max()
            worst_value = non_zero_df["F1-Score"].min()
            highlights["global_best"].update(non_zero_df[non_zero_df["F1-Score"] == best_value].index.tolist())
            highlights["global_worst"].update(non_zero_df[non_zero_df["F1-Score"] == worst_value].index.tolist())
            if "Dataset" in non_zero_df.columns:
                for _, group_df in non_zero_df.groupby("Dataset", dropna=False):
                    local_value = group_df["F1-Score"].max()
                    highlights["local_best"].update(group_df[group_df["F1-Score"] == local_value].index.tolist())
        highlights["local_best"] = highlights["local_best"] - highlights["global_best"] - highlights["global_worst"] - highlights["zero_result"]
        return highlights

    def _format_value(self, value, col: str) -> str:
        if pd.isnull(value):
            return "-"
        if col in ["F1-Score", "Precision", "Recall"]:
            return f"{float(value):.2f}"
        if col in ["FP", "FN"]:
            return str(int(np.ceil(float(value))))
        return str(value)

    def _display_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        formatted = df.copy()
        for col in ["F1-Score", "Precision", "Recall", "FP", "FN"]:
            formatted[col] = formatted[col].apply(lambda value: self._format_value(value, col))
        return formatted

    def _build_cover_page(self, title: str, styles, colors, cm, Paragraph, Spacer, Table, TableStyle):
        title_style = ParagraphStyle("CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=17, leading=21, spaceAfter=10)
        section_style = ParagraphStyle("CoverSection", parent=styles["Heading2"], alignment=TA_LEFT, fontSize=12, leading=15, spaceBefore=10, spaceAfter=6)
        text_style = ParagraphStyle("CoverText", parent=styles["BodyText"], alignment=TA_LEFT, fontSize=9.5, leading=13)
        legend_style = ParagraphStyle("CoverLegend", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=9, leading=11)
        story = [
            Paragraph(title, title_style),
            Spacer(1, 0.35 * cm),
            Paragraph("Decision strategies shown in the table", section_style),
            Paragraph("<b>Fixed / Raw:</b> baseline decision rule. Each anomaly score is compared directly with the established threshold. In the table, this direct decision can appear as <b>raw</b>.", text_style),
            Spacer(1, 0.15 * cm),
            Paragraph("<b>Moving average:</b> temporal decision rule based on causal smoothing of recent anomaly scores. In the table, these configurations can appear as <b>moving_average_w3</b> and <b>moving_average_w5</b>.", text_style),
            Spacer(1, 0.15 * cm),
            Paragraph("<b>Temporal persistence:</b> temporal decision rule in which the anomaly condition must occur repeatedly within a recent window. In the table, these configurations can appear as <b>persistence_2_of_3</b> and <b>persistence_3_of_5</b>.", text_style),
            Spacer(1, 0.35 * cm),
            Paragraph("Highlight legend", section_style),
        ]
        legend_data = [
            [Paragraph("Global best result", legend_style), Paragraph("Local best result by dataset", legend_style), Paragraph("Global worst non-zero result", legend_style), Paragraph("Zero-score result", legend_style)],
            [Paragraph("Highest F1-Score among all evaluated scenarios.", text_style), Paragraph("Highest F1-Score within each dataset.", text_style), Paragraph("Lowest non-zero F1-Score among all evaluated scenarios.", text_style), Paragraph("Rows with F1-Score, Precision, and Recall equal to zero.", text_style)],
        ]
        legend_table = Table(legend_data, colWidths=[5.0 * cm, 5.4 * cm, 5.4 * cm, 4.5 * cm])
        legend_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(self.global_best_color)),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(self.local_best_color)),
            ("BACKGROUND", (2, 0), (2, 0), colors.HexColor(self.global_worst_color)),
            ("BACKGROUND", (3, 0), (3, 0), colors.HexColor(self.zero_result_color)),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(self.grid_color)),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(legend_table)
        story.append(Spacer(1, 0.35 * cm))
        story.append(Paragraph("Reading note", section_style))
        story.append(Paragraph("The Exec_ID is used only internally to select the most recent execution when no specific identifier is provided. It is not displayed in the final table.", text_style))
        return story

    def save_pdf(self, df: pd.DataFrame, title: str = "Complete Results Table", output_dir: Optional[str] = None, filename: Optional[str] = None, rows_per_page: int = 32, col_widths: Optional[Sequence[float]] = None, include_cover_page: bool = True) -> str:
        if df.empty:
            raise ValueError("Empty DataFrame. There is no table to save as PDF.")
        final_dir = self._ensure_output_dir(output_dir)
        pdf_path = os.path.join(final_dir, filename or f"{self._clean_filename(title)}.pdf")
        highlights = self._highlight_indices(df)
        df_pdf = self._display_dataframe(df)
        styles = getSampleStyleSheet()
        cell_style = ParagraphStyle("Cell", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=6.8, leading=8)
        header_style = ParagraphStyle("Header", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=7, leading=8, textColor=colors.white)
        doc = SimpleDocTemplate(pdf_path, pagesize=landscape(A4), rightMargin=0.7 * cm, leftMargin=0.7 * cm, topMargin=0.7 * cm, bottomMargin=0.7 * cm)
        story = []
        if include_cover_page:
            story.extend(self._build_cover_page(title, styles, colors, cm, Paragraph, Spacer, Table, TableStyle))
            story.append(PageBreak())
        columns = list(df_pdf.columns)
        widths = list(col_widths) if col_widths is not None else [4.0, 3.8, 3.2, 2.4, 2.9, 1.55, 1.55, 1.55, 1.2, 1.2]
        widths = [value * cm for value in widths]
        for start in range(0, len(df_pdf), rows_per_page):
            page_df = df_pdf.iloc[start:start + rows_per_page]
            table_data = [[Paragraph(str(col), header_style) for col in columns]]
            for _, row in page_df.iterrows():
                table_data.append([Paragraph(str(row[col]), cell_style) for col in columns])
            table = Table(table_data, colWidths=widths[:len(columns)], repeatRows=1)
            table_style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(self.header_color)), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(self.grid_color)), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]
            for local_idx, original_idx in enumerate(page_df.index, start=1):
                if original_idx in highlights["zero_result"]:
                    table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.zero_result_color)))
                elif original_idx in highlights["global_best"]:
                    table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.global_best_color)))
                elif original_idx in highlights["global_worst"]:
                    table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.global_worst_color)))
                elif original_idx in highlights["local_best"]:
                    table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.local_best_color)))
                elif local_idx % 2 == 0:
                    table_style.append(("BACKGROUND", (0, local_idx), (-1, local_idx), colors.HexColor(self.even_row_color)))
            table.setStyle(TableStyle(table_style))
            story.append(table)
            if start + rows_per_page < len(df_pdf):
                story.append(PageBreak())
        doc.build(story)
        return pdf_path

    def create_pdf(self, csv_dict: Dict[str, Optional[Union[str, int]]], title: str = "Complete Results Table", include_blocks: Optional[Sequence[str]] = None, exclude_blocks: Optional[Sequence[str]] = None, use_exec_filter: bool = True, use_latest_when_exec_id_none: bool = True, output_dir: Optional[str] = None, filename: Optional[str] = None, rows_per_page: int = 32, col_widths: Optional[Sequence[float]] = None, include_cover_page: bool = True) -> pd.DataFrame:
        df = self.build_dataframe(csv_dict=csv_dict, include_blocks=include_blocks, exclude_blocks=exclude_blocks, use_exec_filter=use_exec_filter, use_latest_when_exec_id_none=use_latest_when_exec_id_none)
        if df.empty:
            print("No data found to generate the table.")
            return df
        pdf_path = self.save_pdf(df=df, title=title, output_dir=output_dir, filename=filename, rows_per_page=rows_per_page, col_widths=col_widths, include_cover_page=include_cover_page)
        print(f"PDF saved at: {pdf_path}")
        return df
