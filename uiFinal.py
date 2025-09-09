# uif.py
"""
Data Quality Assessment Tool UI (uif.py)

Run:
    streamlit run uif.py

Notes:
- This file expects optional modules cc_errorReport.py, cc.py, or ccUpd.py exporting
  `run_completeness_check(df)` to be available. If none are present, a demo stub runs.
- Place eyLogo.png and eyLogo2.png in the same folder to show logos in the header.
"""

import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Try to import completeness implementations preferring cc_errorReport then cc then ccUpd
completeness_available = False
for modname in ("cc_errorReport", "cc", "ccUpd"):
    try:
        mod = __import__(modname)
        run_completeness_check = getattr(mod, "run_completeness_check")
        completeness_available = True
        completeness_source = modname
        break
    except Exception:
        run_completeness_check = None
        completeness_source = None

# FALLBACK STUB if completeness not available
if not completeness_available:
    def run_completeness_check(df: pd.DataFrame):
        na_mask = df.isna() | df.applymap(lambda x: isinstance(x, str) and x.strip() == "")
        row_missing_counts = na_mask.sum(axis=1)
        incomplete_idx = row_missing_counts[row_missing_counts > 0].index.tolist()

        incomplete_rows = []
        for i in incomplete_idx:
            missing_cols = list(df.columns[na_mask.loc[i]])
            incomplete_rows.append({
                "Row No.": int(i) + 1,
                "Data Feed Unique ID": f"row_{i+1}",
                "num_missing": int(row_missing_counts.loc[i]),
                "missing_attributes": ", ".join(missing_cols)
            })
        incomplete_df = pd.DataFrame(incomplete_rows)

        attr_overview = pd.DataFrame({
            "attribute": df.columns,
            "missing_count": na_mask.sum(axis=0).astype(int).values
        }).sort_values("missing_count", ascending=False).reset_index(drop=True)

        total_cells = int(df.size) if df.size else 0
        total_missing = int(na_mask.sum().sum()) if df.size else 0
        stats = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_cells": total_cells,
            "total_missing": total_missing,
            "percent_missing": round(100 * (total_missing / total_cells), 2) if total_cells else 0.0
        }

        # remediation bytes
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                incomplete_df.to_excel(writer, sheet_name="IncompleteRows", index=False)
            remediation_bytes = buf.getvalue()
        except Exception:
            remediation_bytes = b""
        return stats, incomplete_df, attr_overview, remediation_bytes

    completeness_source = "stub"

# Basic streamlit config & CSS
st.set_page_config(page_title="Data Quality Assessment Tool", layout="wide")
CSS = """
<style>
html, body, .reportview-container .main {
  background-color: #000000;
  color: #FFFFFF;
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.banner { background:#FFD400; border-radius:8px; padding:10px 14px; display:flex; align-items:center; justify-content:space-between; }
.sidebar-nav .stButton>button { background: #FFD400; color:#000; font-weight:700; width:100%; padding:12px 14px; border-radius:8px; text-align:left; margin-bottom:8px; }
.sidebar-nav .stButton.home>button { background:#333333 !important; color:#fff !important; }
.stButton>button, .stDownloadButton>button { border-radius:8px; }
.upload-strip { background:#fff; color:#000; padding:10px; border-radius:8px; }
.small-grey { color:#bfbfbf; font-size:13px; }
.card { background:#0a0a0a; border:1px solid #222; padding:12px; border-radius:8px; }
.kpi { font-weight:700; font-size:18px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- Utilities ----------
def parse_workbook(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheets = {sn: pd.read_excel(uploaded_file, sheet_name=sn) for sn in xls.sheet_names}
        return sheets, list(xls.sheet_names)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return {}, []

def group_workstreams(sheet_names: List[str]) -> Dict[str, List[str]]:
    groups = {}
    for sn in sheet_names:
        m = re.match(r"^(\d+)[\-_ ]*(.*)$", sn)
        if m:
            key = m.group(1)
        else:
            key = sn.split()[0] if sn.strip() else "misc"
        groups.setdefault(key, []).append(sn)
    return groups

def generate_error_report_from_incomplete(df: pd.DataFrame, incomplete_df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    rows = []
    rowno_col = None
    for c in incomplete_df.columns:
        if "row" in c.lower():
            rowno_col = c
            break
    if rowno_col:
        miss_col = None
        for c in incomplete_df.columns:
            if "missing" in c.lower():
                miss_col = c
                break
        for _, r in incomplete_df.iterrows():
            try:
                rowno = int(r[rowno_col]) - 1
            except Exception:
                continue
            if rowno < 0 or rowno >= len(df):
                continue
            src = df.iloc[rowno]
            if miss_col and not pd.isna(r[miss_col]) and str(r[miss_col]).strip() != "":
                parts = [p.strip() for p in str(r[miss_col]).replace(";", ",").split(",") if p.strip()]
                for attr in parts:
                    if selected_columns and attr not in selected_columns:
                        continue
                    rowd = src.to_dict()
                    rowd["Error Type"] = "Missing Value"
                    rowd["Error Value"] = attr
                    rows.append(rowd)
            else:
                for col in selected_columns:
                    val = src.get(col, None)
                    if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                        rowd = src.to_dict()
                        rowd["Error Type"] = "Missing Value"
                        rowd["Error Value"] = col
                        rows.append(rowd)
    else:
        for idx, src in df.iterrows():
            for col in selected_columns:
                val = src.get(col, None)
                if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                    rowd = src.to_dict()
                    rowd["Error Type"] = "Missing Value"
                    rowd["Error Value"] = col
                    rows.append(rowd)

    if not rows:
        out_cols = list(df.columns) + ["Error Type", "Error Value"]
        return pd.DataFrame(columns=out_cols)
    out_df = pd.DataFrame(rows)
    final_cols = list(df.columns) + ["Error Type", "Error Value"]
    final_cols = [c for c in final_cols if c in out_df.columns]
    return out_df[final_cols].reset_index(drop=True)

def df_to_excel_bytes(d: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        d.to_excel(writer, index=False, sheet_name="error_report")
    return buf.getvalue()

# ---------- Normalizer & robust UI helpers (fix KeyError: 'attribute') ----------
def normalize_attr_overview(attr_df: pd.DataFrame, backing_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Normalize incoming attr_overview into DataFrame with columns ['attribute','missing_count'].
    If attr_df is empty and backing_df provided, compute missing counts from backing_df.
    """
    if attr_df is None or (isinstance(attr_df, pd.DataFrame) and attr_df.empty):
        if backing_df is not None:
            na_mask = backing_df.isna() | backing_df.applymap(lambda x: isinstance(x, str) and x.strip() == "")
            counts = na_mask.sum(axis=0).astype(int)
            out = pd.DataFrame({"attribute": list(counts.index.astype(str)), "missing_count": counts.values})
            return out.sort_values("missing_count", ascending=False).reset_index(drop=True)
        return pd.DataFrame(columns=["attribute", "missing_count"])

    df = attr_df.copy()

    # candidate names
    attr_candidates = [c for c in df.columns if c.lower() in {"attribute", "attr", "column", "field", "name", "column_name"}]
    count_candidates = [c for c in df.columns if c.lower() in {"missing_count", "missing", "count", "nulls", "missingcount"}]

    if attr_candidates and count_candidates:
        a = attr_candidates[0]
        b = count_candidates[0]
        out = pd.DataFrame({"attribute": df[a].astype(str), "missing_count": pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)})
        return out.sort_values("missing_count", ascending=False).reset_index(drop=True)

    if attr_candidates:
        a = attr_candidates[0]
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            b = numeric_cols[0]
            out = pd.DataFrame({"attribute": df[a].astype(str), "missing_count": pd.to_numeric(df[b], errors="coerce").fillna(0).astype(int)})
            return out.sort_values("missing_count", ascending=False).reset_index(drop=True)

    if df.shape[1] == 2:
        col0, col1 = df.columns[0], df.columns[1]
        try:
            counts = pd.to_numeric(df[col1], errors="coerce").fillna(0).astype(int)
            out = pd.DataFrame({"attribute": df[col0].astype(str), "missing_count": counts})
            return out.sort_values("missing_count", ascending=False).reset_index(drop=True)
        except Exception:
            pass

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        count_col = numeric_cols[0]
        string_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        attr_col = string_cols[0] if string_cols else df.columns[0]
        out = pd.DataFrame({"attribute": df[attr_col].astype(str), "missing_count": pd.to_numeric(df[count_col], errors="coerce").fillna(0).astype(int)})
        return out.sort_values("missing_count", ascending=False).reset_index(drop=True)

    return pd.DataFrame(columns=["attribute", "missing_count"])

def show_attr_overview(attr_overview: pd.DataFrame, backing_df: pd.DataFrame = None):
    try:
        norm = normalize_attr_overview(attr_overview, backing_df=backing_df)
    except Exception as e:
        st.error(f"Unable to render attribute overview: {e}")
        return
    if norm.empty:
        st.info("No attribute overview available (no missing counts detected or overview not provided).")
        return
    st.write("Missing Count per Attribute")
    st.dataframe(norm.head(200), use_container_width=True)

def show_bar_chart(attr_overview: pd.DataFrame, backing_df: pd.DataFrame = None):
    try:
        norm = normalize_attr_overview(attr_overview, backing_df=backing_df)
    except Exception as e:
        st.error(f"Unable to prepare chart: {e}")
        return
    if norm.empty:
        st.info("No attribute overview to chart.")
        return
    top = norm.head(10)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(top["attribute"].astype(str), top["missing_count"].astype(int))
    ax.invert_yaxis()
    ax.set_xlabel("Missing Count")
    ax.set_title("Top attributes by missing count")
    st.pyplot(fig)

# ---------- small UI building blocks ----------
def kpi_cards(stats: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='card'><div class='kpi'>Rows</div><div>{stats.get('rows', 0)}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='kpi'>Columns</div><div>{stats.get('columns', 0)}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='kpi'>Missing Cells</div><div>{stats.get('total_missing', 0)}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'><div class='kpi'>% Missing</div><div>{stats.get('percent_missing', 0.0)}%</div></div>", unsafe_allow_html=True)

# ---------- header & nav ----------
def ui_header():
    colL, colC, colR = st.columns([1, 6, 1])
    left_img = None
    right_img = None
    try:
        with open("eyLogo.png", "rb") as f: left_img = f.read()
    except Exception:
        left_img = None
    try:
        with open("eyLogo2.png", "rb") as f: right_img = f.read()
    except Exception:
        right_img = None

    with colL:
        if left_img:
            st.image(left_img, width=80)
        else:
            st.markdown('<div style="background:#000;color:#FFD400;padding:8px;border-radius:6px;font-weight:900;text-align:center">EY</div>', unsafe_allow_html=True)
    with colC:
        st.markdown('<div class="banner"><div style="font-weight:800;font-size:20px;color:#000">Data Quality Assessment Tool</div>'
                    '<div style="font-size:13px;color:#222">Polished UI -- completeness source: <b>{}</b></div></div>'.format(completeness_source),
                    unsafe_allow_html=True)
    with colR:
        if right_img:
            st.image(right_img, width=120)
        else:
            st.markdown('<div style="color:#000;background:#FFD400;padding:6px 8px;border-radius:6px;font-weight:700;text-align:right">Integrity • Insight</div>', unsafe_allow_html=True)

def left_nav_bar():
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "Home"
    nav_items = ["Home", "Completeness", "Consistency", "Lineage", "Validation", "All Checks"]
    st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    for item in nav_items:
        # each button updates session state
        if st.button(item, key=f"nav_{item}"):
            st.session_state.active_nav = item
    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.active_nav

# ---------- main page ----------
ui_header()
left_col, right_col = st.columns([1, 4])

with left_col:
    active = left_nav_bar()

with right_col:
    if active == "Home":
        tabs = st.tabs(["Upload", "Assessment Summary", "Custom Check", "Visualization", "Downloads"])
        # UPLOAD
        with tabs[0]:
            st.markdown('<div class="upload-strip">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Excel workbook", type=["xlsx"], key="main_upload")
            st.markdown("</div>", unsafe_allow_html=True)
            if uploaded_file is not None:
                st.success(f"File uploaded: {getattr(uploaded_file, 'name', '')}")
                sheets_dict, sheet_names = parse_workbook(uploaded_file)
                st.session_state["workbook_dict"] = sheets_dict
                st.session_state["sheet_names"] = sheet_names
            else:
                st.info("Upload an .xlsx workbook to begin (max ~200MB).")

            st.markdown("---")
            st.write("Select sheets / workstreams to include in checks:")
            c1, c2, c3 = st.columns([3, 3, 3])
            with c1:
                st.subheader("Instructions")
                st.markdown("<div class='small-grey'>Upload workbook. Select QC types in center column. Choose sheets on right.</div>", unsafe_allow_html=True)
                st.markdown("- Upload, then Run Check Now to compute completeness.\n- Preview & download outputs below.")
            with c2:
                st.subheader("Quality Checks")
                checks = st.multiselect("Choose checks", ["Completeness Check", "Consistency Check", "Lineage Check", "Validation Check"], default=["Completeness Check"], key="checks_select")
            with c3:
                st.subheader("Sheets / Workstreams")
                sheet_list = st.session_state.get("sheet_names", [])
                sheets = st.multiselect("Select sheets", options=sheet_list, key="sheet_multiselect")
                if sheet_list:
                    groups = group_workstreams(sheet_list)
                    st.write("Detected groups:", ", ".join(list(groups.keys())[:8]))

            if st.button("Run Check Now"):
                if not st.session_state.get("workbook_dict"):
                    st.error("Please upload workbook first.")
                elif not sheets:
                    st.error("Select at least one sheet to run checks on.")
                elif "Completeness Check" not in checks:
                    st.warning("No completeness selected (demo focuses on completeness).")
                else:
                    st.session_state["results"] = {}
                    progress = st.progress(0)
                    total = len(sheets)
                    for i, sheet in enumerate(sheets, start=1):
                        df = st.session_state["workbook_dict"][sheet]
                        try:
                            stats, incomplete_df, attr_overview, remediation_bytes = run_completeness_check(df)
                        except Exception as e:
                            st.error(f"Error running completeness on sheet {sheet}: {e}")
                            stats, incomplete_df, attr_overview, remediation_bytes = {}, pd.DataFrame(), pd.DataFrame(), b""
                        st.session_state["results"][sheet] = {
                            "stats": stats,
                            "incomplete_df": incomplete_df,
                            "attr_overview": attr_overview,
                            "remediation_bytes": remediation_bytes,
                            "error_report": None,
                            "error_report_params": None
                        }
                        progress.progress(int(100 * i / total))
                    st.success("Checks complete. Scroll down for results.")

        # ASSESSMENT SUMMARY
        with tabs[1]:
            st.header("Assessment Summary")
            if not st.session_state.get("results"):
                st.info("Run checks from the Upload tab to populate results.")
            else:
                for sheet, data in st.session_state["results"].items():
                    st.subheader(f"Sheet: {sheet}")
                    kpi_cards(data.get("stats", {}))
                    # pass backing sheet df for robust normalization
                    backing_df = st.session_state.get("workbook_dict", {}).get(sheet, None)
                    show_attr_overview(data.get("attr_overview"), backing_df=backing_df)
                    show_bar_chart(data.get("attr_overview"), backing_df=backing_df)

        # CUSTOM CHECK
        with tabs[2]:
            st.header("Custom Check")
            st.info("Placeholder -- paste custom check UI components here.")

        # VISUALIZATION
        with tabs[3]:
            st.header("Visualization")
            if st.session_state.get("results"):
                first = next(iter(st.session_state["results"].values()))
                backing_df = None
                if first:
                    # try to find a sheet to use as backing df
                    keys = list(st.session_state["results"].keys())
                    if keys:
                        backing_df = st.session_state.get("workbook_dict", {}).get(keys[0], None)
                    show_bar_chart(first.get("attr_overview"), backing_df=backing_df)
                else:
                    st.info("No data to visualize yet.")

        # DOWNLOADS
        with tabs[4]:
            st.header("Downloads")
            if not st.session_state.get("results"):
                st.info("No outputs yet. Run checks to enable downloads.")
            else:
                for sheet, data in st.session_state["results"].items():
                    st.write(f"**{sheet}**")
                    if data["incomplete_df"] is not None and not data["incomplete_df"].empty:
                        st.download_button(
                            label="Download Rows with Incomplete Data (CSV)",
                            data=data["incomplete_df"].to_csv(index=False).encode("utf-8"),
                            file_name=f"{sheet}_IncompleteRows.csv",
                            mime="text/csv"
                        )
                    if data["remediation_bytes"]:
                        st.download_button(
                            label="Download Remediation File (Excel)",
                            data=data["remediation_bytes"],
                            file_name=f"{sheet}_Remediation.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

    else:
        # QC pages with nested tabs
        nested = st.tabs(["Assessment Summary", "Error Report/Preview", "Visualization", "Online Editor", "Rerun Check"])

        # Assessment Summary
        with nested[0]:
            st.header(f"{active} -- Assessment Summary")
            sheet = st.selectbox("Select sheet", options=st.session_state.get("sheet_names", []), key=f"sheet_select_{active}")
            if not sheet:
                st.info("Select a sheet to view QC outputs.")
            elif not st.session_state.get("results") or sheet not in st.session_state["results"]:
                st.info("No results for this sheet yet. Run checks from Home or use Rerun Check.")
            else:
                data = st.session_state["results"][sheet]
                kpi_cards(data.get("stats", {}))
                backing_df = st.session_state.get("workbook_dict", {}).get(sheet, None)
                show_attr_overview(data.get("attr_overview"), backing_df=backing_df)

        # Error Report / Preview
        with nested[1]:
            st.header(f"{active} -- Error Report / Preview")
            sheet = st.selectbox("Select sheet for error report", options=st.session_state.get("sheet_names", []), key=f"er_sheet_{active}")
            if not sheet:
                st.info("Select a sheet to generate error report.")
            else:
                if not st.session_state.get("results") or sheet not in st.session_state["results"]:
                    st.info("No results for this sheet yet. Run checks from Home or use Rerun Check.")
                else:
                    data = st.session_state["results"][sheet]
                    df = st.session_state["workbook_dict"][sheet]
                    default_cols = [c for c in df.columns]
                    chosen = st.multiselect("Columns to check", options=list(df.columns), default=default_cols, key=f"er_cols_{active}")
                    er = generate_error_report_from_incomplete(df, data["incomplete_df"], chosen)
                    total_rows = len(er)
                    st.write(f"Error report rows (one per missing attribute): **{total_rows}**")
                    if total_rows > 0:
                        st.dataframe(er.head(200), use_container_width=True)
                        st.download_button("Download error report", data=er.to_csv(index=False).encode("utf-8"), file_name=f"{sheet}_error_report.csv", mime="text/csv")
                    else:
                        st.info("No missing values found for selected columns.")

        # Visualization
        with nested[2]:
            st.header(f"{active} -- Visualization")
            sheet = st.selectbox("Select sheet", options=st.session_state.get("sheet_names", []), key=f"viz_sheet_{active}")
            if sheet and st.session_state.get("results") and sheet in st.session_state["results"]:
                show_bar_chart(st.session_state["results"][sheet]["attr_overview"], backing_df=st.session_state.get("workbook_dict", {}).get(sheet))
            else:
                st.info("Select a sheet with results to view visualizations.")

        # Online Editor
        with nested[3]:
            st.header(f"{active} -- Online Editor")
            st.write("Edit incomplete rows inline and download updated workbook.")
            sheet = st.selectbox("Select sheet to edit", options=st.session_state.get("sheet_names", []), key=f"editor_sheet_{active}")
            if not sheet:
                st.info("Select a sheet to edit.")
            else:
                data = st.session_state["results"].get(sheet, None)
                if not data or data["incomplete_df"].empty:
                    st.info("No incomplete rows to edit for this sheet.")
                else:
                    df_original = st.session_state["workbook_dict"][sheet].copy()
                    row_indices = (data["incomplete_df"]["Row No."].astype(int) - 1).tolist()
                    editable_subset = df_original.iloc[row_indices].copy()
                    editable_subset.insert(0, "__ROW_INDEX__", editable_subset.index.astype(int))
                    for col in editable_subset.columns:
                        if col != "__ROW_INDEX__":
                            editable_subset[col] = editable_subset[col].astype(str).replace("nan", "")
                    edited = st.data_editor(editable_subset, use_container_width=True, key=f"editor_{active}_{sheet}")
                    if st.button("Save Changes", key=f"save_{active}_{sheet}"):
                        for _, r in edited.iterrows():
                            abs_idx = int(r["__ROW_INDEX__"])
                            to_write = r.drop(labels="__ROW_INDEX__")
                            for col in to_write.index:
                                val = r[col]
                                if val == "" or val is None:
                                    out_val = None
                                else:
                                    out_val = val
                                to_write[col] = out_val
                            st.session_state["workbook_dict"][sheet].loc[abs_idx, to_write.index] = to_write.values
                        st.success("Changes saved to the in-memory workbook.")
                        df_new = st.session_state["workbook_dict"][sheet]
                        stats, incomplete_df, attr_overview, remediation_bytes = run_completeness_check(df_new)
                        st.session_state["results"][sheet] = {
                            "stats": stats,
                            "incomplete_df": incomplete_df,
                            "attr_overview": attr_overview,
                            "remediation_bytes": remediation_bytes,
                        }
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                            for sn, dff in st.session_state["workbook_dict"].items():
                                dff.to_excel(writer, index=False, sheet_name=sn[:31])
                        st.download_button("Download Updated Workbook (Excel)", data=buf.getvalue(), file_name=f"{sheet}_updated.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Rerun Check
        with nested[4]:
            st.header(f"{active} -- Rerun Check")
            sheet = st.selectbox("Select sheet to rerun", options=st.session_state.get("sheet_names", []), key=f"rerun_sheet_{active}")
            if sheet:
                if st.button("Rerun Completeness for this sheet", key=f"rerun_btn_{active}_{sheet}"):
                    df = st.session_state["workbook_dict"][sheet]
                    stats, incomplete_df, attr_overview, remediation_bytes = run_completeness_check(df)
                    st.session_state["results"][sheet] = {
                        "stats": stats,
                        "incomplete_df": incomplete_df,
                        "attr_overview": attr_overview,
                        "remediation_bytes": remediation_bytes,
                        "error_report": None,
                        "error_report_params": None
                    }
                    st.success("Rerun complete.")

# End of uif.py