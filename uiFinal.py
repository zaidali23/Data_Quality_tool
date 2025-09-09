"""
Notes:
** This code still needs cc_errorReport.py and cc.py 
** Image eyLogo needs to be in the same folder as the code.

"""

import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt ##--------- for when we add visualisation
from typing import List, Dict, Tuple

# import completeness check implementations from cc_errorReport.py and cc.py

completeness_available = False
for modname in ("cc_errorReport", "cc"):
    try:
        mod = __import__(modname)
        run_completeness_check = getattr(mod, "run_completeness_check")
        completeness_available = True
        completeness_source = modname
        break
    except Exception:
        run_completeness_check = None
        completeness_source = None

## Page config & CSS for header and logo

st.set_page_config (page_title="Data Quality Tool", layout="wide")

CSS = 
"""
<style>
html, body, .reportview-container .main 
{
  backgroundColor: #000000;
  primaryColor: #FFD500;
  textColor: #000000
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
}

.banner 
{
 background:#FFD500;
 border-radius:8px;
 padding:10px 14px; 
 display:flex;
 align-items:center;
 justify-content:space-between;
}
.stButton>button, .stDownloadButton>button 
{
 background: #FFD400 !important;
 color: #000 !important;
 font-weight: 700;
 border-radius: 8px;
} 
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
                    rowd["Error Type"] = "Completeness Check"
                    rowd["Error Message"] = f"Data Entry missing for column - {attr}"
                    rows.append(rowd)
            else:
                for col in selected_columns:
                    val = src.get(col, None)
                    if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                        rowd = src.to_dict()
                        rowd["Error Type"] = "Completeness Check"
                        rowd["Error Message"] = f"Data Entry missing for column - {attr}"
                        rows.append(rowd)
    else:
        for idx, src in df.iterrows():
            for col in selected_columns:
                val = src.get(col, None)
                if pd.isna(val) or (isinstance(val, str) and val.strip() == ""):
                    rowd = src.to_dict()
                    rowd["Error Type"] = "Completeness Check"
                    rowd["Error Value"] = f"Data Entry missing for column - {attr}"
                    rows.append(rowd)

    if not rows:
        out_cols = list(df.columns) + ["Error Type", "Error Message"]
        return pd.DataFrame(columns=out_cols)
    out_df = pd.DataFrame(rows)
    final_cols = list(df.columns) + ["Error Type", "Error Message"]
    final_cols = [c for c in final_cols if c in out_df.columns]
    return out_df[final_cols].reset_index(drop=True)

def df_to_excel_bytes(d: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        d.to_excel(writer, index=False, sheet_name="error_report")
    return buf.getvalue()

# ---------- Normalizer ----------
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

# ---------- header ----------

def ui_header():
    colLogo, colBanner = st.columns([1, 7])
    left_img = None
    try:
        with open("eyLogo.png", "rb") as f: left_img = f.read()
    except Exception:
        left_img = None

    with colLogo:
        if left_img:
            st.image(left_img, width=80)
        else:
            st.markdown('<div style="background:#000;color:#FFD400;padding:8px;border-radius:6px;font-weight:900;text-align:center">EY</div>', unsafe_allow_html=True)
    with colBanner:
        st.markdown('<div class="banner"><div style="font-weight:800;font-size:20px;color:#000">Data Quality Assessment Tool</div>'
                    '<div style="font-size:13px;color:#222">Polished UI -- completeness source: <b>{}</b></div></div>'.format(completeness_source),
                    unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#000;background:#FFD500;padding:6px 8px;border-radius:6px;font-weight:700;text-align:right">Integrity • Insight</div>', unsafe_allow_html=True)

# ---------- main page ----------

ui_header()

right_col = st.container()

with right_col:
      tabs = st.tabs(["DQT", "Remediation", "Data Visualisation",])

        # Upload bar 

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
            st.write("Select sheets to include in checks:")
            c1, c2 = st.columns([1:1])
            with c1:
                st.subheader("Quality Checks")
                checks = st.multiselect("Choose checks", ["Completeness Check", "Consistency Check", "Lineage Check", "Validation Check"], default=["Completeness Check"], key="checks_select")
            with c2:
                st.subheader("Sheets")
                sheet_list = st.session_state.get("sheet_names", [])
                sheets = st.multiselect("Select sheets", options=sheet_list, key="sheet_multiselect")
            if st.button("Run Check"):
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
                        st.success("Checks complete. Scroll down for summary")
                    
                        st.header("Summary")
                        if not st.session_state.get("results"):
                          st.info("Run checks from the Upload tab to populate results.")
                        else:
                          for sheet, data in st.session_state["results"].items():
                        st.subheader(f"Sheet: {sheet}")
                        kpi_cards(data.get("stats", {}))
                        # pass backing sheet df for robust normalization
                        backing_df = st.session_state.get("workbook_dict", {}).get(sheet, None)
                        
                        # Remediation 
                        with tab[1]:
                        st.header(f"{active} -- Data Remediation")
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

                        
                      # VISUALIZATION
                      
        with tabs[2]:
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

       
                    # Error Report / Preview

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
