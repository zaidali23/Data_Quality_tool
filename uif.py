"""
uif.py -- Data Quality Assessment Tool (Streamlit)

Run:
    streamlit run uif.py

What changed from the demo:
- Filename changed to `uif.py` (as requested).
- Header now uses local images `eyLogo.png` (left) and `eyLogo2.png` (right) -- place both images
  in the same directory as uif.py.
- Completeness logic: attempts to import from `ccUpd.py` first, then `cc.py`. If neither is available,
  falls back to the built-in stub so the UI still runs.

Notes for your meeting:
- Drop `eyLogo.png` and `eyLogo2.png` in the same folder.
- Make sure `cc.py` and/or `ccUpd.py` are present in the same folder (they should expose
  `run_completeness_check(df)` with the signature used in the comments).
- Run with: `streamlit run uif.py`.

Where to paste your completeness code:
- If you want to override the fallback, either:
    * Ensure ccUpd.py exports run_completeness_check, or
    * Put your function into cc.py as run_completeness_check, or
    * Paste your function directly into this file where the fallback stub is defined
      (search for "FALLBACK STUB" in the file).

Minimal deps: pandas, streamlit, openpyxl, xlsxwriter, matplotlib
"""

import io
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# ---------- Attempt to import completeness from ccUpd then cc ----------
completeness_available = False
try:
    # Prefer the edited/updated completeness implementation
    from ccUpd import run_completeness_check  # type: ignore
    completeness_available = True
except Exception:
    try:
        from cc import run_completeness_check  # type: ignore
        completeness_available = True
    except Exception:
        completeness_available = False

# ==== FALLBACK STUB (only used if neither ccUpd nor cc is importable) ====
if not completeness_available:
    def run_completeness_check(df: pd.DataFrame):
        """
        FALLBACK STUB: realistic placeholder to let the UI run end-to-end.
        Signature expected:
            run_completeness_check(df: pd.DataFrame) -> (stats_dict, incomplete_df, attr_overview_df, remediation_bytes)
        """
        na_mask = df.isna() | (df.astype(str).applymap(lambda x: x.strip() == "" if isinstance(x, str) else False))
        row_missing_counts = na_mask.sum(axis=1)
        incomplete_idx = row_missing_counts[row_missing_counts > 0].index.tolist()

        # Build incomplete_df
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

        # attr_overview
        attr_overview = pd.DataFrame({
            "attribute": df.columns,
            "missing_count": na_mask.sum(axis=0).astype(int).values
        }).sort_values("missing_count", ascending=False).reset_index(drop=True)

        # stats
        total_cells = int(df.size) if df.size else 0
        total_missing = int(na_mask.sum().sum()) if df.size else 0
        stats = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_cells": total_cells,
            "total_missing": total_missing,
            "percent_missing": round(100 * (total_missing / total_cells), 2) if total_cells else 0.0
        }

        # remediation bytes: write incomplete rows to excel
        buf = io.BytesIO()
        try:
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                if not incomplete_df.empty:
                    incomplete_df.to_excel(writer, sheet_name="IncompleteRows", index=False)
                else:
                    pd.DataFrame().to_excel(writer, sheet_name="IncompleteRows", index=False)
            remediation_bytes = buf.getvalue()
        except Exception:
            remediation_bytes = b""

        return stats, incomplete_df, attr_overview, remediation_bytes

# ---------- Styling (CSS) ----------
st.set_page_config(page_title="Data Quality Assessment Tool", layout="wide")
CSS = """
<style>
/* Page background and text */
.reportview-container, .main, header, .block-container {
    background-color: #000000;
    color: #FFFFFF;
}
body {
    color: #ffffff;
    font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header banner style */
.header-banner {
    background: linear-gradient(90deg, #FFD400 0%, #FFD400 100%);
    padding: 14px 18px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 12px;
}

/* Logo placeholders - images are shown instead */
.banner-title {
    color: #000;
    font-weight: 800;
    font-size: 20px;
}

/* Left nav */
.left-nav {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
}
.nav-btn {
    background: #FFD400;
    color: #000;
    padding: 10px 12px;
    border-radius: 6px;
    font-weight: 700;
    cursor: pointer;
    border: none;
    text-align: left;
    width: 100%;
}
.nav-btn.grey {
    background: #333333;
    color: #fff;
}
.nav-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(255, 212, 0, 0.12);
}
.nav-btn:focus {
    outline: 2px solid #fff;
}

/* Upload bar */
.upload-bar {
    background: #ffffff;
    color: #000000;
    padding: 10px;
    border-radius: 6px;
}

/* Card styles */
.card {
    background: #0a0a0a;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
}
.kpi {
    font-size: 18px;
    font-weight: 700;
}

/* small helpers */
.small-grey {
    color: #9b9b9b;
    font-size: 13px;
}

/* Ensure multiselect and buttons look crisp on dark bg */
.stMultiSelect, .stButton>button {
    border-radius: 6px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------- UI components ----------
def ui_header():
    """Header with left EY logo (eyLogo.png) and right motto (eyLogo2.png)."""
    # Use images if present, otherwise fallback text blocks.
    left_img = None
    right_img = None
    try:
        left_img = open("eyLogo.png", "rb").read()
    except Exception:
        left_img = None
    try:
        right_img = open("eyLogo2.png", "rb").read()
    except Exception:
        right_img = None

    # Build HTML layout for banner: left image, title, right image
    left_html = f'<img src="data:image/png;base64,{st.image(left_img).image_to_bytes().decode()}" style="height:48px;">' if left_img else '<div class="ey-logo">EY</div>'
    # The above is a clever attempt, but Streamlit doesn't provide image->base64 helper directly.
    # Simpler: use st.image for left and right and custom html for center.
    with st.container():
        cols = st.columns([1, 8, 1])
        with cols[0]:
            if left_img:
                st.image(left_img, width=72, use_column_width=False)
            else:
                st.markdown('<div class="ey-logo" style="background:#000;color:#FFD400;padding:8px;border-radius:6px;font-weight:900">EY</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(
                """
                <div style="display:flex;flex-direction:column;justify-content:center;">
                    <div class="banner-title">Data Quality Assessment Tool</div>
                    <div class="small-grey">Polished UI -- plug in your completeness function (cc.py / ccUpd.py)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with cols[2]:
            if right_img:
                st.image(right_img, width=120, use_column_width=False)
            else:
                st.markdown('<div style="color:#000;background:#FFD400;padding:6px 8px;border-radius:6px;font-weight:700;text-align:right">Integrity • Insight</div>', unsafe_allow_html=True)

def left_nav(selected: str) -> str:
    """Left vertical navigation simulated by a radio (we style visually)."""
    nav_items = ["Home", "Completeness", "Consistency", "Lineage", "Validation", "All Checks"]
    choice = st.radio("", nav_items, index=nav_items.index(selected) if selected in nav_items else 0, key="left_nav_radio", label_visibility="collapsed")
    return choice

def parse_workbook(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Read Excel workbook into dict of DataFrames. Returns (sheets_dict, sheet_names)."""
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheets = {sn: pd.read_excel(uploaded_file, sheet_name=sn) for sn in xls.sheet_names}
        return sheets, list(xls.sheet_names)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return {}, []

def group_workstreams(sheet_names: List[str]) -> Dict[str, List[str]]:
    """
    Detect workstream patterns in sheet names (heuristic grouping).
    Returns mapping group_name -> [sheet_names].
    """
    groups = {}
    for sn in sheet_names:
        m = re.match(r"^(\d+)[\-_ ]*(.*)$", sn)
        if m:
            key = m.group(1)
        else:
            key = sn.split()[0] if sn.strip() else "misc"
        groups.setdefault(key, []).append(sn)
    return groups

# Error report utility (one missing attribute -> one row)
def generate_error_report_from_incomplete(df: pd.DataFrame, incomplete_df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """
    Build error_report where each missing attribute becomes its own row.
    Columns: all original df columns + "Error Type" and "Error Value".
    """
    rows = []
    # detect row no column
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
                parsed = str(r[miss_col]).strip()
                parts = [p.strip() for p in parsed.replace(";", ",").split(",") if p.strip()]
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

# Small UI building blocks
def kpi_cards(stats: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='card'><div class='kpi'>Rows</div><div>{stats.get('rows', 0)}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='kpi'>Columns</div><div>{stats.get('columns', 0)}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='kpi'>Missing Cells</div><div>{stats.get('total_missing', 0)}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card'><div class='kpi'>% Missing</div><div>{stats.get('percent_missing', 0.0)}%</div></div>", unsafe_allow_html=True)

def show_attr_overview(attr_overview: pd.DataFrame):
    st.write("Missing Count per Attribute")
    st.dataframe(attr_overview.head(200), use_container_width=True)

def show_bar_chart(attr_overview: pd.DataFrame):
    top = attr_overview.head(10)
    if top.empty:
        st.info("No attribute overview to chart.")
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(top["attribute"].astype(str), top["missing_count"].astype(int))
    ax.invert_yaxis()
    ax.set_xlabel("Missing Count")
    ax.set_title("Top attributes by missing count")
    st.pyplot(fig)

def df_to_excel_bytes(d: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        d.to_excel(writer, index=False, sheet_name="error_report")
    return buf.getvalue()

# ---------- Main layout ----------
ui_header()

# Layout columns: left navigation (approx 20%) and main content (80%)
left_col, right_col = st.columns([1, 4])

with left_col:
    st.markdown("<div class='left-nav'>", unsafe_allow_html=True)
    active = left_nav(selected="Home")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    if active == "Home":
        tabs = st.tabs(["Upload", "Assessment Summary", "Custom Check", "Visualization", "Downloads"])
        # Upload tab
        with tabs[0]:
            st.markdown('<div class="upload-bar">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Excel workbook", type=["xlsx"], key="main_upload")
            if uploaded_file is not None:
                st.success(f"File uploaded: {getattr(uploaded_file, 'name', 'uploaded_file')}")
                try:
                    workbook_dict, sheet_names = parse_workbook(uploaded_file)
                    st.session_state["workbook"] = workbook_dict
                    st.session_state["sheet_names"] = sheet_names
                except Exception as e:
                    st.error(f"Error parsing workbook: {e}")
            else:
                st.info("Upload an .xlsx workbook to begin (max ~200MB).")
            st.markdown("</div>", unsafe_allow_html=True)

            if st.session_state.get("sheet_names"):
                groups = group_workstreams(st.session_state["sheet_names"])
                st.write("Detected sheet groups / workstreams:")
                for g, items in groups.items():
                    st.write(f"- **{g}**: {', '.join(items)}")

            st.markdown("---")
            st.write("Select sheets / workstreams to include in checks:")
            sheet_selection = st.multiselect("Sheets", options=st.session_state.get("sheet_names", []), key="sheet_select_home")
            checks = st.multiselect("Checks to run", ["Completeness Check", "Consistency Check", "Lineage Check", "Validation Check"], default=["Completeness Check"])

            # Run Check Now
            if st.button("Run Check Now"):
                if not st.session_state.get("workbook"):
                    st.error("Please upload workbook first.")
                elif not sheet_selection:
                    st.error("Select at least one sheet to run checks on.")
                elif "Completeness Check" not in checks:
                    st.warning("No completeness selected (UI demo focuses on completeness).")
                else:
                    st.session_state["results"] = {}
                    progress = st.progress(0)
                    total = len(sheet_selection)
                    for i, sheet in enumerate(sheet_selection, start=1):
                        df = st.session_state["workbook"][sheet]
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
                            "error_report_params": None,
                        }
                        progress.progress(int(100 * i / total))
                    st.success("Checks complete. Scroll down for results.")

        # Assessment Summary tab
        with tabs[1]:
            st.header("Assessment Summary")
            if not st.session_state.get("results"):
                st.info("Run checks from the Upload tab to populate results.")
            else:
                for sheet, data in st.session_state["results"].items():
                    st.subheader(f"Sheet: {sheet}")
                    kpi_cards(data.get("stats", {}))
                    if not data.get("attr_overview", pd.DataFrame()).empty:
                        show_attr_overview(data["attr_overview"])
                        show_bar_chart(data["attr_overview"])

        # Custom Check tab
        with tabs[2]:
            st.header("Custom Check")
            st.info("Placeholder -- paste custom check UI components here.")
            st.write("Tip: Add a form for user-defined rules. Use function logic_* to wire checks.")

        # Visualization tab
        with tabs[3]:
            st.header("Visualization")
            st.info("High-level visualizations appear here.")
            if st.session_state.get("results"):
                first = next(iter(st.session_state["results"].values()))
                if not first["attr_overview"].empty:
                    show_bar_chart(first["attr_overview"])
                else:
                    st.info("No attribute overview yet. Run checks first.")

        # Downloads tab
        with tabs[4]:
            st.header("Downloads")
            if not st.session_state.get("results"):
                st.info("No outputs yet. Run checks to enable downloads.")
            else:
                for sheet, data in st.session_state["results"].items():
                    st.write(f"**{sheet}**")
                    if data["incomplete_df"] is not None and not data["incomplete_df"].empty:
                        st.download_button(
                            "Download Incomplete Rows (CSV)",
                            data=data["incomplete_df"].to_csv(index=False).encode("utf-8"),
                            file_name=f"{sheet}_IncompleteRows.csv",
                            mime="text/csv"
                        )
                    if data["remediation_bytes"]:
                        st.download_button(
                            "Download Remediation File (Excel)",
                            data=data["remediation_bytes"],
                            file_name=f"{sheet}_Remediation.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

    else:
        # QC pages with nested tabs
        qc_tabs = ["Assessment Summary", "Error Report/Preview", "Visualization", "Online Editor", "Rerun Check"]
        nested = st.tabs(qc_tabs)

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
                show_attr_overview(data.get("attr_overview", pd.DataFrame()))

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
                    df = st.session_state["workbook"][sheet]
                    default_cols = [c for c in df.columns]
                    chosen = st.multiselect("Columns to check", options=list(df.columns), default=default_cols, key=f"er_cols_{active}")
                    error_report = generate_error_report_from_incomplete(df, data["incomplete_df"], chosen)
                    total_rows = len(error_report)
                    st.write(f"Error report rows (one per missing attribute): **{total_rows}**")
                    if total_rows > 0:
                        st.dataframe(error_report.head(200), use_container_width=True)
                        st.download_button(
                            "Download error report",
                            data=error_report.to_csv(index=False).encode("utf-8"),
                            file_name=f"{sheet}_error_report.csv",
                            mime="text/csv",
                            key=f"dl_er_{active}_{sheet}"
                        )
                    else:
                        st.info("No missing values found for selected columns.")

        # Visualization
        with nested[2]:
            st.header(f"{active} -- Visualization")
            sheet = st.selectbox("Select sheet", options=st.session_state.get("sheet_names", []), key=f"viz_sheet_{active}")
            if sheet and st.session_state.get("results") and sheet in st.session_state["results"]:
                show_bar_chart(st.session_state["results"][sheet]["attr_overview"])
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
                    df_original = st.session_state["workbook"][sheet].copy()
                    row_indices = (data["incomplete_df"]["Row No."].astype(int) - 1).tolist()
                    editable_subset = df_original.iloc[row_indices].copy()
                    editable_subset.insert(0, "__ROW_INDEX__", editable_subset.index.astype(int))
                    for col in editable_subset.columns:
                        if col != "__ROW_INDEX__":
                            editable_subset[col] = editable_subset[col].astype(str).replace("nan", "")
                    edited = st.data_editor(editable_subset, use_container_width=True, key=f"editor_{active}_{sheet}")
                    if st.button("Save Changes", key=f"save_{active}_{sheet}"):
                        for _, r in edited.iterrows():
                            idx = int(r["__ROW_INDEX__"])
                            for col in editable_subset.columns:
                                if col == "__ROW_INDEX__":
                                    continue
                                val = r[col]
                                if val == "" or val is None:
                                    out_v = None
                                else:
                                    out_v = val
                                st.session_state["workbook"][sheet].at[idx, col] = out_v
                        st.success("Changes saved to in-memory workbook.")
                        # recompute results for this sheet
                        df_new = st.session_state["workbook"][sheet]
                        stats, incomplete_df, attr_overview, remediation_bytes = run_completeness_check(df_new)
                        st.session_state["results"][sheet] = {
                            "stats": stats,
                            "incomplete_df": incomplete_df,
                            "attr_overview": attr_overview,
                            "remediation_bytes": remediation_bytes,
                        }
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                            for sn, dff in st.session_state["workbook"].items():
                                dff.to_excel(writer, index=False, sheet_name=sn[:31])
                        buf.seek(0)
                        st.download_button("Download Updated Workbook (Excel)", data=buf.getvalue(), file_name=f"{sheet}_updated.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Rerun Check
        with nested[4]:
            st.header(f"{active} -- Rerun Check")
            sheet = st.selectbox("Select sheet to rerun", options=st.session_state.get("sheet_names", []), key=f"rerun_sheet_{active}")
            if sheet:
                if st.button("Rerun Completeness for this sheet", key=f"rerun_btn_{active}_{sheet}"):
                    df = st.session_state["workbook"][sheet]
                    stats, incomplete_df, attr_overview, remediation_bytes = run_completeness_check(df)
                    st.session_state["results"][sheet] = {
                        "stats": stats,
                        "incomplete_df": incomplete_df,
                        "attr_overview": attr_overview,
                        "remediation_bytes": remediation_bytes,
                    }
                    st.success("Rerun complete.")
# End of file