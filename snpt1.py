import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Data Quality Assessment Tool",
    layout="wide",   # page width = wide
    page_icon="📊"
)

# --- Theme Config (in .streamlit/config.toml) ---
"""
[theme]
primaryColor = "#FFD700"        # Yellow
backgroundColor = "#000000"     # Black
secondaryBackgroundColor = "#FFFFFF" # White (for reference clarity)
textColor = "#FFFFFF"           # White text
"""

# --- header with logo ---

# --- Minimal Header CSS ---
st.markdown(
    """
    <style>
    .top-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 225px;
        background-color: #393939; /* dark gray */
        display: flex;
        align-items: center;           /* vertical centering */
        justify-content: space-between; /* left + right logos */
        padding: 0 40px;
        z-index: 9999;
    }
    .top-header img {
        max-height: 180px;  /* scale logos */
        object-fit: contain;
    }
    </style>

    <!-- Header with two logos -->
    <div class="top-header">
        <img src="logo_left.png" alt="Left Logo">
        <img src="logo_right.png" alt="Right Logo">
    </div>
    """,
    unsafe_allow_html=True
)