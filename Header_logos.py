import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Data Quality Assessment Tool",
    layout="wide",
    page_icon="📊"
)

# --- Header container ---
with st.container():
    col1, col2 = st.columns([1, 5])  # adjust ratio

    with col1:
        st.image("eyLogo.png", width=120)   # EY logo (left)

    with col2:
        st.image("eyLogo2.png", width=400)  # Motto image (right)

# --- Style header container ---
st.markdown("""
    <style>
    /* Style ONLY the first container */
    [data-testid="stVerticalBlock"] > div:first-child {
        background-color: #393939; /* dark gray */
        height: 161px;             /* fixed height */
        display: flex;
        align-items: center;       /* vertical centering */
    }

    /* Fine-tune EY logo (push slightly right) */
    [data-testid="stVerticalBlock"] img:first-of-type {
        margin-left: 20px;
    }

    /* Fine-tune Motto (stick to right edge) */
    [data-testid="stVerticalBlock"] img:last-of-type {
        margin-left: auto;   /* push to right */
        margin-right: 40px;  /* spacing from edge */
    }
    </style>
""", unsafe_allow_html=True)