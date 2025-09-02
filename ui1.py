import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Data Quality Assessment Tool",
    layout="wide",
    page_icon="📊"
)

# --- Header container ---
with st.container():
    col1, col2 = st.columns([1, 5])  # adjust ratio to taste

    with col1:
        st.image("eyLogo.png", width=120)  # EY logo (left)

    with col2:
        st.image("motto.png", width=400)   # Motto image (right)

# --- Style the container as a header ---
st.markdown("""
    <style>
    /* Style first container (header) */
    [data-testid="stVerticalBlock"] > div:first-child {
        background-color: #393939; /* dark gray */
        height: 161px;             /* header height */
        display: flex;
        align-items: center;       /* vertical centering */
        justify-content: space-between;
        padding: 0 40px;
    }
    </style>
""", unsafe_allow_html=True)
