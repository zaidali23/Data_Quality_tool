import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Data Quality Assessment Tool",
    layout="wide",
    page_icon="📊"
)

# --- Header container ---
with st.container():
    col1, col2 = st.columns([1, 5])  # left = EY, right = Motto

    with col1:
        st.image("eyLogo.png", width=120)   # EY logo

    with col2:
        st.image("eyLogo2.png", width=400)  # Motto logo

# --- Style header container ---
st.markdown("""
    <style>
    /* Style ONLY the first container */
    [data-testid="stVerticalBlock"] > div:first-child {
        background-color: #393939; /* dark gray */
        height: 161px;             /* fixed header height */
        display: flex;
        align-items: center;
        justify-content: space-between; /* EY left, Motto right */
        margin: 0 !important;     /* remove Streamlit's side margins */
        padding: 0 20px;           /* add breathing room inside */
        width: 100% !important;    /* stretch full width */
    }
    </style>
""", unsafe_allow_html=True)