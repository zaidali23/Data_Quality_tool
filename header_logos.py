import streamlit as st

# --- Header container ---
with st.container():
    col1, col2 = st.columns([1, 5])  # adjust ratio to control spacing

    with col1:
        st.image("eyLogo.png", width=120)   # EY Logo (left, inset a bit)

    with col2:
        st.image("eyLogo2.png", width=400)  # Motto (sticks right)

# --- Style header container ---
st.markdown("""
    <style>
    [data-testid="stVerticalBlock"] > div:first-child {
        background-color: #393939; 
        height: 161px; 
        display: flex; 
        align-items: center;
        justify-content: space-between; /* left + right placement */
        padding: 0 40px; /* add breathing room */
    }
    [data-testid="stVerticalBlock"] img {
        margin-top: auto;
        margin-bottom: auto; /* vertical centering */
    }
    </style>
""", unsafe_allow_html=True)