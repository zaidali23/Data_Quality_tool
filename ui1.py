import streamlit as st

# --- Custom CSS ---
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
        align-items: center;          /* vertical centering */
        justify-content: space-between; /* left + right logos */
        padding: 0 40px;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- H1 container with two logos ---
with st.container():
    # open header <div>
    st.markdown('<div class="top-header">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])  # two cols inside header
    with col1:
        st.image("logo_left.png", width=180)
    with col2:
        st.image("logo_right.png", width=180)

    # close header </div>
    st.markdown('</div>', unsafe_allow_html=True)