import streamlit as st

st.set_page_config(
    page_title="Hotel Booking Intelligence",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 Hotel Booking Intelligence Dashboard")

st.markdown("""
Welcome to our end-to-end Hotel Booking project.

Use the sidebar to navigate between pages:

- **AI Prediction Center**
- **Loyalty Analysis**
- **Time Series Analysis**
- **Similarity Search**
""")

st.info("Select a page from the sidebar to start.")