import streamlit as st
import pandas as pd
from typing import Optional


# --- placeholders / small framework ---------------------------------
@st.cache_resource
def load_model():
    """Placeholder model loader. Replace with actual model loading.

    Example:
        return joblib.load("models/model.joblib")
    """
    return None


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV into a DataFrame. Returns empty DataFrame on failure."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        # try resetting file pointer and a fallback read
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            return pd.DataFrame()


def extract_events_from_df(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Placeholder extraction function.

    Replace this with your real extraction/ML inference. Should return a DataFrame
    of results (one row per input or extracted event rows).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # simple demo: add a placeholder prediction column
    out["is_event_pred"] = 0
    return out


def main() -> None:
    st.title("Event Extracting Model")
    st.write("Hello!")

    uploaded_file = st.file_uploader("Upload a raw data file", type=["csv"])
    df = parse_uploaded_file(uploaded_file)

    if not df.empty:
        st.subheader("Preview")
        st.dataframe(df.head())

    # load model (cached between reruns)
    model = load_model()

    if st.button("Extract Events"):
        if df.empty:
            st.warning("Please upload a CSV file first.")
        else:
            with st.spinner("Extracting events..."):
                result = extract_events_from_df(df, model)
            st.success("Done!")
            st.dataframe(result)

    st.markdown("---")
    st.info("This is a demo. Need to replace `load_model` and `extract_events_from_df` with trained ML model.")


if __name__ == "__main__":
    main()