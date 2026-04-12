import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run /abs/path/to/app/streamlit_app.py` from outside the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference_service import (
    load_model as svc_load_model,
    validate_input,
    extract_events_from_df as svc_extract_events_from_df,
    format_events_for_ui,
    dataframe_to_bytes,
    get_required_columns,
    get_artifact_status,
)


# demo model builder (sklearn) used for quick local demos when real model not available
@st.cache_resource
def build_demo_model():
    class KeywordDemoModel:
        KEYWORDS = (
            "open house",
            "event",
            "session",
            "workshop",
            "tour",
            "webinar",
        )

        def predict(self, texts):
            preds = []
            for text in texts:
                value = str(text).strip().lower()
                preds.append(int(any(keyword in value for keyword in self.KEYWORDS)))
            return preds

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return KeywordDemoModel()

    texts = [
        "open house saturday 10am",
        "event on monday at 5pm",
        "virtual info session july 12",
        "this is not an event",
        "reminder submit application",
    ]
    labels = [1, 1, 1, 0, 0]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(texts, labels)
    return pipe


def demo_extract_events(df: pd.DataFrame, demo_pipe) -> list[dict]:
    """Run a tiny demo classifier over `text_context` and return list[dict] events."""
    if df is None or df.empty or demo_pipe is None:
        return []

    working_df = df.copy()
    if "source" not in working_df.columns:
        working_df["source"] = "uploaded_file"
    else:
        working_df["source"] = working_df["source"].fillna("uploaded_file")

    events = []
    for src, grp in working_df.groupby("source", dropna=False):
        X = grp["text_context"].fillna("").astype(str)
        try:
            preds = demo_pipe.predict(X)
        except Exception:
            preds = [0] * len(X)

        for local_idx, (idx, row) in enumerate(grp.iterrows()):
            p = preds[local_idx] if local_idx < len(preds) else 0
            if int(p) == 1:
                evt = {"source": src, "node_range": f"{idx}-{idx}"}
                # preserve any provided metadata columns
                for f in ("name", "date", "time", "location"):
                    if f in row.index and pd.notna(row[f]):
                        evt[f] = row[f]
                events.append(evt)

    return events


@st.cache_data
def parse_uploaded_file(file_bytes: bytes) -> pd.DataFrame:
    try:
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes))
    except Exception:
        return pd.DataFrame()


@st.cache_resource
def load_cached_model():
    try:
        return svc_load_model()
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Event Extraction Model", layout="wide")

    st.title("Event Extraction Model")
    st.write("Upload a CSV file and run the trained event-extraction pipeline.")
    st.caption("For a quick smoke test, upload `tests/fixtures/streamlit/sample_for_ui_min.csv`.")

    model_bundle = load_cached_model()
    demo_model = build_demo_model()
    artifact_status = get_artifact_status()

    # allow user to explicitly (re)load the real model into session state
    if "model_bundle" not in st.session_state:
        st.session_state.model_bundle = model_bundle

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Load real model"):
            try:
                st.session_state.model_bundle = svc_load_model()
            except Exception:
                st.session_state.model_bundle = None
            st.experimental_rerun()
    with col2:
        if st.session_state.model_bundle is None:
            if demo_model is not None:
                st.info("Real model not loaded. Demo model is ready for smoke testing.")
            else:
                st.warning("Real model not loaded and the demo model is unavailable.")
        else:
            st.success("Real model loaded into session.")

    with st.expander("Model status", expanded=False):
        st.write(f"Checkpoint found: **{artifact_status['checkpoint_exists']}**")
        st.write(f"Classifier found: **{artifact_status['classifier_exists']}**")
        st.write(f"PyTorch available: **{artifact_status['torch_available']}**")
        st.write(f"Real model ready: **{artifact_status['real_model_ready']}**")
        st.write(f"Device: **{artifact_status['device']}**")
        st.code(artifact_status["checkpoint_path"])
        st.code(artifact_status["classifier_path"])

    uploaded_file = st.file_uploader("Upload a raw data file", type=["csv"])

    df = pd.DataFrame()
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df = parse_uploaded_file(file_bytes)

        if df.empty:
            st.error("Failed to read the uploaded CSV file. Please check the file format.")
        else:
            st.subheader("Preview")
            st.dataframe(df.head(20), use_container_width=True)

            with st.expander("Detected columns", expanded=False):
                st.write(list(df.columns))

            with st.expander("Required columns", expanded=False):
                st.write(get_required_columns(st.session_state.get("model_bundle")))
                if st.session_state.get("model_bundle") is None:
                    st.caption("Demo mode only needs the columns above. `source` is optional.")

    if st.button("Extract Events", type="primary"):
        if uploaded_file is None or df.empty:
            st.warning("Please upload a valid CSV file first.")
            return

        model_bundle = st.session_state.get("model_bundle")
        if model_bundle is None and demo_model is None:
            st.error("Neither the real model bundle nor the demo model is available.")
            return

        ok, msg = validate_input(df, model_bundle)
        if not ok:
            st.error(msg)
            return

        try:
            with st.spinner("Running inference..."):
                if model_bundle is None:
                    events = demo_extract_events(df, demo_model)
                else:
                    events = svc_extract_events_from_df(df, model_bundle)
                final_df = format_events_for_ui(events)

            if model_bundle is None:
                st.success("Done! Demo mode completed.")
            else:
                st.success("Done!")

            if final_df.empty:
                st.info("No events detected.")
            else:
                st.subheader("Extracted Events")
                st.dataframe(final_df, use_container_width=True)

                st.download_button(
                    "Download extracted events (CSV)",
                    data=dataframe_to_bytes(final_df),
                    file_name="extracted_events.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Inference failed: {e}")

    st.markdown("---")
    st.caption("Powered by the trained DOM extractor + field classifier pipeline.")


if __name__ == "__main__":
    main()
