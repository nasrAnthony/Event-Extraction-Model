import io
import pandas as pd
from pathlib import Path
import importlib.util

# This is a basic unit test for the data processing function in streamlit_app,
# not a full Streamlit UI test.

# load the app/streamlit_app.py module by path to avoid import name collision with top-level app.py
HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "app" / "streamlit_app.py"
spec = importlib.util.spec_from_file_location("streamlit_app", MODULE_PATH)
streamlit_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streamlit_app)


def test_parse_uploaded_file_reads_csv():
    csv = "text_context,tag,event_id\nhello,tag1,\nworld,tag2,e1\n"
    f = io.BytesIO(csv.encode())
    df = streamlit_app.parse_uploaded_file(f)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["text_context", "tag", "event_id"]


def test_parse_uploaded_file_none_returns_empty():
    df = streamlit_app.parse_uploaded_file(None)
    assert df.empty


def test_extract_events_adds_column():
    df = pd.DataFrame({"text_context": ["a", "b"], "tag": ["t", "t2"], "event_id": [None, "e1"]})
    out = streamlit_app.extract_events_from_df(df, model=None)
    assert "is_event_pred" in out.columns
    assert len(out) == 2
    assert out["is_event_pred"].tolist() == [0, 0]
