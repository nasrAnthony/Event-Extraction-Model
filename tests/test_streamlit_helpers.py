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
    csv = "rendering_order,text_context,tag,parent_tag\n1,hello,p,div\n2,world,span,div\n"
    df = streamlit_app.parse_uploaded_file(csv.encode())
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["rendering_order", "text_context", "tag", "parent_tag"]


def test_parse_uploaded_file_none_returns_empty():
    df = streamlit_app.parse_uploaded_file(None)
    assert df.empty


def test_demo_extract_events_uses_default_source():
    class MockDemoModel:
        def predict(self, X):
            return [1 for _ in range(len(X))]

    df = pd.DataFrame({
        "rendering_order": [1],
        "text_context": ["open house saturday 10am"],
        "tag": ["p"],
        "parent_tag": ["div"],
    })

    events = streamlit_app.demo_extract_events(df, MockDemoModel())
    assert events == [{"source": "uploaded_file", "node_range": "0-0"}]
