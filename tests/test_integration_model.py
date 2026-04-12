import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tempfile import TemporaryDirectory

from app.inference_service import run_inference


def test_integration_mock_pipeline_end_to_end():
    # Train a tiny pipeline on two example sentences
    texts = ["event on monday at 5pm", "this text has no event"]
    labels = [1, 0]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit(texts, labels)

    # Save and reload the pipeline to emulate a committed joblib artifact
    with TemporaryDirectory() as td:
        path = f"{td}/mock_pipeline.joblib"
        joblib.dump(pipe, path)
        loaded = joblib.load(path)

        # Prepare a DataFrame that matches expected input for run_inference
        df = pd.DataFrame(
            {
                "source": ["s1", "s1"],
                "rendering_order": [1, 2],
                "text_context": texts,
                "tag": ["p", "p"],
                "parent_tag": ["root", "root"],
            }
        )

        out = run_inference(df.copy(), loaded)

        # Basic assertions: same number of rows, has is_event_pred column, values are 0/1
        assert out.shape[0] == df.shape[0]
        assert "is_event_pred" in out.columns
        assert set(int(x) for x in out["is_event_pred"].astype(int).unique()).issubset({0, 1})
