import pandas as pd
from app import inference_service as svc


def test_validate_input_success_and_missing():
    df = pd.DataFrame({
        'source': ['s1'],
        'rendering_order': [1],
        'text_context': ['hello'],
        'tag': ['t'],
        'parent_tag': ['pt']
    })
    ok, msg = svc.validate_input(df)
    assert ok is True
    # missing column
    df2 = df.drop(columns=['tag'])
    ok2, msg2 = svc.validate_input(df2)
    assert ok2 is False
    assert 'Missing required columns' in msg2


def test_run_inference_with_mock_model():
    # mock model with predict signature
    class MockModel:
        def predict(self, X):
            # return 1 for first row, 0 for others
            return [1 if i == 0 else 0 for i in range(len(X))]

    df = pd.DataFrame({
        'text_context': ['a', 'b', 'c'],
        'tag': ['t1', 't2', 't3'],
        'parent_tag': ['p1', 'p2', 'p3'],
        'source': ['s1', 's1', 's1'],
        'rendering_order': [1, 2, 3]
    })

    out = svc.run_inference(df, MockModel())
    assert 'is_event_pred' in out.columns
    assert int(out['is_event_pred'].sum()) == 1


def test_format_events_for_ui():
    events = [
        {'source': 's1', 'node_range': '0-0', 'name': 'E1'},
        {'source': 's1', 'node_range': '1-1', 'name': 'E2'},
        {'source': 's2', 'node_range': '0-0', 'name': 'F1'}
    ]
    df = svc.format_events_for_ui(events)
    assert 'source' in df.columns
    assert 'event_index' in df.columns
    # event_index per source should start at 0
    assert df[df['source']=='s1']['event_index'].tolist() == [0, 1]
