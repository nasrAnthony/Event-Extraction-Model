Production-facing tests live here.

Run the suite with:

```bash
python -m pytest tests -q
```

Manual Streamlit upload samples live in `tests/fixtures/streamlit/`:

- `sample_for_ui_min.csv`: smallest demo-mode smoke test
- `sample_for_ui_pretrained_hawaii_window.csv`: compact pretrained-model sample, expected to extract 3 events
- `sample_for_ui_pretrained_hawaii_page.csv`: fuller pretrained-model sample, expected to extract 7 events
