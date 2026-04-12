import os
import pandas as pd
from helpers.clean_data import clean
from helpers.concat import fuse


def test_clean_strips_and_dedups():
    df = pd.DataFrame({
        'tag': [' a', 'a', 'b '],
        'text_context': [' hello ', 'hello', 'world'],
    'parent_tag': [' p ', 'p', 'p'],
    'label': [' L ', 'L', 'L'],
    'link': ['', '', '']
    })
    out = clean(df)
    # duplicates removed and stripped
    assert out['tag'].str.startswith('a').any()
    assert out['text_context'].str.contains('hello').any()
    assert out['parent_tag'].str.contains('p').all()


def test_fuse_creates_full_csv(tmp_path):
    clean_dir = tmp_path / 'clean'
    out_dir = tmp_path / 'out'
    clean_dir.mkdir()
    out_dir.mkdir()

    df1 = pd.DataFrame({'text_context': ['a'], 'event_id': [None]})
    df2 = pd.DataFrame({'text_context': ['b'], 'event_id': ['e1']})
    (clean_dir / 'a.csv').write_text(df1.to_csv(index=False))
    (clean_dir / 'b.csv').write_text(df2.to_csv(index=False))

    fuse(str(clean_dir), str(out_dir), ['a.csv', 'b.csv'])

    combined = pd.read_csv(out_dir / 'full_data.csv')
    assert len(combined) == 2
    assert 'source' in combined.columns
