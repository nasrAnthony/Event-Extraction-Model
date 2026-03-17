"""
synth_data.py

Synthetic page generator to fix the +1 BIO boundary overshoot.

The core problem:
  - 85 of 165 event starts (51%) are Date/DateTime nodes in real data
  - Model learns that high Prob_B correlates with date-looking text
  - On Name-first pages, the model correctly identifies the event region
    but assigns peak B-probability to the date node (+1 offset)

This module generates synthetic pages where:
  1. Name nodes are always the B boundary (fixes Date-first bias)
  2. Field order within events is shuffled (prevents layout memorization)
  3. Noisy O-nodes are injected between events (prevents gap memorization)
  4. Multiple layout styles are simulated (table, card, list)

Usage:
    from synth_data import generate_synthetic_df, augment_with_synthetics
    synth_df = generate_synthetic_df(n_pages=60, seed=42)
    full_df = augment_with_synthetics(real_df, n_synth_pages=60)
"""

import random
import numpy as np
import pandas as pd
from copy import deepcopy


# ── Field pools ──────────────────────────────────────────────────────────────

VENUE_NAMES = [
    "riverside conference center", "oak park gymnasium", "central high school fieldhouse",
    "downtown civic arena", "lakeside sports complex", "westview community center",
    "northgate university ballroom", "sunrise athletic complex", "valley expo hall",
    "hillcrest academy gymnasium", "peninsula conference center", "harbor view arena",
    "midtown athletic center", "eastside cultural center", "forest park fieldhouse",
    "union station event hall", "crown plaza ballroom", "summit center gymnasium",
    "lakeview high school", "shoreline community college gymnasium",
    "cedar ridge recreation center", "blue hills sports dome", "college of the pines gymnasium",
]

CITY_STATES = [
    "portland, or", "seattle, wa", "boise, id", "denver, co", "phoenix, az",
    "salt lake city, ut", "spokane, wa", "tucson, az", "albuquerque, nm",
    "sacramento, ca", "fresno, ca", "reno, nv", "las vegas, nv", "helena, mt",
    "missoula, mt", "eugene, or", "tacoma, wa", "bellevue, wa", "olympia, wa",
    "tempe, az", "mesa, az", "provo, ut", "ogden, ut", "fort collins, co",
    "boulder, co", "colorado springs, co", "cheyenne, wy", "billings, mt",
    "boston, ma", "cambridge, ma", "worcester, ma", "springfield, ma",
    "providence, ri", "hartford, ct", "new haven, ct", "bridgeport, ct",
    "manchester, nh", "nashua, nh", "concord, nh", "portland, me",
    "burlington, vt", "albany, ny", "rochester, ny", "buffalo, ny",
    "richmond, va", "norfolk, va", "raleigh, nc", "charlotte, nc",
    "columbia, sc", "charleston, sc", "atlanta, ga", "savannah, ga",
    "jacksonville, fl", "orlando, fl", "tampa, fl", "miami, fl",
    "nashville, tn", "memphis, tn", "louisville, ky", "lexington, ky",
    "indianapolis, in", "columbus, oh", "cleveland, oh", "cincinnati, oh",
    "detroit, mi", "grand rapids, mi", "milwaukee, wi", "madison, wi",
    "minneapolis, mn", "st. paul, mn", "des moines, ia", "iowa city, ia",
    "omaha, ne", "lincoln, ne", "kansas city, mo", "st. louis, mo",
    "chicago, il", "springfield, il", "st. louis, mo",
]

DATE_STRINGS = [
    "monday, january 12th", "tuesday, january 20th", "wednesday, february 4th",
    "thursday, february 19th", "friday, march 6th", "saturday, march 14th",
    "sunday, march 22nd", "monday, april 7th", "tuesday, april 15th",
    "wednesday, april 23rd", "thursday, may 1st", "friday, may 9th",
    "saturday, may 17th", "sunday, may 25th", "monday, june 2nd",
    "tuesday, june 10th", "wednesday, june 18th", "thursday, june 26th",
    "friday, july 4th", "saturday, july 12th", "sunday, july 20th",
    "monday, july 28th", "tuesday, august 5th", "wednesday, august 13th",
    "thursday, august 21st", "friday, august 29th", "saturday, september 6th",
    "sunday, september 14th", "monday, september 22nd", "tuesday, september 30th",
    "wednesday, october 8th", "thursday, october 16th", "friday, october 24th",
    "saturday, november 1st", "sunday, november 9th", "monday, november 17th",
    "tuesday, november 25th", "wednesday, december 3rd", "thursday, december 11th",
    "friday, december 19th",
    "jan 12, 2026", "feb 3, 2026", "mar 15, 2026", "apr 7, 2026",
    "may 22, 2026", "jun 5, 2026", "jul 18, 2026", "aug 9, 2026",
    "sep 23, 2026", "oct 14, 2026", "nov 6, 2026", "dec 1, 2026",
    "1/12/26", "2/3/26", "3/15/26", "4/7/26", "5/22/26",
]

TIME_STRINGS = [
    "9:00am", "9:30am", "10:00am", "10:30am", "11:00am", "11:30am",
    "12:00pm", "12:30pm", "1:00pm", "1:30pm", "2:00pm", "2:30pm",
    "3:00pm", "3:30pm", "4:00pm", "4:30pm", "5:00pm", "5:30pm",
    "6:00pm", "6:30pm", "7:00pm", "7:30pm", "8:00pm",
    "9am", "10am", "11am", "noon", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm",
]

END_TIME_STRINGS = [
    "11:00am", "11:30am", "12:00pm", "12:30pm", "1:00pm", "2:00pm",
    "3:00pm", "4:00pm", "5:00pm", "6:00pm", "7:00pm", "8:00pm", "9:00pm",
    "11am", "noon", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm",
]

INSTITUTION_NAMES = [
    "lincoln high school", "washington university", "riverside college",
    "oak valley high school", "central state university", "lakewood academy",
    "northview college", "sunset high school", "heritage university",
    "meadowbrook high school", "coastal community college", "ridgemont high school",
    "summit university", "pinecrest high school", "bayside college",
    "westfield high school", "springdale university", "clearwater academy",
    "greenhill high school", "hillside college", "fairview high school",
    "harborview university", "stonegate high school", "millbrook college",
    "brookside high school", "lakeview university", "cedarwood high school",
    "evergreen college", "mapleton high school", "silverlake university",
    "willowbrook high school", "mountain view college", "creekside academy",
    "bluewater high school", "ashwood university", "elmwood high school",
    "riverdale college", "highpoint academy", "brookfield high school",
    "northshore university", "fernwood high school", "westbrook college",
    "clearview high school", "ridgewood university", "stonehill academy",
    "harborside high school", "valley view college", "crestwood academy",
    "lakeside high school", "greenfield university",
]

PRICE_STRINGS = [
    "$50", "$75", "$100", "$125", "$150", "$175", "$200", "$225", "$250",
    "$300", "$350", "$400", "$500", "$50.00", "$75.00", "$100.00",
    "free", "no charge", "$0",
]

DESCRIPTION_STRINGS = [
    "join us for this exciting college fair event",
    "open to all prospective students and their families",
    "meet representatives from over 50 colleges and universities",
    "registration is required for all attendees",
    "open to the public",
    "free admission for all students",
    "college and university representatives will be available",
    "learn about admissions requirements and financial aid",
    "bring your transcripts and test scores",
    "dress business casual",
    "parking available on site",
    "light refreshments provided",
    "this event is open to all high school students",
    "pre-registration is encouraged but walk-ins welcome",
]

# O-node noise pools (realistic navigational/boilerplate content)
NOISE_TEXT_POOL = [
    "home", "about", "contact", "search", "menu", "login", "register",
    "events", "calendar", "news", "resources", "members", "join",
    "back to top", "view all events", "learn more", "read more",
    "skip to main content", "navigation", "footer", "header",
    "© 2026 all rights reserved", "privacy policy", "terms of use",
    "follow us", "facebook", "instagram", "twitter", "linkedin",
    "newsletter signup", "donate", "volunteer", "sponsors",
    "upcoming events", "past events", "featured events",
    "filter by date", "filter by location", "sort by",
    "page 1 of 3", "showing 1-10 of 30", "load more",
    "print page", "share this page", "bookmark",
    "college fair information", "event details", "venue information",
    "registration information", "exhibitor guidelines",
    "for more information contact", "questions? email us",
    "this event has passed", "registration closed",
    "check back for updates", "stay tuned",
    "powered by", "site map", "accessibility",
]

# ── Column spec (must match full_data.csv) ────────────────────────────────────

# Columns that need real values for the model
BOOL_COLS = [
    "has_link", "link_is_absolute", "parent_has_link", "is_leaf",
    "has_class", "has_id", "attr_has_word_name", "attr_has_word_date",
    "attr_has_word_time", "attr_has_word_location", "attr_has_word_link",
    "text_has_word_name", "text_has_word_date", "text_word_time",
    "text_word_description", "text_word_location", "text_word_am", "text_word_pm",
    "contains_date", "contains_time", "starts_with_digit", "ends_with_digit",
]

NUM_COLS = [
    "depth", "sibling_index", "children_count",
    "same_tag_sibling_count", "same_text_sibling_count",
    "text_length", "word_count", "letter_ratio", "digit_ratio", "whitespace_ratio",
    "attribute_count",
]


def _bool_signals_for_label(label: str, text: str) -> dict:
    """Generate realistic boolean feature values based on label type and text."""
    text_lower = text.lower()
    d = {col: 0 for col in BOOL_COLS}
    d["is_leaf"] = 1

    if label in ("Name", "NameLink"):
        d["attr_has_word_name"] = 1
        d["text_has_word_name"] = int("name" in text_lower)
        d["has_class"] = 1
        d["has_link"] = int(label == "NameLink")
        d["link_is_absolute"] = int(label == "NameLink")

    elif label in ("Date", "DateTime"):
        d["attr_has_word_date"] = 1
        d["text_has_word_date"] = int(any(w in text_lower for w in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec","monday","tuesday","wednesday","thursday","friday","saturday","sunday"]))
        d["contains_date"] = 1
        d["starts_with_digit"] = int(text_lower[0].isdigit() if text_lower else 0)
        d["has_class"] = 1

    elif label in ("Time", "StartTime", "EndTime"):
        d["attr_has_word_time"] = 1
        d["contains_time"] = 1
        d["text_word_time"] = int("time" in text_lower)
        d["text_word_am"] = int("am" in text_lower)
        d["text_word_pm"] = int("pm" in text_lower)
        d["ends_with_digit"] = 0
        d["has_class"] = 1

    elif label == "Location":
        d["attr_has_word_location"] = 1
        d["text_word_location"] = int("location" in text_lower)
        d["has_class"] = 1

    elif label == "Price":
        d["starts_with_digit"] = int(text_lower[0].isdigit() if text_lower else 0)
        d["digit_ratio"] = 0.3
        d["has_class"] = 1

    elif label == "Description":
        d["text_word_description"] = int("description" in text_lower)
        d["word_count"] = len(text.split())
        d["has_class"] = 1

    return d


def _num_features_for_label(label: str, text: str, sibling_idx: int, depth: int) -> dict:
    """Generate realistic numeric feature values."""
    text_len = len(text)
    letters = sum(c.isalpha() or c == " " for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = text.count(" ")
    d = {
        "depth": depth,
        "sibling_index": sibling_idx,
        "children_count": 0,
        "same_tag_sibling_count": random.randint(2, 15),
        "same_text_sibling_count": 0,
        "text_length": text_len,
        "word_count": len(text.split()),
        "letter_ratio": letters / max(text_len, 1),
        "digit_ratio": digits / max(text_len, 1),
        "whitespace_ratio": spaces / max(text_len, 1),
        "attribute_count": random.randint(1, 4),
    }
    return d


def _make_node(text: str, label: str, bio: int, event_id,
               source: str, rendering_order: int,
               sibling_idx: int = 0, depth: int = 5) -> dict:
    """Build a single node row matching the full_data.csv schema."""
    bools = _bool_signals_for_label(label, text)
    nums = _num_features_for_label(label, text, sibling_idx, depth)

    row = {
        "rendering_order": rendering_order,
        "tag": random.choice(["Div", "Span", "P", "Li", "Td", "A"]),
        "attributes": f"class:event-field-{label.lower()}",
        "text_context": text,
        "parent_index": rendering_order - 1,
        "parent_tag": random.choice(["Div", "Ul", "Table", "Section"]),
        "link": "",
        "label": label,
        "event_id": event_id,
        "source": source,
        "bio": bio,
        **bools,
        **nums,
    }
    return row


def _make_noise_node(source: str, rendering_order: int, n_event_siblings: int = 10) -> dict:
    """Build an O-node (non-event noise node)."""
    text = random.choice(NOISE_TEXT_POOL)
    bools = {col: 0 for col in BOOL_COLS}
    bools["is_leaf"] = 1
    bools["has_class"] = random.randint(0, 1)

    text_len = len(text)
    nums = {
        "depth": random.randint(3, 8),
        "sibling_index": random.randint(0, 5),
        "children_count": 0,
        "same_tag_sibling_count": random.randint(0, 8),
        "same_text_sibling_count": 0,
        "text_length": text_len,
        "word_count": len(text.split()),
        "letter_ratio": sum(c.isalpha() for c in text) / max(text_len, 1),
        "digit_ratio": sum(c.isdigit() for c in text) / max(text_len, 1),
        "whitespace_ratio": text.count(" ") / max(text_len, 1),
        "attribute_count": random.randint(0, 3),
    }

    return {
        "rendering_order": rendering_order,
        "tag": random.choice(["Div", "Span", "P", "Li", "A", "Nav", "Header"]),
        "attributes": "class:nav-item",
        "text_context": text,
        "parent_index": rendering_order - 1,
        "parent_tag": random.choice(["Nav", "Header", "Footer", "Div"]),
        "link": "",
        "label": "Other",
        "event_id": None,
        "source": source,
        "bio": 0,
        **bools,
        **nums,
    }


# ── Layout Styles ─────────────────────────────────────────────────────────────

# Each layout style defines:
#   - field_order: sequence of labels per event
#   - optional_fields: fields to randomly include/exclude
#   - name_variants: pool to draw from for the Name field

LAYOUT_STYLES = {
    "name_first_full": {
        # Classic: Name → Date → Time → Location → optional extras
        "required": ["Name", "Date", "Time", "Location"],
        "optional": ["Price", "Description"],
        "name_first": True,
    },
    "name_first_sparse": {
        # Sparse: Name → Date → Location only
        "required": ["Name", "Date", "Location"],
        "optional": [],
        "name_first": True,
    },
    "name_first_with_endtime": {
        # Name → Date → StartTime → EndTime → Location
        "required": ["Name", "Date", "StartTime", "EndTime", "Location"],
        "optional": ["Description"],
        "name_first": True,
    },
    "name_link_first": {
        # NameLink → DateTime → Location
        "required": ["NameLink", "DateTime", "Location"],
        "optional": ["Price"],
        "name_first": True,
    },
    "date_first_full": {
        # Date-first: mirrors real data (don't remove entirely — model needs to handle it)
        "required": ["Date", "Name", "Location"],
        "optional": ["Time", "Description"],
        "name_first": False,
    },
    "date_first_sparse": {
        "required": ["DateTime", "Name", "Location"],
        "optional": [],
        "name_first": False,
    },
    "shuffled": {
        # Fully randomized order — forces model to rely on content not position
        "required": ["Name", "Date", "Location", "Time"],
        "optional": ["Price", "Description"],
        "name_first": None,  # will be shuffled
    },
}


def _build_event_nodes(event_id: int, layout_style: str,
                       source: str, start_ro: int) -> list[dict]:
    """
    Build all nodes for a single synthetic event.
    Returns list of row dicts, with BIO labels assigned.
    """
    style = LAYOUT_STYLES[layout_style]
    fields = list(style["required"])

    # Add optional fields with 50% probability each
    for opt in style["optional"]:
        if random.random() < 0.5:
            fields.append(opt)

    # Determine field order
    if style["name_first"] is True:
        pass  # keep as-is (name is already first in required)
    elif style["name_first"] is False:
        pass  # keep as-is (date is first)
    else:
        # shuffle
        random.shuffle(fields)

    # Generate text for each field
    field_text = {}
    for f in fields:
        if f in ("Name", "NameLink"):
            field_text[f] = random.choice(INSTITUTION_NAMES)
        elif f in ("Date", "DateTime"):
            field_text[f] = random.choice(DATE_STRINGS)
        elif f in ("Time", "StartTime"):
            field_text[f] = random.choice(TIME_STRINGS)
        elif f == "EndTime":
            field_text[f] = random.choice(END_TIME_STRINGS)
        elif f == "Location":
            if random.random() < 0.6:
                field_text[f] = random.choice(VENUE_NAMES)
            else:
                field_text[f] = random.choice(CITY_STATES)
        elif f == "Price":
            field_text[f] = random.choice(PRICE_STRINGS)
        elif f == "Description":
            field_text[f] = random.choice(DESCRIPTION_STRINGS)
        else:
            field_text[f] = f.lower()

    # Assign BIO: first field = B, rest = I
    nodes = []
    for i, f in enumerate(fields):
        bio = 1 if i == 0 else 2  # B=1 for first, I=2 for rest
        node = _make_node(
            text=field_text[f],
            label=f,
            bio=bio,
            event_id=event_id,
            source=source,
            rendering_order=start_ro + i,
            sibling_idx=i,
            depth=random.randint(4, 7),
        )
        nodes.append(node)

    return nodes


def generate_synthetic_page(source_name: str,
                             n_events: int = None,
                             layout_style: str = None,
                             min_noise_before: int = 3,
                             max_noise_before: int = 15,
                             min_noise_between: int = 1,
                             max_noise_between: int = 6,
                             min_noise_after: int = 2,
                             max_noise_after: int = 8,
                             seed: int = None) -> pd.DataFrame:
    """
    Generate one synthetic page as a DataFrame.

    Args:
        source_name:         Unique identifier for this page
        n_events:            Number of events on the page (default: 3-10)
        layout_style:        One of LAYOUT_STYLES keys, or None for random
        min/max_noise_*:     Control how many O-nodes surround events
        seed:                Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    if n_events is None:
        n_events = random.randint(3, 10)

    rows = []
    ro = 0  # rendering_order counter

    # Header noise
    n_header = random.randint(min_noise_before, max_noise_before)
    for _ in range(n_header):
        rows.append(_make_noise_node(source_name, ro))
        ro += 1

    for ev_id in range(n_events):
        # Pick layout
        if layout_style is None:
            # Weighted sampling: Name-first gets 65% of synthetic pages
            weights = {
                "name_first_full": 20,
                "name_first_sparse": 15,
                "name_first_with_endtime": 15,
                "name_link_first": 15,
                "date_first_full": 10,
                "date_first_sparse": 10,
                "shuffled": 15,
            }
            style_key = random.choices(
                list(weights.keys()),
                weights=list(weights.values()),
            )[0]
        else:
            style_key = layout_style

        event_nodes = _build_event_nodes(ev_id, style_key, source_name, ro)
        rows.extend(event_nodes)
        ro += len(event_nodes)

        # Inter-event noise (except after last event)
        if ev_id < n_events - 1:
            n_between = random.randint(min_noise_between, max_noise_between)
            for _ in range(n_between):
                rows.append(_make_noise_node(source_name, ro))
                ro += 1

    # Footer noise
    n_footer = random.randint(min_noise_after, max_noise_after)
    for _ in range(n_footer):
        rows.append(_make_noise_node(source_name, ro))
        ro += 1

    df = pd.DataFrame(rows)
    return df


def generate_synthetic_df(n_pages: int = 60,
                          seed: int = 42,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Generate a full synthetic dataset of n_pages pages.

    Strategy:
      - 40% pure Name-first pages (all events start with Name/NameLink)
      - 20% pure Date-first pages (to keep model honest)
      - 40% mixed/shuffled pages (field order varies per event)

    Args:
        n_pages:  Number of synthetic pages to generate
        seed:     Master random seed
        verbose:  Print summary statistics
    """
    random.seed(seed)
    np.random.seed(seed)

    all_dfs = []
    n_name_first = int(n_pages * 0.40)
    n_date_first = int(n_pages * 0.20)
    n_mixed = n_pages - n_name_first - n_date_first

    # Name-first pages
    name_first_styles = ["name_first_full", "name_first_sparse",
                         "name_first_with_endtime", "name_link_first"]
    for i in range(n_name_first):
        style = random.choice(name_first_styles)
        src = f"synth_namefirst_{i:03d}"
        df = generate_synthetic_page(src, layout_style=style,
                                     seed=seed + i)
        all_dfs.append(df)

    # Date-first pages
    date_first_styles = ["date_first_full", "date_first_sparse"]
    for i in range(n_date_first):
        style = random.choice(date_first_styles)
        src = f"synth_datefirst_{i:03d}"
        df = generate_synthetic_page(src, layout_style=style,
                                     seed=seed + n_name_first + i)
        all_dfs.append(df)

    # Mixed/shuffled pages
    for i in range(n_mixed):
        src = f"synth_mixed_{i:03d}"
        df = generate_synthetic_page(src, layout_style=None,
                                     seed=seed + n_name_first + n_date_first + i)
        all_dfs.append(df)

    synth_df = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        bio_counts = synth_df["bio"].value_counts().sort_index()
        print(f"Synthetic dataset: {len(synth_df)} nodes across {n_pages} pages")
        print(f"  BIO distribution: O={bio_counts.get(0,0)}, B={bio_counts.get(1,0)}, I={bio_counts.get(2,0)}")
        b_labels = synth_df[synth_df["bio"] == 1]["label"].value_counts()
        print(f"  B-node label distribution (this is what matters):")
        print(f"  {b_labels.to_dict()}")
        print(f"  Pages: Name-first={n_name_first}, Date-first={n_date_first}, Mixed={n_mixed}")

    return synth_df


def augment_with_synthetics(real_df: pd.DataFrame,
                             n_synth_pages: int = 60,
                             seed: int = 42,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Combine real data with synthetic data for training.

    The synthetic pages are added to training folds only — never to
    validation/test folds. Use the 'source' column to split: all sources
    starting with 'synth_' are synthetic.

    Args:
        real_df:        Your existing full_data.csv DataFrame (with bio col)
        n_synth_pages:  How many synthetic pages to generate
        seed:           Random seed
        verbose:        Print summary

    Returns:
        Combined DataFrame. Real pages keep their original sources.
        Synthetic pages have sources like 'synth_namefirst_000'.
    """
    synth_df = generate_synthetic_df(n_synth_pages, seed=seed, verbose=verbose)

    # Align columns — synthetic df may be missing some real df columns
    for col in real_df.columns:
        if col not in synth_df.columns:
            synth_df[col] = 0 if real_df[col].dtype in [np.float64, np.int64] else ""

    combined = pd.concat([real_df, synth_df[real_df.columns]], ignore_index=True)

    if verbose:
        real_bio = real_df["bio"].value_counts().sort_index()
        synth_bio = synth_df["bio"].value_counts().sort_index()
        print(f"\nCombined dataset: {len(combined)} nodes")
        print(f"  Real B-nodes: {real_bio.get(1, 0)}, Synth B-nodes: {synth_bio.get(1, 0)}")
        print(f"  Total B-nodes: {real_bio.get(1, 0) + synth_bio.get(1, 0)}")
        print(f"  B-ratio improved: {real_bio.get(1,0)/max(len(real_df),1):.3f} → "
              f"{(real_bio.get(1,0)+synth_bio.get(1,0))/max(len(combined),1):.3f}")

    return combined


# ── Augmentation: shuffle field order within real events ──────────────────────

def shuffle_event_fields(df: pd.DataFrame,
                          p_shuffle: float = 0.5,
                          seed: int = 42) -> pd.DataFrame:
    """
    Data augmentation: for each real page, create a copy where the
    field order within each event is randomly permuted.

    This prevents the model from memorizing "Name is always at position 0
    within an event" and instead forces it to rely on content signals.

    Only use augmented copies in training, never in validation.

    Args:
        df:          DataFrame with bio/event_id/source columns
        p_shuffle:   Probability of shuffling each event's fields
        seed:        Random seed

    Returns:
        New DataFrame of augmented pages (source suffix: '_shuffled')
    """
    random.seed(seed)
    augmented_pages = []

    for src, page_df in df.groupby("source"):
        new_src = f"{src}_shuffled"
        page_copy = page_df.copy()
        page_copy["source"] = new_src

        for ev_id, ev_group in page_copy.groupby("event_id"):
            if pd.isna(ev_id):
                continue
            if random.random() > p_shuffle:
                continue

            idxs = ev_group.index.tolist()
            if len(idxs) <= 1:
                continue

            # Shuffle all but preserve first = B, rest = I labels
            shuffled_idxs = idxs[:]
            random.shuffle(shuffled_idxs)

            # Reassign bio: new first = B, rest = I
            for new_pos, orig_idx in enumerate(shuffled_idxs):
                page_copy.loc[orig_idx, "bio"] = 1 if new_pos == 0 else 2

            # Swap the text and label content to match shuffled positions
            texts = page_copy.loc[idxs, "text_context"].tolist()
            labels = page_copy.loc[idxs, "label"].tolist()
            shuffled_texts = [texts[idxs.index(i)] for i in shuffled_idxs]
            shuffled_labels = [labels[idxs.index(i)] for i in shuffled_idxs]

            for pos, orig_idx in enumerate(idxs):
                page_copy.loc[orig_idx, "text_context"] = shuffled_texts[pos]
                page_copy.loc[orig_idx, "label"] = shuffled_labels[pos]
                page_copy.loc[orig_idx, "bio"] = 1 if pos == 0 else 2

        augmented_pages.append(page_copy)

    if not augmented_pages:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(augmented_pages, ignore_index=True)


if __name__ == "__main__":
    # Quick smoke test
    print("=== Synthetic data generator smoke test ===\n")

    # Single page
    page = generate_synthetic_page("test_page_001", n_events=4, seed=42)
    print(f"Single page: {len(page)} nodes")
    print(page[["bio", "label", "text_context"]].to_string())
    print()

    # Full dataset
    synth = generate_synthetic_df(n_pages=30, seed=42)
    print()

    # Check B-label distribution is now Name-heavy
    b_nodes = synth[synth["bio"] == 1]
    print("B-node label distribution in synthetic data:")
    print(b_nodes["label"].value_counts())
    print()
    print("Compare to real data (from full_data.csv analysis):")
    print("  DateTime: 47, Date: 38, NameLink: 30, Name: 29 (Date-heavy)")
    print("  Goal: flip this so Name/NameLink dominate")
