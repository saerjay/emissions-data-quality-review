"""
End-to-end tests of the app: runs the real Streamlit script headlessly and checks what a
non-technical visitor sees and can do.
"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import app

pytestmark = pytest.mark.skipif(not os.path.exists(app.DEFAULT_DATA),
                                reason="run `python data_prep.py` first")

TABS = ["Dashboard", "Emissions Map", "Review Records", "Test the Checker", "Data Coverage", "Methods", "About This Project"]


@pytest.fixture
def ui():
    return AppTest.from_file(os.path.join(os.path.dirname(app.__file__), "app.py"), default_timeout=300).run()


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_page_opens_on_a_plain_language_dashboard(ui):
    assert not ui.exception
    assert [t.label for t in ui.tabs] == TABS
    for label in ["Records checked", "Ready to use", "Need a check", "Must fix"]:
        assert metric(ui, label)
    assert metric(ui, "Must fix") == "0"
    assert any("Key takeaways" in m.value for m in ui.markdown)


def test_readiness_verdict_is_an_accessible_status_card(ui):
    cards = [m.value for m in ui.markdown if 'role="status"' in m.value]
    assert cards, "verdict card missing"
    assert any(("Ready after review" in c) or ("Not ready" in c) or ("Ready to use" in c) for c in cards)


def test_dashboard_has_the_redesigned_sections(ui):
    text = " ".join(m.value for m in ui.markdown)
    for heading in ["Where the records stand", "Where to look first", "Flagged records by year",
                    "How reliable is the checker", "Coverage"]:
        assert heading in text, heading


def test_flagged_by_year_shows_shares_and_an_automated_insight(ui):
    text = " ".join(m.value for m in ui.markdown)
    assert "Share of each year's records flagged" in text
    assert "stands out" in text          # the real data has 2003 above the typical year


def test_part_to_whole_legend_lists_every_group_with_counts(ui):
    legend = next(m.value for m in ui.markdown if 'class="legend"' in m.value)
    for group in ["Ready", "Low", "Medium", "High"]:
        assert group in legend
    assert "4,836" in legend or "%" in legend


def test_there_is_no_sidebar_or_practice_mode(ui):
    # No upload, no practice mode, no adjustable settings: the app always shows the tested defaults
    assert not ui.sidebar.children
    source = open(os.path.join(os.path.dirname(app.__file__), 'app.py'), encoding='utf-8').read().lower()
    for removed in ['try it yourself', 'practice mode', 'st.sidebar', 'file_uploader', 'advanced settings']:
        assert removed not in source, removed


def test_record_picker_and_filters_work(ui):
    picker = next(s for s in ui.selectbox if s.label.startswith("Choose a record"))
    picker.set_value(picker.value).run()
    priority = next(m for m in ui.multiselect if m.label.startswith("Priority"))
    priority.set_value(["Low"]).run()
    assert not ui.exception


def test_reliability_note_states_which_settings_were_measured(ui):
    assert any("settings this app uses" in e.value for e in ui.info)


def test_methods_tab_explains_the_study_design(ui):
    text = " ".join(m.value for m in ui.markdown)
    assert "Study design" in text
    assert "What was compared" in text
    assert "Model strictness" in text


def test_map_tab_renders_and_responds_to_choices(ui):
    assert ui.get("plotly_chart"), "map not rendered"
    measure = next(s for s in ui.selectbox if s.label == "Emission type")
    measure.set_value('methane').run()
    country = next(s for s in ui.selectbox if s.label == "Country")
    country.set_value('India').run()
    assert not ui.exception
    assert any('India' in m.value for m in ui.markdown)
