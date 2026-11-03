"""Headless dashboard tests (streamlit.testing.AppTest): every page renders; role gating works."""

import glob
from pathlib import Path

import pytest

from retailpulse import config

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

if not (config.DATA_PROC / "churn_scores.parquet").exists():
    pytest.skip("run the pipeline first", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(glob.glob(str(ROOT / "app" / "pages" / "*.py")))


@pytest.fixture(autouse=True)
def isolated_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOGS", tmp_path)  # never write test events to the real audit log


def run(page: str, **state) -> AppTest:
    at = AppTest.from_file(page, default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.mark.parametrize("page", PAGES, ids=[Path(p).stem for p in PAGES])
def test_every_page_renders_without_exceptions(page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_viewer_is_locked_out_of_exports_and_admin_sections():
    at = run(str(ROOT / "app" / "pages" / "7_Reports.py"))
    assert any("Analyst" in w.value for w in at.markdown), "reports must be gated for Viewer"
    assert not at.button, "no generate button for Viewer"


def test_analyst_unlocks_reports_but_not_audit_log():
    at = run(str(ROOT / "app" / "pages" / "7_Reports.py"), role="Analyst", user="analyst")
    assert at.button, "Analyst sees the generate-PDF button"
    mlops = run(str(ROOT / "app" / "pages" / "5_MLOps_and_Drift.py"), role="Analyst", user="analyst")
    assert any("Admin" in m.value and "access required" in m.value for m in mlops.markdown)


def test_admin_sees_audit_section_and_wrong_pin_is_rejected():
    mlops = run(str(ROOT / "app" / "pages" / "5_MLOps_and_Drift.py"), role="Admin", user="admin")
    assert not mlops.exception
    assert not any("Admin</b> access required" in m.value for m in mlops.markdown)

    at = run(str(ROOT / "app" / "pages" / "4_Inventory.py"))
    at.sidebar.selectbox(key="role_sel").select("Admin").run()
    at.sidebar.text_input(key="pin_in").set_value("definitely-wrong")
    at.sidebar.button(key="unlock").click().run()
    assert at.session_state["role"] == "Viewer"
    assert at.sidebar.error, "wrong passcode must show an error"
    at.sidebar.text_input(key="pin_in").set_value("admin")
    at.sidebar.button(key="unlock").click().run()
    assert at.session_state["role"] == "Admin"
    events = [l for l in (config.LOGS / "audit.log").read_text().splitlines()]
    assert any("login_failed" in l for l in events) and any('"login"' in l for l in events)


def test_inventory_what_if_changes_the_plan():
    page = str(ROOT / "app" / "pages" / "4_Inventory.py")
    base = run(page)
    plan0 = [m.value for m in base.markdown if "Purchase plan" in m.value or "PURCHASE PLAN" in m.value.upper()]
    at = run(page)
    at.slider[2].set_value(1.5).run()  # demand multiplier
    plan1 = [m.value for m in at.markdown if "Purchase plan" in m.value or "PURCHASE PLAN" in m.value.upper()]
    assert not at.exception and plan0 and plan1 and plan0 != plan1
