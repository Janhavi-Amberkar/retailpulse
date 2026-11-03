import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrain", ROOT / "scripts" / "retrain.py")
retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain)


def test_gate_promotes_equal_or_better_and_rejects_regressions():
    assert retrain.challenger_ok(0.80, 0.80, 12.0, 12.0)  # identical -> promote
    assert retrain.challenger_ok(0.80, 0.85, 12.0, 10.0)  # better -> promote
    assert retrain.challenger_ok(0.80, 0.795, 12.0, 12.5)  # inside tolerance -> promote
    assert not retrain.challenger_ok(0.80, 0.70, 12.0, 12.0)  # AUC regression -> roll back
    assert not retrain.challenger_ok(0.80, 0.80, 12.0, 14.0)  # MAPE regression -> roll back


def test_snapshot_and_restore_roundtrip(tmp_path, monkeypatch):
    cfg = retrain.config
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "proc").mkdir()
    monkeypatch.setattr(cfg, "ROOT", tmp_path)
    monkeypatch.setattr(cfg, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(cfg, "MODELS", tmp_path / "models")
    monkeypatch.setattr(cfg, "DATA_PROC", tmp_path / "proc")
    (tmp_path / "reports" / "churn_metrics.json").write_text(json.dumps({"final": {"auc_roc": 0.8}}))
    (tmp_path / "models" / "churn_xgb.json").write_text("champion-model")
    snap = retrain.snapshot("t1")
    # challenger overwrites artefacts...
    (tmp_path / "reports" / "churn_metrics.json").write_text(json.dumps({"final": {"auc_roc": 0.5}}))
    (tmp_path / "models" / "churn_xgb.json").write_text("bad-challenger")
    retrain.restore(snap)  # ...and is rolled back
    assert json.loads((tmp_path / "reports" / "churn_metrics.json").read_text())["final"]["auc_roc"] == 0.8
    assert (tmp_path / "models" / "churn_xgb.json").read_text() == "champion-model"
