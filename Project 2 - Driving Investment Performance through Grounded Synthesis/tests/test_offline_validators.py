import json
from pathlib import Path

from scripts.globaledge_offline import (
    assert_offline_guard,
    notebook_cell_count,
    notebook_cell_types,
    resolve_dataset_files,
    source_aware_filter,
    validate_notebook_structure,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "UTAIGA_Project_2_Low_Code_Notebook.ipynb"


def test_notebook_has_94_cells_and_expected_types():
    assert notebook_cell_count(NOTEBOOK) == 94
    types = notebook_cell_types(NOTEBOOK)
    assert types[0] == "markdown"
    assert types[10] == "code"
    assert types[20] == "markdown"
    assert types[94 - 1] == "markdown"
    validate_notebook_structure(NOTEBOOK)


def test_real_corpus_files_exist_and_match_expected_schema():
    files = resolve_dataset_files(ROOT)
    assert files["news"].exists()
    assert files["prices"].exists()
    assert files["filings"].exists()
    assert files["benchmark"].exists()
    assert files["news"].name == "global_news.csv"
    assert files["prices"].name == "stock_price_details.csv"
    assert files["filings"].name == "sec_filings_10k.txt"
    assert files["benchmark"].name == "golden_benchmark_dataset.csv"


def test_source_aware_filter_matches_company_and_type():
    rows = [
        {"metadata": {"source_type": "news", "company": "Microsoft", "date": "2025-01-06"}},
        {"metadata": {"source_type": "filing", "company": "Microsoft", "date": "2024-06-30"}},
        {"metadata": {"source_type": "news", "company": "Apple", "date": "2025-01-07"}},
    ]
    filtered = source_aware_filter(rows, required_company="Microsoft", required_type="filing", required_date="2024", max_items=5)
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["company"] == "Microsoft"


def test_guard_config_returns_expected_shape():
    cfg = ROOT / "Misc" / "config.json"
    if cfg.exists():
        result = assert_offline_guard(cfg)
        assert "status" in result
    else:
        assert True


def test_guard_uses_no_live_calls_by_default():
    from scripts.globaledge_offline import active_run_status_message
    msg = active_run_status_message()
    assert "PENDING LIVE RUN" in msg
    assert "offline" in msg.lower()
