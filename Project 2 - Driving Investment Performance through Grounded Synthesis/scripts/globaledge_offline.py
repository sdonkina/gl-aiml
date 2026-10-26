from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_FILES = {
    "news": "Financial_Dataset/global_news.csv",
    "prices": "Financial_Dataset/stock_price_details.csv",
    "filings": "Financial_Dataset/sec_filings_10k.txt",
    "benchmark": "Financial_Dataset/golden_benchmark_dataset.csv",
}


@dataclass
class EvidenceRecord:
    doc_id: str
    source_type: str
    source_file: str
    company: str = ""
    ticker: str = ""
    date: str = ""
    excerpt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_project_root(start: Path | str | None = None) -> Path:
    base = Path(start) if start else Path(__file__).resolve().parent.parent
    if not (base / "UTAIGA_Project_2_Low_Code_Notebook.ipynb").exists():
        # allow work tree path if project folder root differs
        for candidate in sorted(base.parents):
            if (candidate / "UTAIGA_Project_2_Low_Code_Notebook.ipynb").exists():
                return candidate
    return base


def resolve_dataset_files(project_root: Path | str | None = None) -> dict[str, Path]:
    root = detect_project_root(project_root)
    lookup = {}
    for key, rel in DEFAULT_DATASET_FILES.items():
        candidates = [
            root / rel,
            root / "Financial_Dataset" / Path(rel).name,
            root / "Financial_Dataset" / rel.replace("Financial_Dataset/", ""),
            root / "Financial_Dataset" / "Financial_Dataset" / Path(rel).name,
        ]
        p = next((c for c in candidates if c.exists()), candidates[0])
        lookup[key] = p
    return lookup


def load_json_file(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing JSON file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def notebook_cell_count(notebook_path: Path | str) -> int:
    nb = load_json_file(notebook_path)
    return len(nb.get("cells", []))


def notebook_cell_types(notebook_path: Path | str) -> list[str]:
    nb = load_json_file(notebook_path)
    return [cell.get("cell_type", "") for cell in nb.get("cells", [])]


def validate_notebook_structure(notebook_path: Path | str) -> dict[str, Any]:
    nb = load_json_file(notebook_path)
    cells = nb.get("cells", [])
    if len(cells) != 94:
        raise ValueError(f"Expected 94 cells, found {len(cells)}")
    types = [cell.get("cell_type", "") for cell in cells]
    if types[0] != "markdown" or types[10] != "code" or types[19] != "markdown":
        raise ValueError("Notebook did not preserve the required low-code section layout.")
    if types[-1] != "markdown":
        raise ValueError("Notebook conclusion cell is not markdown as required.")
    headings = []
    for cell in cells:
        source = "".join(cell.get("source", []))
        if source.startswith("#"):
            headings.append(source.strip().splitlines()[0])
    required_heading_starts = [
        "## Problem Statement",
        "### Business Context",
        "### Objective",
        "### Data Description",
        "# **Please read the instructions carefully before starting the project.**",
        "# Installing and Importing Necessary Libraries and Dependencies",
        "## Environment Setup and Project Initialization",
        "# Section 1: Baseline Retrieval-Augmented Generation (RAG) Pipeline",
        "# Section 2: DeepEval Prompt Optimization Layer",
        "# Section 3: RAG Configuration Tuning and Evaluation",
        "# Section 4: Final RAG Inference Pipeline with the Optimized Prompt and Best RAG Configuration",
        "# Conclusion",
    ]
    present = [h for h in headings if h in required_heading_starts]
    if len(present) < len(required_heading_starts):
        missing = [h for h in required_heading_starts if h not in present]
        raise ValueError(f"Required headings are missing: {missing}")
    return {"cell_count": len(cells), "types": types, "headings": headings[:12]}


def load_csv_schema(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    import pandas as pd
    df = pd.read_csv(p)
    return {"shape": df.shape, "columns": df.columns.tolist(), "sample": df.head(1).to_dict(orient="records")[0]}


def assert_offline_guard(config_path: Path | str) -> dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return {"status": "not_testable", "reason": "Config file missing"}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "status": "ok",
        "api_key_present": bool(data.get("OPENAI_API_KEY") and data["OPENAI_API_KEY"] != "your-openai-key"),
        "base_url_present": bool(data.get("OPENAI_API_BASE") or data.get("OPENAI_BASE_URL")),
    }


def source_aware_filter(
    rows: Iterable[dict[str, Any]],
    *,
    required_company: str | None = None,
    required_type: str | None = None,
    required_date: str | None = None,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata", {})
        source_type = metadata.get("source_type") or row.get("source_type")
        if required_type and source_type and source_type != required_type:
            continue
        if required_company:
            company = str(metadata.get("company") or row.get("company", "")).lower()
            target = required_company.lower()
            if company and target not in company:
                continue
        if required_date:
            date = str(metadata.get("date") or row.get("date", "")).lower()
            if date and required_date.lower() not in date:
                continue
        selected.append(row)
        if len(selected) >= max_items:
            break
    return selected


def make_citation_id(index: int) -> str:
    return f"Source {index}"


def active_run_status_message() -> str:
    return "[PENDING LIVE RUN] Historical dataset only; no direct model execution or paid API calls were run in this offline validation pass."

