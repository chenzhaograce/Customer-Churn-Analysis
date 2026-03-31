#!/usr/bin/env python3
"""Quick dependency check before `streamlit run app.py`. Run from repo root."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "telco_churn.csv"

MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "openai",
    "sqlalchemy",
    "openpyxl",
]


def main() -> int:
    print("Customer Churn Analysis — environment check")
    print(f"Project root: {ROOT}")
    ok = True

    if not DATA_CSV.is_file():
        print(f"  [WARN] Missing sample data: {DATA_CSV}")
        print("         Clone the full repo or add data/telco_churn.csv.")
    else:
        print(f"  [OK]   Sample CSV exists ({DATA_CSV.stat().st_size // 1024} KB)")

    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"  [OK]   {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            ok = False

    try:
        from google import genai  # noqa: F401
        print("  [OK]   google.genai (Gemini)")
    except Exception as e:
        print(f"  [FAIL] google.genai: {e}")
        ok = False

    print()
    if ok:
        print("Next step (from this directory):")
        print("  streamlit run app.py")
        print("Do not use: python app.py  (that does not start the web server.)")
        return 0

    print("Fix failed imports, then:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
