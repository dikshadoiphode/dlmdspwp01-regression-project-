# DLMDSPWP01 – Regression / Ideal Function Mapping Project

This project implements the IU DLMDSPWP01 assignment requirements:
- Load training/ideal/test CSVs
- Persist them into SQLite using SQLAlchemy
- Select 4 best-fit ideal functions by least squares (SSE)
- Map test points using the √2 deviation rule
- Visualize with Bokeh
- Provide unit tests

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m dlmdspwp01.main --train data/train.csv --ideal data/ideal.csv --test data/test.csv --db outputs/assignment.sqlite
pytest -q
```

Outputs:
- SQLite DB at `outputs/assignment.sqlite`
- HTML plots under `reports/`
