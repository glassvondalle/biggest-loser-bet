## biggest-loser-bet

Parallel project repo (separate from `uefa-internal-bet`).

## Streamlit app (timeline + bet scoring)

This repo includes a small Streamlit app that:

- Generates measure dates for the 1st and 3rd Friday of each month from January to June (for a chosen year).
- Reads a CSV file with each measurement in long format (`name,date,weight`).
- Plots a timeline chart (one line per person).
- Calculates:
  - Fines: compared to the immediate previous measure for each person; for every 100g gained since the previous measure, the fine is $1,000 COP.
  - Winner: the person with the highest % weight loss from the first measure (baseline) to the last measure (June's third Friday).

### Run it

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start Streamlit:
   - `streamlit run app.py`

### Data input

In the app, upload a CSV file with:

- Required CSV columns:
  - `name`: participant name
  - `date`: measure date in `YYYY-MM-DD`
  - `weight`: weight in grams

Missing weights are ignored for fines/winner calculations at the affected steps.

## Next steps

1. Decide the stack (Python/Node/etc.).
2. Add your initial code under `src/` and tests under `tests/`.

