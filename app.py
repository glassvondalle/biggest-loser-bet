from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bet_logic import (
    compute_fines_by_step,
    compute_winner_by_percent_loss,
    first_and_third_fridays,
)


st.set_page_config(page_title="Biggest Loser Bet", layout="wide")

st.title("Biggest Loser Bet (Jan–Jun timeline)")
st.caption("Upload CSV in long format (name,date,weight) or wide format (one row per person)")

st.divider()

now = datetime.now()
default_year = now.year
year = st.number_input("Year", min_value=2000, max_value=2100, value=int(default_year), step=1)

points = first_and_third_fridays(int(year))
date_columns = [p.measure_date.isoformat() for p in points]
allowed_dates = set(date_columns)


def _build_template_csv() -> str:
    rows = ["name,date,weight"]
    sample_people = ["Persona A", "Persona B", "Persona C"]
    for person in sample_people:
        for d in date_columns:
            rows.append(f"{person},{d},")
    return "\n".join(rows) + "\n"


def _build_template_csv_wide() -> str:
    headers = ["name", *date_columns]
    rows = [",".join(headers)]
    for person in ["Persona A", "Persona B", "Persona C"]:
        rows.append(",".join([person] + [""] * len(date_columns)))
    return "\n".join(rows) + "\n"


def _prepare_long_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower() for c in raw_df.columns]
    rename_map = dict(zip(raw_df.columns, cols))
    df = raw_df.rename(columns=rename_map).copy()

    required_cols = {"name", "date", "weight"}
    missing = required_cols.difference(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}. CSV must be: name,date,weight")
        st.stop()

    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.lower().isin(["", "nan", "none"])].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    invalid_date_rows = df[df["date"].isna()]
    if not invalid_date_rows.empty:
        st.error("Some rows have invalid dates. Use YYYY-MM-DD format.")
        st.dataframe(invalid_date_rows.head(20), use_container_width=True)
        st.stop()

    off_schedule = df[~df["date"].isin(allowed_dates)]
    if not off_schedule.empty:
        st.error(
            "Some dates are outside the Jan-Jun 1st/3rd Friday schedule for the selected year."
        )
        st.dataframe(off_schedule.head(20), use_container_width=True)
        st.stop()

    # Keep latest entry if duplicate person+date exists.
    df = df.dropna(subset=["weight"])
    df = df.drop_duplicates(subset=["name", "date"], keep="last")
    return df[["name", "date", "weight"]]


def _prepare_wide_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip().lower() for c in raw_df.columns]
    rename_map = dict(zip(raw_df.columns, cols))
    df = raw_df.rename(columns=rename_map).copy()

    if "name" not in df.columns:
        st.error("Wide CSV must include a 'name' column.")
        st.stop()

    value_cols = [c for c in df.columns if c != "name"]
    if not value_cols:
        st.error("Wide CSV must include date columns besides 'name'.")
        st.stop()

    invalid_cols = [c for c in value_cols if c not in allowed_dates]
    if invalid_cols:
        st.error(
            "Wide CSV has invalid date columns. Allowed columns are the schedule dates for the selected year."
        )
        st.write("Invalid columns:", ", ".join(invalid_cols))
        st.stop()

    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.lower().isin(["", "nan", "none"])].copy()
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long_df = df.melt(id_vars="name", value_vars=value_cols, var_name="date", value_name="weight")
    long_df["date"] = long_df["date"].astype(str)
    long_df = long_df.dropna(subset=["weight"])
    long_df = long_df.drop_duplicates(subset=["name", "date"], keep="last")
    return long_df[["name", "date", "weight"]]


def _prepare_any_csv(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cols = {str(c).strip().lower() for c in raw_df.columns}
    if {"name", "date", "weight"}.issubset(cols):
        return _prepare_long_df(raw_df), "long"
    if "name" in cols:
        return _prepare_wide_df(raw_df), "wide"
    st.error("CSV format not recognized. Use long (name,date,weight) or wide (name + date columns).")
    st.stop()


st.subheader("CSV input")
st.write("Supported formats: `name,date,weight` (long) or `name,<date1>,<date2>,...` (wide)")
st.download_button(
    "Download long CSV template",
    data=_build_template_csv(),
    file_name=f"weights_template_long_{int(year)}.csv",
    mime="text/csv",
)
st.download_button(
    "Download wide CSV template (one row per person)",
    data=_build_template_csv_wide(),
    file_name=f"weights_template_wide_{int(year)}.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
default_csv_path = Path(__file__).with_name("weights.csv")

if uploaded_file:
    raw_csv_df = pd.read_csv(uploaded_file)
    st.caption("Using uploaded CSV file.")
elif default_csv_path.exists():
    raw_csv_df = pd.read_csv(default_csv_path)
    st.caption(f"Using default CSV from project folder: {default_csv_path.name}")
else:
    st.error(
        f"No uploaded CSV and default file not found: {default_csv_path.name}. "
        "Upload a CSV or add the default file to the project folder."
    )
    st.stop()

long_df, csv_mode = _prepare_any_csv(raw_csv_df)
st.caption(f"Detected CSV format: {csv_mode}")

if long_df.empty:
    st.warning("No valid rows found in CSV after cleaning.")
    st.stop()

weights_df = (
    long_df.pivot(index="name", columns="date", values="weight")
    .reindex(columns=date_columns)
    .reset_index()
    .rename(columns={"name": "person"})
)

st.subheader("Timeline graph")
st.caption("For the graph only: missing scheduled weights are filled with the previous known value.")
chart_df = weights_df.copy()
chart_df[date_columns] = chart_df[date_columns].ffill(axis=1)
chart_df = chart_df.melt(
    id_vars="person",
    value_vars=date_columns,
    var_name="measure_date",
    value_name="weight_g",
)
chart_df["measure_date"] = pd.to_datetime(chart_df["measure_date"])
chart_df = chart_df.sort_values(["person", "measure_date"])

fig = px.line(
    chart_df,
    x="measure_date",
    y="weight_g",
    color="person",
    markers=True,
    title="Weights across scheduled measures",
)
fig.update_layout(legend_title_text="Person", xaxis_title="Measure date", yaxis_title="Weight (g)")
st.plotly_chart(fig, use_container_width=True)


st.subheader("Fines ($1,000 COP per 100g gained step, $20,000 COP if date is missing)")
step_df, totals_df = compute_fines_by_step(weights_df, date_columns=date_columns)
st.dataframe(totals_df, use_container_width=True)

with st.expander("Breakdown per step (fines only)", expanded=False):
    # Show all rows that produced a fee (gain fee or missing-date fee).
    show_df = step_df.copy()
    show_df = show_df[show_df["fine_cop"] > 0]
    show_df = show_df.sort_values(["person", "to_date"])
    st.dataframe(
        show_df[
            [
                "person",
                "from_date",
                "to_date",
                "prev_weight_g",
                "curr_weight_g",
                "gain_g",
                "fine_cop",
                "fee_reason",
            ]
        ],
        use_container_width=True,
    )


st.subheader("Winner (max % loss vs first measure)")
score_df, winner = compute_winner_by_percent_loss(weights_df, date_columns=date_columns)

if winner:
    st.success(f"Winner: {winner}")
else:
    st.warning("Winner cannot be computed yet (missing baseline/final weights).")

score_df_display = score_df.copy()
score_df_display["pct_loss"] = score_df_display["pct_loss"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
st.dataframe(score_df_display, use_container_width=True)

