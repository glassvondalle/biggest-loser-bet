from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from bet_logic import (
    compute_fines_by_step,
    compute_winner_by_percent_loss,
)


st.set_page_config(page_title="Biggest Loser Bet", layout="wide")

st.title("Biggest Loser Bet")
st.caption("Reading data from `weights.csv` in the project folder.")

st.divider()


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

    parsed_dates = pd.to_datetime(value_cols, errors="coerce")
    invalid_cols = [c for c, parsed in zip(value_cols, parsed_dates) if pd.isna(parsed)]
    if invalid_cols:
        st.error(
            "Wide CSV has invalid date columns. Date columns must be parseable dates (example: YYYY-MM-DD)."
        )
        st.write("Invalid columns:", ", ".join(invalid_cols))
        st.stop()
    normalized_value_cols = [d.strftime("%Y-%m-%d") for d in parsed_dates]
    date_rename_map = dict(zip(value_cols, normalized_value_cols))
    df = df.rename(columns=date_rename_map)
    value_cols = normalized_value_cols

    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.lower().isin(["", "nan", "none"])].copy()
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    long_df = df.melt(id_vars="name", value_vars=value_cols, var_name="date", value_name="weight")
    long_df["date"] = long_df["date"].astype(str)
    long_df = long_df.dropna(subset=["weight"])
    long_df = long_df.drop_duplicates(subset=["name", "date"], keep="last")
    return long_df[["name", "date", "weight"]]


def _prepare_any_csv(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    cols = {str(c).strip().lower() for c in raw_df.columns}
    if {"name", "date", "weight"}.issubset(cols):
        long_df = _prepare_long_df(raw_df)
        return long_df, "long", sorted(long_df["date"].dropna().unique().tolist())
    if "name" in cols:
        long_df = _prepare_wide_df(raw_df)
        return long_df, "wide", sorted(long_df["date"].dropna().unique().tolist())
    st.error("CSV format not recognized. Use long (name,date,weight) or wide (name + date columns).")
    st.stop()


default_csv_path = Path(__file__).with_name("weights.csv")
if default_csv_path.exists():
    raw_csv_df = pd.read_csv(default_csv_path)
    st.caption(f"Using CSV file: {default_csv_path.name}")
else:
    st.error(f"Required file not found: {default_csv_path.name}")
    st.stop()

long_df, csv_mode, date_columns = _prepare_any_csv(raw_csv_df)
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

person_options = sorted(weights_df["person"].dropna().astype(str).unique().tolist())
contrast_palette = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # cyan
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
]
person_color_map = {
    person: contrast_palette[idx % len(contrast_palette)]
    for idx, person in enumerate(person_options)
}
selected_people = st.multiselect(
    "People to show in graphs",
    options=person_options,
    default=person_options,
)

if not selected_people:
    st.warning("Select at least one person to display the graphs.")
    st.stop()

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
chart_df_filtered = chart_df[chart_df["person"].isin(selected_people)].copy()

fig = px.line(
    chart_df_filtered,
    x="measure_date",
    y="weight_g",
    color="person",
    color_discrete_map=person_color_map,
    markers=True,
    title="Weights across scheduled measures",
)
fig.update_layout(legend_title_text="Person", xaxis_title="Measure date", yaxis_title="Weight (g)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Delta by % (vs initial weight)")
delta_pct_df = chart_df_filtered.copy()
delta_pct_df["initial_weight_g"] = delta_pct_df.groupby("person")["weight_g"].transform("first")
delta_pct_df["pct_delta"] = (
    (delta_pct_df["initial_weight_g"] - delta_pct_df["weight_g"])
    / delta_pct_df["initial_weight_g"]
    * 100.0
)
delta_pct_df = delta_pct_df.dropna(subset=["pct_delta", "initial_weight_g"])

fig_delta = px.line(
    delta_pct_df,
    x="measure_date",
    y="pct_delta",
    color="person",
    color_discrete_map=person_color_map,
    markers=True,
    title="Percentage loss versus initial weight",
)
fig_delta.update_layout(
    legend_title_text="Person",
    xaxis_title="Measure date",
    yaxis_title="Delta (%)",
)
fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig_delta, use_container_width=True)

st.subheader("Historic positions by date")
st.caption("Ranking is based on % loss versus each person's first recorded weight.")

baseline_col = date_columns[0]
ranking_rows = []
for _, row in weights_df.iterrows():
    person = str(row["person"])
    baseline = row.get(baseline_col, pd.NA)
    for date_col in date_columns:
        current = row.get(date_col, pd.NA)
        if pd.isna(baseline) or pd.isna(current) or float(baseline) <= 0:
            pct_loss = pd.NA
        else:
            pct_loss = (float(baseline) - float(current)) / float(baseline) * 100.0
        ranking_rows.append({"person": person, "date": date_col, "pct_loss": pct_loss})

ranking_df = pd.DataFrame(ranking_rows)
ranking_df["position"] = (
    ranking_df.groupby("date")["pct_loss"].rank(method="min", ascending=False)
)

def _position_to_medal(position: float | int | None) -> str:
    if pd.isna(position):
        return "—"
    pos = int(position)
    if 10 <= (pos % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(pos % 10, "th")
    return f"{pos}{suffix}"

ranking_df["position_label"] = ranking_df["position"].map(_position_to_medal)
historic_table = (
    ranking_df.pivot(index="person", columns="date", values="position_label")
    .reindex(index=person_options, columns=date_columns)
    .reset_index()
)

def _style_position_cell(val: object) -> str:
    if val == "1st":
        return "background-color: #FFD700; color: #1a1a1a; font-weight: 700;"
    if val == "2nd":
        return "background-color: #C0C0C0; color: #1a1a1a; font-weight: 700;"
    if val == "3rd":
        return "background-color: #CD7F32; color: #1a1a1a; font-weight: 700;"
    return ""

date_cols_for_style = [c for c in historic_table.columns if c != "person"]
historic_styler = historic_table.style.map(_style_position_cell, subset=date_cols_for_style)
st.dataframe(historic_styler, use_container_width=True)


st.subheader("Fines (10 COP per gram gained, $20,000 COP if date is missing)")
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

