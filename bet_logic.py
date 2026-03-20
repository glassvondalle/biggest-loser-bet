from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

MISSING_WEIGHT_FINE_COP = 20000


@dataclass(frozen=True)
class TimelinePoint:
    measure_date: date
    label: str


def _first_friday_of_month(year: int, month: int) -> date:
    # calendar.weekday: Monday=0 ... Sunday=6
    first_weekday_of_month = date(year, month, 1).weekday()
    friday_index = calendar.FRIDAY  # Monday=0 ... Friday=4
    day = 1 + ((friday_index - first_weekday_of_month) % 7)
    return date(year, month, day)


def first_and_third_fridays(
    year: int, months: Iterable[int] = range(1, 7)
) -> List[TimelinePoint]:
    """
    Returns the measure dates for the first and third Friday of each month.
    Default months are Jan..Jun.
    """
    points: List[TimelinePoint] = []
    for m in months:
        first_fri = _first_friday_of_month(year, m)
        third_fri = date(year, m, first_fri.day + 14)
        points.append(TimelinePoint(measure_date=first_fri, label=f"{calendar.month_abbr[m]} (1st Fri)"))
        points.append(TimelinePoint(measure_date=third_fri, label=f"{calendar.month_abbr[m]} (3rd Fri)"))
    return points


def _cop_fine_for_gain_g(gr_gain: float) -> int:
    """
    For each 100 grams over (i.e., gained), the fine is $1,000 COP.
    This is proportional: 700g => 7,000 COP; 50g => 500 COP.
    """
    if not np.isfinite(gr_gain) or gr_gain <= 0:
        return 0
    return int(round(float(gr_gain) * 1000.0))


def compute_fines_by_step(
    weights_df: pd.DataFrame, date_columns: List[str], player_column: str = "person"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes per-step gained weight and fines.

    Returns:
      - step_df: one row per (person, step) with delta, gain, fine
      - totals_df: one row per person with total fine
    """
    if player_column not in weights_df.columns:
        raise ValueError(f"Missing column {player_column!r} in weights_df")

    persons = weights_df[player_column].astype(str)
    step_rows = []
    totals = {p: 0 for p in persons}

    prev_cols = date_columns[:-1]
    curr_cols = date_columns[1:]

    for idx, p in enumerate(persons):
        # Convert row to array for speed; still robust to NaNs.
        row_vals = weights_df.iloc[idx]

        # Missing the first scheduled measure also has a fee.
        first_col = date_columns[0]
        first_w = row_vals.get(first_col, np.nan)
        if not np.isfinite(first_w):
            totals[p] += MISSING_WEIGHT_FINE_COP
            step_rows.append(
                {
                    "person": p,
                    "from_date": "",
                    "to_date": first_col,
                    "prev_weight_g": np.nan,
                    "curr_weight_g": np.nan,
                    "delta_g": np.nan,
                    "gain_g": np.nan,
                    "fine_cop": MISSING_WEIGHT_FINE_COP,
                    "fee_reason": "missing_measure",
                }
            )

        for prev_col, curr_col in zip(prev_cols, curr_cols):
            prev_w = row_vals.get(prev_col, np.nan)
            curr_w = row_vals.get(curr_col, np.nan)
            if not np.isfinite(curr_w):
                delta = np.nan
                gain = np.nan
                fine = MISSING_WEIGHT_FINE_COP
                fee_reason = "missing_measure"
            elif not np.isfinite(prev_w):
                delta = np.nan
                gain = np.nan
                fine = 0
                fee_reason = "no_previous_measure"
            else:
                delta = float(curr_w) - float(prev_w)
                gain = max(0.0, delta)
                fine = _cop_fine_for_gain_g(gain)
                fee_reason = "weight_gain" if fine > 0 else "no_fine"

            totals[p] += fine
            step_rows.append(
                {
                    "person": p,
                    "from_date": prev_col,
                    "to_date": curr_col,
                    "prev_weight_g": prev_w if np.isfinite(prev_w) else np.nan,
                    "curr_weight_g": curr_w if np.isfinite(curr_w) else np.nan,
                    "delta_g": delta,
                    "gain_g": gain,
                    "fine_cop": fine,
                    "fee_reason": fee_reason,
                }
            )

    step_df = pd.DataFrame(step_rows)
    totals_df = pd.DataFrame(
        [{"person": p, "total_fine_cop": int(total)} for p, total in totals.items()]
    ).sort_values("total_fine_cop", ascending=False)

    return step_df, totals_df


def compute_winner_by_percent_loss(
    weights_df: pd.DataFrame,
    date_columns: List[str],
    player_column: str = "person",
) -> Tuple[pd.DataFrame, str]:
    """
    Winner is the person who lost the most % weight using the first measure as baseline.
    """
    if player_column not in weights_df.columns:
        raise ValueError(f"Missing column {player_column!r} in weights_df")
    if len(date_columns) < 2:
        raise ValueError("Need at least two measure dates to score a percent change.")

    baseline_col = date_columns[0]
    final_col = date_columns[-1]

    rows = []
    for _, r in weights_df.iterrows():
        person = str(r.get(player_column, "")).strip()
        baseline = r.get(baseline_col, np.nan)
        final = r.get(final_col, np.nan)

        if person == "":
            continue

        if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(final):
            pct_loss = np.nan
        else:
            # positive means lost weight
            pct_loss = (float(baseline) - float(final)) / float(baseline) * 100.0

        rows.append(
            {
                "person": person,
                "baseline_g": baseline if np.isfinite(baseline) else np.nan,
                "final_g": final if np.isfinite(final) else np.nan,
                "pct_loss": pct_loss,
            }
        )

    score_df = pd.DataFrame(rows).sort_values("pct_loss", ascending=False)
    # Winner is first non-NaN
    winner_row = score_df[score_df["pct_loss"].notna()].head(1)
    winner = winner_row["person"].iloc[0] if not winner_row.empty else ""
    return score_df, winner

