"""Compare LCD-station-derived daily wet-bulb against the existing NLDAS-derived rows."""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any, cast

import psycopg
from dotenv import load_dotenv

from shared_config import DATABASE_CONFIG_HINT, resolve_database_uri

pd = cast("Any", importlib.import_module("pandas"))
np = cast("Any", importlib.import_module("numpy"))

type DataFrame = Any
type Series = Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def load_nldas_rows(location_ids: list[int]) -> DataFrame:
    """Return existing NLDAS-derived wetbulb rows (from Postgres) for the given cities."""
    db_uri = resolve_database_uri()
    if not db_uri:
        msg = f"No database credentials configured. {DATABASE_CONFIG_HINT}"
        raise SystemExit(msg)

    query = """
        SELECT w.location_id, l.city, l.state, w.date, w.wetbulb, w.wetbulb_avg
        FROM public.wetbulb w
        JOIN public.locations l ON l.id = w.location_id
        WHERE w.location_id = ANY(%(location_ids)s)
        ORDER BY w.location_id, w.date
    """
    with psycopg.connect(db_uri) as conn:
        return pd.read_sql(query, conn, params={"location_ids": location_ids})


def build_comparison(lcd_df: DataFrame, nldas_df: DataFrame) -> DataFrame:
    """Return an inner join of LCD and NLDAS daily wetbulb on (location_id, date)."""
    lcd = lcd_df.copy()
    lcd["date"] = pd.to_datetime(lcd["date"]).dt.date
    nldas_rows = nldas_df.copy()
    nldas_rows["date"] = pd.to_datetime(nldas_rows["date"]).dt.date

    merged = lcd.merge(
        nldas_rows,
        on=["location_id", "date"],
        how="inner",
        suffixes=("_lcd", "_nldas"),
    )
    merged["diff"] = merged["wetbulb_lcd"] - merged["wetbulb_nldas"]
    merged["diff_avg"] = merged["wetbulb_avg_lcd"] - merged["wetbulb_avg_nldas"]
    merged["decade"] = (pd.to_datetime(merged["date"]).dt.year // 10) * 10
    return merged


def _summarize(group: DataFrame) -> Series:
    return pd.Series(
        {
            "n_days": len(group),
            "bias_max": group["diff"].mean(),
            "mae_max": group["diff"].abs().mean(),
            "p95_abs_max": group["diff"].abs().quantile(0.95),
            "bias_avg": group["diff_avg"].mean(),
            "mae_avg": group["diff_avg"].abs().mean(),
        },
    )


def summarize_comparison(merged: DataFrame) -> dict[str, DataFrame]:
    """Return overall, per-city, and per-decade summary tables."""
    overall = _summarize(merged).to_frame().T
    by_city = (
        merged.groupby(["location_id", "city", "state"])
        .apply(_summarize, include_groups=False)
        .reset_index()
    )
    by_decade = (
        merged.groupby("decade").apply(_summarize, include_groups=False).reset_index()
    )
    return {"overall": overall, "by_city": by_city, "by_decade": by_decade}


def _coverage_report(lcd_df: DataFrame, nldas_df: DataFrame) -> DataFrame:
    """Return per-city day counts to surface where LCD has gaps NLDAS doesn't (or vice versa)."""
    lcd_counts = lcd_df.groupby("location_id").size().rename("lcd_days")
    nldas_counts = nldas_df.groupby("location_id").size().rename("nldas_days")
    coverage = pd.concat([lcd_counts, nldas_counts], axis=1).fillna(0).astype(int)
    coverage["missing_from_lcd"] = coverage["nldas_days"] - coverage["lcd_days"]
    return coverage.reset_index()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcd-csv", default="lcd_pilot_wetbulb.csv")
    parser.add_argument("--out-prefix", default="lcd_vs_nldas")
    return parser.parse_args()


def main() -> None:
    """Build and write the LCD-vs-NLDAS comparison report."""
    load_dotenv(override=False)
    args = _parse_args()
    lcd_df = pd.read_csv(args.lcd_csv)
    location_ids = sorted(lcd_df["location_id"].unique().tolist())

    LOGGER.info(
        "Loading existing NLDAS-derived rows for %d location(s)...", len(location_ids)
    )
    nldas_df = load_nldas_rows(location_ids)

    merged = build_comparison(lcd_df, nldas_df)
    summaries = summarize_comparison(merged)
    coverage = _coverage_report(lcd_df, nldas_df)

    for name, table in summaries.items():
        path = f"{args.out_prefix}_{name}.csv"
        table.to_csv(path, index=False)
        LOGGER.info("Wrote %s (%d row(s)).", path, len(table))

    coverage_path = f"{args.out_prefix}_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    LOGGER.info("Wrote %s (%d row(s)).", coverage_path, len(coverage))

    LOGGER.info("\n%s", summaries["overall"].to_string(index=False))


if __name__ == "__main__":
    main()
