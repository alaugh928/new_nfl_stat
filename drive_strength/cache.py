"""Parquet cache for raw PBP."""

from __future__ import annotations

from pathlib import Path

import polars as pl

try:
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


def season_list_tag(seasons: list[int]) -> str:
    return "_".join(str(s) for s in sorted(seasons))


def pbp_cache_path(cache_dir: Path, seasons: list[int]) -> Path:
    tag = season_list_tag(seasons)
    suffix = ".parquet" if _HAS_PYARROW else ".csv"
    return cache_dir / f"pbp_{tag}{suffix}"


def write_pbp_cache(path: Path, pbp_df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_PYARROW:
        pbp_df.write_parquet(path, compression="snappy")
    else:
        out = path if path.suffix == ".csv" else path.with_suffix(".csv")
        pbp_df.write_csv(out)


def read_pbp_cache(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def pbp_cache_exists(path: Path) -> bool:
    if path.exists():
        return True
    if path.suffix == ".parquet":
        return path.with_suffix(".csv").exists()
    return False


def resolve_pbp_cache_read_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix == ".parquet":
        csv_alt = path.with_suffix(".csv")
        if csv_alt.exists():
            return csv_alt
    return path
