"""Load nflverse play-by-play via nflreadpy with optional disk cache."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

try:
    import nflreadpy as _nfl
except ImportError:
    _nfl = None  # type: ignore

from .cache import (
    pbp_cache_exists,
    pbp_cache_path,
    read_pbp_cache,
    resolve_pbp_cache_read_path,
    write_pbp_cache,
)
from .constants import CACHE_DIR, PBP_COLUMNS

logger = logging.getLogger(__name__)


def _select_available(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.debug("PBP missing columns (skipped): %s", ", ".join(missing))
    return df.select(present)


def load_pbp(
    seasons: list[int],
    *,
    cache_dir: Path = CACHE_DIR,
    no_cache: bool = False,
) -> pl.DataFrame:
    """Load play-by-play for ``seasons``; cache to Parquet when possible."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = pbp_cache_path(cache_dir, seasons)

    if not no_cache and pbp_cache_exists(path):
        logger.info("Reading PBP cache %s", path)
        return _select_available(read_pbp_cache(resolve_pbp_cache_read_path(path)), PBP_COLUMNS)

    logger.info("Downloading PBP for seasons %s", seasons)
    if _nfl is not None:
        raw = _nfl.load_pbp(seasons)
        if isinstance(raw, pl.DataFrame):
            pbp = raw
        else:
            pbp = pl.from_pandas(raw) if hasattr(raw, "to_pandas") else pl.DataFrame(raw)
    else:
        import nfl_data_py as nfl_legacy

        pdf = nfl_legacy.import_pbp_data(seasons)
        pbp = pl.from_pandas(pdf)
        pbp = _select_available(pbp, PBP_COLUMNS)

    pbp = _select_available(pbp, PBP_COLUMNS)
    if not no_cache:
        write_pbp_cache(path, pbp)
        logger.info("Wrote PBP cache %s", path)
    return pbp
