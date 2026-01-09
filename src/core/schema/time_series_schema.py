from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Utilities
# ---------------------------

DEFAULT_TIME_CANDIDATES = (
    "time",
    "date",
    "datetime",
    "timestamp",
    "period",
    "ds",
    "t",
)

def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # Try parse a sample
    try:
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def _coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def _safe_float(x: Any) -> Optional[float]:
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def _to_builtin(obj: Any) -> Any:
    """Convert numpy/pandas objects into JSON-serializable builtins."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _safe_float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj)
    if isinstance(obj, (pd.Series,)):
        return obj.to_list()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="list")
    return obj


def _round_floats(obj: Any, digits: Optional[int]) -> Any:
    def walk(x: Any) -> Any:
        x = _to_builtin(x)
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, float):
            return round(x, digits) if digits is not None else x
        return x

    return walk(obj)


def _mad_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust z-score using MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + eps
    return 0.6745 * (x - med) / mad


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Normalized autocorrelation (fast, no external deps)."""
    x = x.astype(float)
    x = x - np.mean(x)
    denom = np.dot(x, x) + 1e-12
    out = np.zeros(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = float(np.dot(x[:-k], x[k:]) / denom)
    return out


def _try_pacf(x: np.ndarray, nlags: int) -> Optional[np.ndarray]:
    """PACF via statsmodels if available; otherwise return None."""
    try:
        from statsmodels.tsa.stattools import pacf as sm_pacf  # type: ignore
        vals = sm_pacf(x, nlags=nlags, method="ywunbiased")
        return np.asarray(vals, dtype=float)
    except Exception:
        return None


def _fft_top_periods(x: np.ndarray, top_k: int, min_period: int) -> List[Dict[str, Any]]:
    """Return top periods (in steps) and normalized strengths using FFT power."""
    x = x.astype(float)
    y = x - np.mean(x)
    n = len(y)
    if n < 8:
        return []

    spec = np.fft.rfft(y)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    candidates: List[Tuple[float, float]] = []
    for i in range(1, len(freqs)):
        f = float(freqs[i])
        if f <= 0:
            continue
        period = 1.0 / f
        if period < float(min_period):
            continue
        candidates.append((period, float(power[i])))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[1], reverse=True)
    top = candidates[: max(1, top_k)]

    total = float(np.sum(power[1:])) if float(np.sum(power[1:])) > 0 else 1.0
    out: List[Dict[str, Any]] = []
    for period, pwr in top:
        out.append(
            {
                "period_steps": _safe_float(period),
                "strength": _safe_float(pwr / total),
            }
        )
    return out


def _infer_step_and_irregularity(dt_index: pd.DatetimeIndex) -> Dict[str, Any]:
    """Estimate sampling step and irregularity ratio based on time diffs."""
    if dt_index.size < 3:
        return {"has_time": True, "step": None, "irregular_ratio": None}

    diffs = dt_index.to_series().diff().dropna().values
    # diffs is array of timedeltas
    diffs_ns = np.array([d.astype("timedelta64[ns]").astype(np.int64) for d in diffs], dtype=np.int64)
    if diffs_ns.size == 0:
        return {"has_time": True, "step": None, "irregular_ratio": None}

    step_ns = int(np.median(diffs_ns))
    # irregular if too far from median
    rel = np.abs(diffs_ns - step_ns) / max(step_ns, 1)
    irregular_ratio = float(np.mean(rel > 0.1))

    return {
        "has_time": True,
        "step": str(pd.to_timedelta(step_ns, unit="ns")),
        "irregular_ratio": _safe_float(irregular_ratio),
    }


# ---------------------------
# Config & Schema
# ---------------------------

@dataclass(frozen=True)
class ExtractorConfig:
    # Schema hints (optional)
    time_col: Optional[str] = None
    id_col: Optional[str] = None
    value_col: Optional[str] = None

    # Autodetect knobs
    time_candidates: Tuple[str, ...] = DEFAULT_TIME_CANDIDATES
    long_format_max_unique_ids_ratio: float = 0.3  # heuristic

    # Preprocessing
    parse_dates: bool = True
    sort_by_time: bool = True
    drop_all_nan_series: bool = True

    # If datetime index exists:
    resample_rule: Optional[str] = None  # e.g. "1H", "1D"
    resample_agg: str = "mean"           # mean/sum/last/first

    # Missing values
    fill_method: Optional[str] = "ffill"  # ffill/bfill/interpolate/None
    interpolate_method: str = "linear"
    fill_value: Optional[float] = None

    # Stats knobs
    detail: str = "lite"  # "lite" or "full"
    round_digits: Optional[int] = 6

    max_lag: int = 40
    pacf_max_lag: int = 20
    selected_lags: Tuple[int, ...] = (1, 2, 5, 10, 20)

    rolling_window: int = 20

    top_k_periods: int = 3
    min_period: int = 2

    # Outlier/spike detection
    spike_robust_z: float = 3.5
    spike_min_gap: int = 1

    # Categorical handling
    include_categorical: bool = False
    categorical_top_k: int = 10

    # Cross-series
    compute_cross_corr: bool = True
    compute_cross_lagged: bool = True
    cross_lag_max: int = 12
    cross_lag_top_k: int = 3


@dataclass(frozen=True)
class DetectedSchema:
    format: str  # "wide" or "long"
    time_col: Optional[str]
    id_col: Optional[str]
    value_col: Optional[str]
    numeric_cols: Tuple[str, ...]
    categorical_cols: Tuple[str, ...]
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class CanonicalDataset:
    series: Dict[str, pd.Series]               # numeric time series
    categorical: Dict[str, pd.Series]          # non-numeric columns (optional)
    time_index: Optional[pd.Index]             # datetime index or RangeIndex
    schema: DetectedSchema
    quality: Dict[str, Any]                    # data quality + time info
