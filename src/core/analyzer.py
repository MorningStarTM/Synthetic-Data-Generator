from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from src.core.schema.time_series_schema import _safe_float, _acf, _try_pacf, _fft_top_periods, _mad_zscore
from src.core.schema.time_series_schema import CanonicalDataset, ExtractorConfig


class Analyzer:
    name: str = "base"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return True

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        raise NotImplementedError


class MarginalAnalyzer(Analyzer):
    name = "marginal"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 2 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        q = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
        ps = pd.Series(x)
        out = {
            "mean": _safe_float(np.mean(x)),
            "std": _safe_float(np.std(x, ddof=1)),
            "min": _safe_float(np.min(x)),
            "max": _safe_float(np.max(x)),
            "q05": _safe_float(q[0]),
            "q25": _safe_float(q[1]),
            "q50": _safe_float(q[2]),
            "q75": _safe_float(q[3]),
            "q95": _safe_float(q[4]),
            "skew": _safe_float(ps.skew()),
            "kurtosis": _safe_float(ps.kurtosis()),
        }
        return out


class DeltaAnalyzer(Analyzer):
    name = "delta"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 4 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        dx = np.diff(x)
        adx = np.abs(dx)

        out = {
            "delta_mean": _safe_float(np.mean(dx)),
            "delta_std": _safe_float(np.std(dx, ddof=1)) if dx.size >= 2 else None,
            "abs_delta_q50": _safe_float(np.quantile(adx, 0.50)),
            "abs_delta_q90": _safe_float(np.quantile(adx, 0.90)),
            "abs_delta_q95": _safe_float(np.quantile(adx, 0.95)),
        }
        return out


class TemporalAnalyzer(Analyzer):
    name = "temporal"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        if s.dropna().size < 10:
            return False
        if not pd.api.types.is_numeric_dtype(s):
            return False
        # If time is datetime and irregular, still okay for ACF if sequence order is consistent,
        # but warn via quality report. We'll compute anyway.
        return True

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        max_lag = min(cfg.max_lag, len(x) - 2)
        if max_lag < 1:
            return {"max_lag": 0}

        acf_vals = _acf(x, nlags=max_lag)

        out: Dict[str, Any] = {
            "max_lag": int(max_lag),
            "selected_lags": {
                f"lag_{k}": _safe_float(acf_vals[k]) for k in cfg.selected_lags if k <= max_lag
            },
        }

        if cfg.detail == "full":
            out["acf"] = [ _safe_float(v) for v in acf_vals[: max_lag + 1] ]

        pacf_max = min(cfg.pacf_max_lag, max_lag)
        pacf_vals = _try_pacf(x, nlags=pacf_max)
        if pacf_vals is not None:
            out["pacf_max_lag"] = int(pacf_max)
            if cfg.detail == "full":
                out["pacf"] = [ _safe_float(v) for v in pacf_vals[: pacf_max + 1] ]
            else:
                out["pacf_selected_lags"] = {
                    f"lag_{k}": _safe_float(pacf_vals[k])
                    for k in cfg.selected_lags
                    if k <= pacf_max
                }
        else:
            out["pacf_note"] = "statsmodels_not_available_or_failed"

        return out


class TrendAnalyzer(Analyzer):
    name = "trend"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 10 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        t = np.arange(len(x), dtype=float)
        var_t = np.var(t, ddof=1)
        slope = float(np.cov(t, x, ddof=1)[0, 1] / var_t) if var_t > 0 else 0.0
        intercept = float(np.mean(x) - slope * np.mean(t))

        w = min(cfg.rolling_window, max(5, len(x) // 5))
        rm = pd.Series(x).rolling(w).mean().dropna().values
        rm_slope = None
        if rm.size >= 10:
            tt = np.arange(len(rm), dtype=float)
            var_tt = np.var(tt, ddof=1)
            rm_slope = float(np.cov(tt, rm, ddof=1)[0, 1] / var_tt) if var_tt > 0 else 0.0

        return {
            "linear_slope": _safe_float(slope),
            "linear_intercept": _safe_float(intercept),
            "rolling_window": int(w),
            "rolling_mean_slope": _safe_float(rm_slope),
        }


class SeasonalityAnalyzer(Analyzer):
    name = "seasonality"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 16 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        top = _fft_top_periods(x, top_k=cfg.top_k_periods, min_period=cfg.min_period)
        return {"top_periods": top}


class VolatilityAnalyzer(Analyzer):
    name = "volatility"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 10 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float)
        w = min(cfg.rolling_window, max(5, len(x) // 5))
        rstd = x.rolling(w).std().dropna()
        if rstd.empty:
            return {"rolling_window": int(w), "rolling_std": None}

        q10, q50, q90 = np.quantile(rstd.values, [0.10, 0.50, 0.90])
        burst = float(q90 / (q50 + 1e-12))
        return {
            "rolling_window": int(w),
            "rolling_std_q10": _safe_float(q10),
            "rolling_std_q50": _safe_float(q50),
            "rolling_std_q90": _safe_float(q90),
            "burstiness_ratio_q90_over_q50": _safe_float(burst),
        }


class SpikeAnalyzer(Analyzer):
    name = "spikes"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return s.dropna().size >= 10 and pd.api.types.is_numeric_dtype(s)

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        x = s.dropna().astype(float).values
        rz = _mad_zscore(x)
        idx = np.where(np.abs(rz) >= cfg.spike_robust_z)[0]
        if idx.size == 0:
            return {"robust_z_threshold": float(cfg.spike_robust_z), "count": 0}

        merged = [int(idx[0])]
        for i in idx[1:]:
            if int(i) - merged[-1] > cfg.spike_min_gap:
                merged.append(int(i))

        vals = x[merged]
        return {
            "robust_z_threshold": float(cfg.spike_robust_z),
            "count": int(len(merged)),
            "mean_abs_robust_z": _safe_float(float(np.mean(np.abs(rz[merged])))),
            "mean_abs_value": _safe_float(float(np.mean(np.abs(vals)))),
            "max_abs_value": _safe_float(float(np.max(np.abs(vals)))),
            "indices_sample": merged[:10],
        }


class DataQualityAnalyzer(Analyzer):
    name = "quality"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return True

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        total = int(len(s))
        missing = int(pd.isna(s).sum())
        missing_ratio = float(missing / total) if total > 0 else None

        out: Dict[str, Any] = {
            "n_total": total,
            "n_missing": missing,
            "missing_ratio": _safe_float(missing_ratio),
        }

        if pd.api.types.is_numeric_dtype(s) and s.dropna().size >= 10:
            x = s.dropna().astype(float).values
            rz = _mad_zscore(x)
            out["outlier_ratio_robust_z_ge_3_5"] = _safe_float(float(np.mean(np.abs(rz) >= 3.5)))

        return out


class CategoricalAnalyzer(Analyzer):
    name = "categorical"

    def is_applicable(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> bool:
        return cfg.include_categorical and (not pd.api.types.is_numeric_dtype(s))

    def compute(self, s: pd.Series, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        ss = s.dropna().astype(str)
        vc = ss.value_counts()
        top = vc.head(cfg.categorical_top_k)
        total = int(vc.sum()) if vc.sum() > 0 else 1

        return {
            "unique": int(vc.size),
            "top_values": [
                {"value": v, "count": int(c), "ratio": _safe_float(float(c / total))}
                for v, c in top.items()
            ],
        }


# ---------------------------
# Cross-series analyzer (uses canonical dataset, not per-series)
# ---------------------------

class CrossSeriesAnalyzer:
    name: str = "cross_series"

    def compute(self, ctx: CanonicalDataset, cfg: ExtractorConfig) -> Dict[str, Any]:
        if not cfg.compute_cross_corr or len(ctx.series) < 2:
            return {}

        names = list(ctx.series.keys())
        series_list = [ctx.series[k] for k in names]

        # Align
        if all(isinstance(s.index, pd.DatetimeIndex) for s in series_list):
            df = pd.concat(series_list, axis=1, join="inner")
            df.columns = names
        else:
            m = min(len(s) for s in series_list)
            df = pd.DataFrame({k: ctx.series[k].values[:m] for k in names})

        df = df.dropna()
        out: Dict[str, Any] = {"aligned_length": int(len(df))}

        # Same-time correlation
        corr = df.corr().fillna(0.0)
        out["corr_matrix"] = corr.to_dict()

        # Lagged cross-correlation (top-k lags) — compact for prompts
        if cfg.compute_cross_lagged and len(df) >= 20:
            out["lagged"] = self._lagged_summary(df, cfg)

        return out

    def _lagged_summary(self, df: pd.DataFrame, cfg: ExtractorConfig) -> Dict[str, Any]:
        cols = list(df.columns)
        max_lag = int(min(cfg.cross_lag_max, max(1, len(df) - 5)))

        # For each ordered pair (a <- b): corr(a[t], b[t-k])
        lagged: Dict[str, Any] = {}
        for a in cols:
            lagged[a] = {}
            ya = df[a].values.astype(float)
            for b in cols:
                if a == b:
                    continue
                xb = df[b].values.astype(float)

                best: List[Tuple[int, float]] = []
                for k in range(1, max_lag + 1):
                    y = ya[k:]
                    x = xb[:-k]
                    if y.size < 10:
                        break
                    r = np.corrcoef(y, x)[0, 1]
                    if np.isnan(r) or np.isinf(r):
                        continue
                    best.append((k, float(r)))

                if not best:
                    continue

                best.sort(key=lambda t: abs(t[1]), reverse=True)
                top = best[: max(1, cfg.cross_lag_top_k)]
                lagged[a][b] = [{"lag": int(k), "corr": _safe_float(r)} for k, r in top]

        return lagged
