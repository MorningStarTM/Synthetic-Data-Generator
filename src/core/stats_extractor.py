from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from src.core.schema.time_series_schema import _round_floats, _is_datetime_like, _coerce_datetime, _infer_step_and_irregularity
from src.core.schema.time_series_schema import CanonicalDataset, ExtractorConfig, DetectedSchema
from src.core.analyzer import (
    Analyzer, CrossSeriesAnalyzer, 
    DataQualityAnalyzer, MarginalAnalyzer,
    DeltaAnalyzer, TemporalAnalyzer, 
    TrendAnalyzer, SeasonalityAnalyzer,
    VolatilityAnalyzer, SpikeAnalyzer,
    CategoricalAnalyzer, CanonicalDataset
)





class TimeSeriesStatsExtractor:
    """
    General-purpose time-series stats extractor (CSV -> prompt-friendly JSON).

    Design goals:
    - Works for wide OR long time-series CSVs
    - Robust schema auto-detection (time col, wide vs long)
    - Canonical internal format (dict of pd.Series)
    - Plugin analyzers with applicability checks (graceful degradation)
    - Detail levels: "lite" (LLM-friendly) vs "full" (debug/research)
    - Optional categorical support


    Main entry:
      - from_csv(path) -> dict stats
      - from_dataframe(df) -> dict stats
      - to_prompt_json(stats) -> pretty JSON string
    """

    def __init__(
        self,
        cfg: ExtractorConfig,
        analyzers: Optional[Sequence[Analyzer]] = None,
        cross_analyzer: Optional[CrossSeriesAnalyzer] = None,
    ):
        self.cfg = cfg
        self.analyzers = list(analyzers) if analyzers is not None else [
            DataQualityAnalyzer(),
            MarginalAnalyzer(),
            DeltaAnalyzer(),
            TemporalAnalyzer(),
            TrendAnalyzer(),
            SeasonalityAnalyzer(),
            VolatilityAnalyzer(),
            SpikeAnalyzer(),
        ]
        self.cross_analyzer = cross_analyzer if cross_analyzer is not None else CrossSeriesAnalyzer()

    def from_csv(self, csv_path: str, **read_csv_kwargs: Any) -> Dict[str, Any]:
        df = pd.read_csv(csv_path, **read_csv_kwargs)
        return self.from_dataframe(df)

    def from_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        schema = self._detect_schema(df)
        canonical = self._to_canonical(df, schema)
        canonical = self._preprocess(canonical)

        per_series = self._compute_per_series(canonical)
        cross = self.cross_analyzer.compute(canonical, self.cfg) if self.cfg.compute_cross_corr else {}

        payload = {
            "meta": self._meta(canonical),
            "schema_report": {
                "format": schema.format,
                "time_col": schema.time_col,
                "id_col": schema.id_col,
                "value_col": schema.value_col,
                "numeric_cols": list(schema.numeric_cols),
                "categorical_cols": list(schema.categorical_cols),
                "notes": list(schema.notes),
            },
            "data_quality": canonical.quality,
            "per_series": per_series,
            "cross_series": cross,
        }

        return _round_floats(payload, self.cfg.round_digits)

    def to_prompt_json(self, stats: Dict[str, Any], indent: int = 2) -> str:
        return json.dumps(stats, ensure_ascii=False, indent=indent)

    # -----------------------
    # Schema detection
    # -----------------------

    def _detect_schema(self, df: pd.DataFrame) -> DetectedSchema:
        notes: List[str] = []
        cols = list(df.columns)

        # Explicit long-format config wins
        if self.cfg.time_col and self.cfg.id_col and self.cfg.value_col:
            if all(c in df.columns for c in (self.cfg.time_col, self.cfg.id_col, self.cfg.value_col)):
                numeric_cols: List[str] = []
                categorical_cols: List[str] = []
                for c in cols:
                    if c in (self.cfg.time_col, self.cfg.id_col, self.cfg.value_col):
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        numeric_cols.append(c)
                    else:
                        categorical_cols.append(c)
                return DetectedSchema(
                    format="long",
                    time_col=self.cfg.time_col,
                    id_col=self.cfg.id_col,
                    value_col=self.cfg.value_col,
                    numeric_cols=tuple(numeric_cols),
                    categorical_cols=tuple(categorical_cols),
                    notes=("explicit_long_schema",),
                )

        # Candidate time col
        time_col = self.cfg.time_col if (self.cfg.time_col in df.columns if self.cfg.time_col else False) else None
        if time_col is None:
            # name-based candidates
            lowered = {c.lower(): c for c in cols}
            for cand in self.cfg.time_candidates:
                if cand in lowered:
                    candidate = lowered[cand]
                    if _is_datetime_like(df[candidate]):
                        time_col = candidate
                        notes.append(f"autodetected_time_col:{candidate}")
                        break
            # heuristic: any datetime-like column
            if time_col is None:
                for c in cols:
                    if _is_datetime_like(df[c]):
                        time_col = c
                        notes.append(f"autodetected_time_col:{c}")
                        break

        # Long-format heuristic: look for (time, id, value) structure
        # If there is one datetime-like col, one categorical-like col, one numeric-like col
        # and rows >> unique ids, likely long.
        def is_numeric_col(c: str) -> bool:
            return pd.api.types.is_numeric_dtype(df[c])

        datetime_cols = [c for c in cols if (c == time_col) or _is_datetime_like(df[c])]
        numeric_cols_all = [c for c in cols if is_numeric_col(c)]
        non_numeric_cols = [c for c in cols if c not in numeric_cols_all]

        # Try infer id_col and value_col
        id_col = None
        value_col = None

        if time_col is not None:
            # candidate id is non-numeric and not time_col
            id_candidates = [c for c in non_numeric_cols if c != time_col]
            # candidate value is numeric
            value_candidates = [c for c in numeric_cols_all if c != time_col]

            if id_candidates and value_candidates:
                # pick most "id-like": many repeats but not all unique
                best = None
                for c in id_candidates:
                    nunique = df[c].nunique(dropna=True)
                    ratio = nunique / max(len(df), 1)
                    if ratio < self.cfg.long_format_max_unique_ids_ratio:
                        score = nunique  # more unique IDs but still repeated
                        if best is None or score > best[0]:
                            best = (score, c)
                if best is not None:
                    id_col = best[1]
                    value_col = value_candidates[0]
                    notes.append("long_format_heuristic_triggered")

        # Decide format
        if time_col is not None and id_col is not None and value_col is not None:
            # Long format
            extra_numeric = [c for c in numeric_cols_all if c not in (value_col,) and c != time_col]
            extra_categ = [c for c in non_numeric_cols if c not in (time_col, id_col)]
            return DetectedSchema(
                format="long",
                time_col=time_col,
                id_col=id_col,
                value_col=value_col,
                numeric_cols=tuple(extra_numeric),
                categorical_cols=tuple(extra_categ),
                notes=tuple(notes),
            )

        # Otherwise wide
        numeric_cols = [c for c in numeric_cols_all if c != time_col]
        categorical_cols = [c for c in cols if c not in numeric_cols and c != time_col]
        if time_col is None:
            notes.append("no_time_col_detected_using_range_index")

        return DetectedSchema(
            format="wide",
            time_col=time_col,
            id_col=None,
            value_col=None,
            numeric_cols=tuple(numeric_cols),
            categorical_cols=tuple(categorical_cols),
            notes=tuple(notes),
        )

    # -----------------------
    # Canonical conversion
    # -----------------------

    def _to_canonical(self, df: pd.DataFrame, schema: DetectedSchema) -> CanonicalDataset:
        df = df.copy()

        # Parse time col if exists
        time_index: Optional[pd.Index] = None
        if schema.time_col is not None and schema.time_col in df.columns and self.cfg.parse_dates:
            df[schema.time_col] = _coerce_datetime(df[schema.time_col])

        if schema.format == "long":
            assert schema.time_col and schema.id_col and schema.value_col
            # Drop rows without values
            df = df.dropna(subset=[schema.value_col])
            if schema.time_col in df.columns:
                df = df.sort_values(schema.time_col)

            series: Dict[str, pd.Series] = {}
            for sid, g in df.groupby(schema.id_col):
                s = g[schema.value_col].astype(float)
                if schema.time_col in g.columns:
                    s.index = pd.DatetimeIndex(g[schema.time_col])
                series[str(sid)] = s

            categorical: Dict[str, pd.Series] = {}
            if self.cfg.include_categorical:
                # Keep extra categorical columns only if they are constant per time/id group
                for c in schema.categorical_cols:
                    categorical[c] = df[c]

            # time index is not single shared in long-format; keep None
            time_index = None

            quality = self._dataset_quality(series, schema, time_index)
            return CanonicalDataset(series=series, categorical=categorical, time_index=time_index, schema=schema, quality=quality)

        # Wide format
        if schema.time_col is not None and schema.time_col in df.columns:
            if self.cfg.sort_by_time:
                df = df.sort_values(schema.time_col)
            df = df.set_index(schema.time_col)
            time_index = df.index
        else:
            time_index = pd.RangeIndex(start=0, stop=len(df), step=1)
            df.index = time_index

        series = {}
        for c in schema.numeric_cols:
            if c not in df.columns:
                continue
            s = df[c]
            if self.cfg.drop_all_nan_series and s.dropna().empty:
                continue
            series[str(c)] = s

        categorical: Dict[str, pd.Series] = {}
        if self.cfg.include_categorical:
            for c in schema.categorical_cols:
                if c in df.columns:
                    categorical[str(c)] = df[c]

        quality = self._dataset_quality(series, schema, time_index)
        return CanonicalDataset(series=series, categorical=categorical, time_index=time_index, schema=schema, quality=quality)

    # -----------------------
    # Preprocessing
    # -----------------------

    def _preprocess(self, ds: CanonicalDataset) -> CanonicalDataset:
        series = {k: v.copy() for k, v in ds.series.items()}

        # If datetime index exists, handle duplicates, resample, etc.
        # For wide format, time index is shared (ds.time_index). For long format, each series may have its own dt index.
        for name, s in series.items():
            # Ensure time order if datetime
            if isinstance(s.index, pd.DatetimeIndex):
                s = s.sort_index()
                # drop duplicate timestamps by aggregation
                if s.index.has_duplicates:
                    s = s.groupby(level=0).mean()
                # resample if configured
                if self.cfg.resample_rule:
                    s = self._resample(s)
            # fill missing
            s = self._fill_missing(s)
            series[name] = s

        # Update quality after preprocessing
        quality = self._dataset_quality(series, ds.schema, ds.time_index)
        return CanonicalDataset(series=series, categorical=ds.categorical, time_index=ds.time_index, schema=ds.schema, quality=quality)

    def _resample(self, s: pd.Series) -> pd.Series:
        rule = self.cfg.resample_rule
        assert rule is not None
        if not isinstance(s.index, pd.DatetimeIndex):
            return s

        agg = self.cfg.resample_agg
        if agg == "mean":
            return s.resample(rule).mean()
        if agg == "sum":
            return s.resample(rule).sum()
        if agg == "last":
            return s.resample(rule).last()
        if agg == "first":
            return s.resample(rule).first()

        raise ValueError(f"Unsupported resample_agg: {agg}")

    def _fill_missing(self, s: pd.Series) -> pd.Series:
        s2 = s.copy()
        if self.cfg.fill_method == "ffill":
            s2 = s2.ffill()
        elif self.cfg.fill_method == "bfill":
            s2 = s2.bfill()
        elif self.cfg.fill_method == "interpolate":
            # only works for numeric
            if pd.api.types.is_numeric_dtype(s2):
                s2 = s2.interpolate(method=self.cfg.interpolate_method, limit_direction="both")
        elif self.cfg.fill_method is None:
            pass
        else:
            raise ValueError(f"Unsupported fill_method: {self.cfg.fill_method}")

        if self.cfg.fill_value is not None:
            s2 = s2.fillna(self.cfg.fill_value)

        return s2.dropna()

    # -----------------------
    # Compute
    # -----------------------

    def _compute_per_series(self, ds: CanonicalDataset) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, s in ds.series.items():
            entry: Dict[str, Any] = {"name": name, "n": int(s.dropna().size)}

            for analyzer in self.analyzers:
                if analyzer.is_applicable(s, ds, self.cfg):
                    try:
                        entry[analyzer.name] = analyzer.compute(s, ds, self.cfg)
                    except Exception as e:
                        entry[analyzer.name] = {"error": str(e)}
                else:
                    if self.cfg.detail == "full":
                        entry[analyzer.name] = {"skipped": True}

            out[name] = entry

        # Optional categorical stats
        if self.cfg.include_categorical and ds.categorical:
            cat_an = CategoricalAnalyzer()
            out["_categorical"] = {}
            for cname, cs in ds.categorical.items():
                try:
                    out["_categorical"][cname] = cat_an.compute(cs, ds, self.cfg)
                except Exception as e:
                    out["_categorical"][cname] = {"error": str(e)}

        return out

    # -----------------------
    # Meta / Quality
    # -----------------------

    def _dataset_quality(self, series: Dict[str, pd.Series], schema: DetectedSchema, time_index: Optional[pd.Index]) -> Dict[str, Any]:
        lengths = {k: int(v.dropna().size) for k, v in series.items()}
        has_dt = False
        irregular_ratio = None
        step = None

        # For wide format, time_index can be datetime
        if isinstance(time_index, pd.DatetimeIndex):
            has_dt = True
            info = _infer_step_and_irregularity(time_index)
            step = info.get("step")
            irregular_ratio = info.get("irregular_ratio")
        else:
            # For long format, if any series has DatetimeIndex, try infer on first one
            for s in series.values():
                if isinstance(s.index, pd.DatetimeIndex):
                    has_dt = True
                    info = _infer_step_and_irregularity(s.index)
                    step = info.get("step")
                    irregular_ratio = info.get("irregular_ratio")
                    break

        return {
            "num_series": int(len(series)),
            "lengths": lengths,
            "has_datetime_index": bool(has_dt),
            "estimated_step": step,
            "irregular_ratio": irregular_ratio,
        }

    def _meta(self, ds: CanonicalDataset) -> Dict[str, Any]:
        return {
            "detail": self.cfg.detail,
            "round_digits": self.cfg.round_digits,
            "series_count": int(len(ds.series)),
        }



# ---------------------------
# Example usage
# ---------------------------

# if __name__ == "__main__":
#     cfg = ExtractorConfig(
#         detail="lite",               # "lite" for LLM prompts, "full" for debugging
#         include_categorical=False,    # set True if you want categorical stats too
#         resample_rule=None,           # set e.g. "1D" if needed
#         fill_method="ffill",
#         compute_cross_corr=True,
#         compute_cross_lagged=True,
#     )

#     extractor = TimeSeriesStatsExtractor(cfg)

#     stats = extractor.from_csv("your_timeseries.csv")
#     print(extractor.to_prompt_json(stats))