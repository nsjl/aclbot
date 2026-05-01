# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

# plot_from_results.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re

PlotItem = Dict[str, Any]


def _as_number(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        return None


def plot_from_cypher_result(user_input: str, cypher: str, result: Any) -> List[PlotItem]:
    """
    Turn Cypher results into plot specifications.
    Returns a list of PlotItems.

    PlotItem keys (minimal contract):
      - type: "bar" | "line"
      - title: str
      - x: list
      - y: list
      - x_label: str
      - y_label: str
    """
    if not isinstance(result, list) or not result:
        return []

    # Ensure dict-like rows
    if not isinstance(result[0], dict):
        return []

    # Avoid plotting tiny results (usually meaningless)
    if len(result) < 3:
        return []    

    keys = set(result[0].keys())

    # Heuristics: common numeric keys
    numeric_candidates = [k for k in keys if re.search(r"(count|num|total|value|score)", k, re.IGNORECASE)]
    year_like = "year" in keys

    plots: List[PlotItem] = []

    # --- Case A: time series (year + count/value) => line chart
    if year_like and numeric_candidates:
        y_key = numeric_candidates[0]
        rows = []
        for r in result:
            yv = _as_number(r.get(y_key))
            if isinstance(r.get("year"), int) and yv is not None:
                rows.append((r["year"], yv))

        if rows:
            rows.sort(key=lambda t: t[0])
            x = [a for a, _ in rows]
            y = [b for _, b in rows]
            plots.append({
                "type": "line",
                "title": f"{y_key} by year",
                "x": x,
                "y": y,
                "x_label": "Year",
                "y_label": y_key,
            })
            return plots  # usually enough

    # --- Case B: category + numeric => bar chart
    # Find a string-like category key (common)
    category_candidates = [k for k in keys if k.lower() in ("name", "area", "task", "dataset", "method", "venue", "event", "type", "author")]
    if not category_candidates:
        # fallback: any key that isn't numeric-like
        category_candidates = [k for k in keys if k != "year"]

    if numeric_candidates and category_candidates:
        y_key = numeric_candidates[0]
        x_key = category_candidates[0]

        rows = []
        for r in result:
            xv = r.get(x_key)
            yv = _as_number(r.get(y_key))
            if xv is not None and yv is not None:
                rows.append((str(xv), yv))

        if rows:
            # Sort descending by value, keep top 20 to avoid insane bars
            rows.sort(key=lambda t: t[1], reverse=True)
            
            # keep reasonable size
            rows = rows[:15]

            if len(rows) < 3:
                return []
            
            x = [a for a, _ in rows]
            y = [b for _, b in rows]
            plots.append({
                "type": "bar",
                "title": f"{y_key} by {x_key}",
                "x": x,
                "y": y,
                "x_label": x_key,
                "y_label": y_key,
            })

    return plots