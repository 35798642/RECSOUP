import os
import re
import itertools
from typing import Callable, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pymoo.config import Config
Config.warnings['not_compiled'] = False

from pymoo.indicators.hv import Hypervolume

# ============================================================
# Controllability Evaluation
# ============================================================

def calculate_controllability(
    x_values: pd.Series,
    y_values: pd.Series,
    condition: Callable[[float, float, float, float], bool]
) -> float:
    """
    Compute controllability score based on pairwise violations.

    Controllability is defined as:
        1 - (#violating_pairs / #total_pairs)

    Parameters
    ----------
    x_values : pd.Series
        Values on X-axis metric.
    y_values : pd.Series
        Values on Y-axis metric.
    condition : Callable
        A function defining a valid ordering between two points.

    Returns
    -------
    float
        Controllability score in [0, 1].
    """
    values = list(zip(x_values, y_values))
    n = len(values)

    if n < 2:
        return 1.0

    total_pairs = n * (n - 1) // 2
    violations = 0

    for (x1, y1), (x2, y2) in itertools.combinations(values, 2):
        if not condition(x1, y1, x2, y2):
            violations += 1

    return 1.0 - violations / total_pairs


def positive_correlation_condition(
    x1: float, y1: float,
    x2: float, y2: float
) -> bool:
    """
    Condition for positive monotonic controllability.

    A violation occurs if:
        (x1 > x2 and y1 > y2) OR (x1 < x2 and y1 < y2)

    Returns
    -------
    bool
        True if the pair satisfies controllability.
    """
    return not (
        (x1 > x2 and y1 > y2) or
        (x1 < x2 and y1 < y2)
    )


# ============================================================
# Pareto Front Extraction
# ============================================================

def extract_pareto_front(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    maximize_y: bool = True
) -> pd.DataFrame:
    """
    Extract Pareto-optimal points under (x maximize, y maximize/minimize).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_metric : str
        X-axis metric column name.
    y_metric : str
        Y-axis metric column name.
    maximize_y : bool, default=True
        Whether Y-axis metric should be maximized.

    Returns
    -------
    pd.DataFrame
        Pareto front dataframe.
    """
    data = df[[x_metric, y_metric]].dropna().copy()

    if not maximize_y:
        data[y_metric] = -data[y_metric]

    data = data.sort_values(
        by=[x_metric, y_metric],
        ascending=[False, False]
    )

    pareto_rows = []
    best_y = -np.inf

    for _, row in data.iterrows():
        if row[y_metric] > best_y:
            pareto_rows.append(row)
            best_y = row[y_metric]

    pareto_df = pd.DataFrame(pareto_rows)

    if not maximize_y:
        pareto_df[y_metric] = -pareto_df[y_metric]

    return pareto_df


# ============================================================
# Hypervolume & Method-Level Evaluation
# ============================================================

def evaluate_method(
    csv_path: str,
    x_metric: str,
    y_metric: str,
    maximize_y: bool = True,
    reference_point: Optional[Tuple[float, float]] = None
) -> Dict:
    """
    Evaluate a method using controllability, Pareto front, and hypervolume.

    Parameters
    ----------
    csv_path : str
        Path to CSV result file.
    x_metric : str
        X-axis metric.
    y_metric : str
        Y-axis metric.
    maximize_y : bool, default=True
        Whether Y metric is maximized.
    reference_point : tuple, optional
        Reference point for hypervolume computation.

    Returns
    -------
    Dict
        Evaluation results including metrics and Pareto front.
    """
    df = read_same_epoch_results(csv_path,x_metric,y_metric,epoch=5)

    controllability = calculate_controllability(
        df[x_metric],
        df[y_metric],
        positive_correlation_condition
    )

    pareto_df = extract_pareto_front(
        df,
        x_metric,
        y_metric,
        maximize_y
    )

    pareto_points = -pareto_df[[x_metric, y_metric]].values

    if reference_point is None:
        reference_point = (
            -(df[x_metric].min() - 0.01),
            -(df[y_metric].min() - 0.01),
        )

    hv = Hypervolume(reference_point)
    hypervolume_value = hv.do(pareto_points)

    return {
        "method_name": os.path.basename(csv_path),
        "data": df,
        "pareto_front": pareto_df,
        "controllability": controllability,
        "pareto_size": len(pareto_df),
        "hypervolume": hypervolume_value,
        "reference_point": reference_point,
    }


# ============================================================
# CSV Reading Utilities
# ============================================================

def read_csv_metrics(
    csv_path: str,
    x_metric: str,
    y_metric: str
) -> pd.DataFrame:
    """
    Read CSV and extract metric columns.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)
    return df[[x_metric, y_metric]].dropna().copy()


def read_mmr_csv_metrics(
    csv_path: str,
    x_metric: str,
    y_metric: str
) -> pd.DataFrame:
    """
    Read MMR-style CSV where the second row is invalid.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path, skiprows=[1])
    return df[[x_metric, y_metric]].dropna().copy()


def read_same_epoch_results(
    csv_path: str,
    x_metric: str,
    y_metric: str,
    epoch: int
) -> pd.DataFrame:
    """
    Extract results at a fixed training epoch.

    Parameters
    ----------
    epoch : int
        Training step to filter.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(csv_path)

    required_cols = ["step", x_metric, y_metric]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    filtered = df[df["step"] == epoch].copy()
    if filtered.empty:
        raise ValueError(f"No records found for epoch={epoch}")

    return filtered[[x_metric, y_metric]].dropna().copy()


