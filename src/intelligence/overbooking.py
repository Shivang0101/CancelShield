"""
src/intelligence/overbooking.py
================================
CancelShield Overbooking Recommendation Engine

Algorithm: Cornell Hotel School Cost-Ratio Method
Reference: Talluri & van Ryzin (2004), "The Theory and Practice of Revenue Management"

Core insight: Overbook until the marginal cost of walking one more guest
equals the marginal revenue of the extra booking.

Threshold ratio = C_walk / (C_walk + C_empty)
                = 2×ADR / (2×ADR + 1×ADR)
                = 2/3 = 0.667

Find N = largest integer where P(total_cancellations >= N) >= 0.667
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson

logger = logging.getLogger(__name__)


# Cancellation Distribution Modelling

def cancellation_distribution(
    cancel_proba: np.ndarray,
) -> Tuple[float, float, object]:
    """
    Model the distribution of total cancellations for a set of bookings.

    Why Poisson?
    Each booking independently cancels with some probability p_i.
    The sum of independent Bernoulli(p_i) random variables is well-approximated
    by Poisson(lambda = sum(p_i)) when all p_i are small (Poisson limit theorem).

    This is standard practice in hotel revenue management.

    Parameters
    ----------
    cancel_proba : Array of individual P(cancel) from Module 1

    Returns
    -------
    (lambda_hat, std_hat, poisson_distribution_object)
    """
    lambda_hat = float(np.sum(cancel_proba))
    std_hat = float(np.sqrt(lambda_hat))  # Poisson std = sqrt(lambda)

    dist = poisson(lambda_hat)

    logger.info(
        "Cancellation distribution: Poisson(λ=%.1f), std=%.1f, "
        "95%% CI=[%.0f, %.0f]",
        lambda_hat, std_hat,
        dist.ppf(0.025), dist.ppf(0.975),
    )

    return lambda_hat, std_hat, dist


# Optimal Overbooking Buffer

def optimal_overbook(
    cancel_proba: np.ndarray,
    mean_adr: float,
    risk_tolerance: float = 0.5,
    walk_cost_base_multiplier: float = 2.0,
    max_overbook: int = 100,
) -> Dict:
    """
    Compute the optimal number of rooms to overbook using the cost-ratio method.

    Algorithm:
    ----------
    1. Compute lambda = sum(P_i) — expected total cancellations
    2. Compute C_walk = walk_cost_multiplier × mean_adr
       C_empty = mean_adr
    3. Compute threshold_ratio = C_walk / (C_walk + C_empty)
    4. Find N = largest integer such that P(cancellations >= N) >= threshold_ratio
    5. That N is the safe overbooking buffer

    Risk Tolerance Dial:
    --------------------
    risk_tolerance ∈ [0, 1] adjusts the C_walk multiplier:
      0.0 (conservative):   C_walk = 5× ADR → overbook very little
      0.5 (balanced):       C_walk = 2× ADR → standard overbooking
      1.0 (aggressive):     C_walk = 1× ADR → maximise overbooking

    Parameters
    ----------
    cancel_proba       : Array of individual P(cancel) values
    mean_adr           : Mean ADR for this arrival date (EUR)
    risk_tolerance     : 0.0–1.0 scalar
    max_overbook       : Safety cap on buffer (never recommend more than this)

    Returns
    -------
    dict: lambda, std, threshold_ratio, recommended_buffer, walk_risk_pct,
          expected_extra_revenue, cost_walk, cost_empty
    """
    lambda_hat, std_hat, dist = cancellation_distribution(cancel_proba)

    # Adjust walk cost based on risk tolerance
    # Interpolate: risk_tolerance=0 → multiplier=5, risk_tolerance=1 → multiplier=1
    walk_multiplier = 5.0 - (4.0 * risk_tolerance)
    c_walk = walk_multiplier * mean_adr
    c_empty = mean_adr

    threshold_ratio = c_walk / (c_walk + c_empty)

    # Find optimal N: largest N where P(total_cancellations >= N) >= threshold_ratio
    # P(X >= N) = 1 - P(X <= N-1) = 1 - CDF(N-1)
    best_n = 0
    for n in range(0, min(int(lambda_hat * 3) + 1, max_overbook)):
        p_at_least_n = 1 - dist.cdf(n - 1)  # P(cancellations >= n)
        if p_at_least_n >= threshold_ratio:
            best_n = n
        else:
            break  # CDF is monotone; once we fall below, we're done

    best_n = min(best_n, max_overbook)

    # Walk risk = P(cancellations < best_n) = P(fewer cancellations than we assumed)
    walk_risk = float(dist.cdf(best_n - 1)) if best_n > 0 else 0.0

    expected_extra_revenue = float(best_n * mean_adr)
    expected_walked = sum(
        max(0, best_n - k) * dist.pmf(k)
        for k in range(0, int(lambda_hat * 4) + 1)
    )
    expected_walk_cost = float(expected_walked * c_walk)
    result = {
        "lambda": round(lambda_hat, 2),
        "std": round(std_hat, 2),
        "threshold_ratio": round(threshold_ratio, 4),
        "walk_multiplier": round(walk_multiplier, 2),
        "c_walk_eur": round(c_walk, 2),
        "c_empty_eur": round(c_empty, 2),
        "recommended_buffer": int(best_n),
        "walk_risk_pct": round(walk_risk * 100, 1),
        "expected_extra_revenue_eur": round(expected_extra_revenue, 2),
        "expected_walk_cost_eur": round(expected_walk_cost, 2),
        "net_expected_gain_eur": round(expected_extra_revenue - expected_walk_cost, 2),
        "risk_tolerance": risk_tolerance,
        "n_bookings_analysed": len(cancel_proba),
    }

    logger.info(
        "Overbooking recommendation: buffer=%d | walk_risk=%.1f%% | "
        "extra_revenue=€%.0f | C_walk=%.1fx ADR",
        best_n, walk_risk * 100, expected_extra_revenue, walk_multiplier,
    )

    return result


# ===========================================================================
# Overbooking Calendar (All Arrival Dates)
# ===========================================================================

def overbook_calendar(
    bookings_df: pd.DataFrame,
    clf_model: Any,
    X_features: pd.DataFrame,
    adr_series: pd.Series,
    risk_tolerance: float = 0.5,
    date_col: str = "arrival_date",
    min_bookings_per_date: int = 5,
) -> pd.DataFrame:
    """
    Run the overbooking recommendation for every arrival date in the dataset.

    Parameters
    ----------
    bookings_df           : Full bookings DataFrame
    clf_model             : Fitted Module 1 classifier
    X_features            : Feature matrix (same order as bookings_df)
    adr_series            : ADR values (actual or predicted from Module 2)
    risk_tolerance        : Global risk tolerance parameter
    min_bookings_per_date : Minimum bookings on a date to make a recommendation

    Returns
    -------
    DataFrame: arrival_date | n_bookings | lambda | std | buffer |
               walk_risk_pct | extra_revenue | mean_adr
    Sorted by arrival_date.
    """
    # Get probabilities
    proba = clf_model.predict_proba(X_features.values)[:, 1]

    df = bookings_df.copy().reset_index(drop=True)
    df["p_cancel"] = proba
    df["adr_used"] = adr_series.values if isinstance(adr_series, pd.Series) else adr_series

    if date_col not in df.columns:
        logger.warning("date_col '%s' not found in DataFrame.", date_col)
        return pd.DataFrame()

    rows = []
    for date, group in df.groupby(date_col):
        if len(group) < min_bookings_per_date:
            continue

        mean_adr = float(group["adr_used"].mean())
        cancel_p = group["p_cancel"].values

        rec = optimal_overbook(cancel_p, mean_adr, risk_tolerance)
        rows.append({
            "arrival_date": date,
            "n_bookings": len(group),
            "predicted_cancellations": rec["lambda"],
            "cancellation_std": rec["std"],
            "recommended_buffer": rec["recommended_buffer"],
            "walk_risk_pct": rec["walk_risk_pct"],
            "expected_extra_revenue_eur": rec["expected_extra_revenue_eur"],
            "expected_walk_cost_eur": rec["expected_walk_cost_eur"],
            "net_expected_gain_eur": rec["net_expected_gain_eur"],
            "mean_adr_eur": round(mean_adr, 2),
            "threshold_ratio": rec["threshold_ratio"],
        })

    calendar_df = pd.DataFrame(rows).sort_values("arrival_date").reset_index(drop=True)
    logger.info("Overbooking calendar: %d dates processed.", len(calendar_df))
    return calendar_df


# ===========================================================================
# Revenue Impact Analysis
# ===========================================================================

def revenue_impact_analysis(
    buffer_range: range,
    lambda_hat: float,
    mean_adr: float,
    capacity: int = 200,
    risk_tolerance: float = 0.5,
) -> pd.DataFrame:
    """
    Compute expected revenue and cost for a range of overbooking buffers.
    Useful for the sensitivity analysis in the dashboard.

    Returns
    -------
    DataFrame with columns: buffer, extra_revenue, walk_cost, net_gain,
                             walk_risk_pct, expected_rooms_walked
    """
    dist = poisson(lambda_hat)
    walk_multiplier = 5.0 - (4.0 * risk_tolerance)

    rows = []
    for n in buffer_range:
        # Expected extra revenue from n additional bookings that don't cancel
        extra_revenue = n * mean_adr

        # Expected rooms we need to walk: max(0, bookings_that_stayed - capacity)
        # = max(0, (capacity + n) - cancellations - capacity)
        # = max(0, n - cancellations)
        # E[max(0, n - X)] where X ~ Poisson(lambda)
        expected_walked = sum(
            max(0, n - k) * dist.pmf(k) for k in range(0, int(lambda_hat * 4) + 1)
        )
        walk_cost = expected_walked * walk_multiplier * mean_adr

        # P(walk at least one guest) = P(n > cancellations) = P(X < n) = CDF(n-1)
        walk_risk = float(dist.cdf(n - 1)) if n > 0 else 0.0

        rows.append({
            "buffer": n,
            "extra_revenue_eur": round(extra_revenue, 2),
            "walk_cost_eur": round(walk_cost, 2),
            "net_gain_eur": round(extra_revenue - walk_cost, 2),
            "walk_risk_pct": round(walk_risk * 100, 1),
            "expected_rooms_walked": round(expected_walked, 2),
        })

    return pd.DataFrame(rows)

# Sensitivity Analysis

def sensitivity_analysis(
    cancel_proba: np.ndarray,
    mean_adr: float,
    risk_tolerances: List[float] = None,
) -> pd.DataFrame:
    """
    Show how the recommended buffer changes across risk tolerance values.
    This powers the risk slider in the Plotly Dash dashboard.

    Returns
    -------
    DataFrame: risk_tolerance | walk_multiplier | recommended_buffer |
               walk_risk_pct | extra_revenue | net_gain
    """
    if risk_tolerances is None:
        risk_tolerances = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rows = []
    for rt in risk_tolerances:
        rec = optimal_overbook(cancel_proba, mean_adr, risk_tolerance=rt)
        rows.append({
            "risk_tolerance": rt,
            "walk_cost_multiplier": rec["walk_multiplier"],
            "recommended_buffer": rec["recommended_buffer"],
            "walk_risk_pct": rec["walk_risk_pct"],
            "expected_extra_revenue_eur": rec["expected_extra_revenue_eur"],
            "net_expected_gain_eur": rec["net_expected_gain_eur"],
        })

    return pd.DataFrame(rows)


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_cancellation_distribution(
    cancel_proba: np.ndarray,
    recommended_buffer: int,
    mean_adr: float,
    risk_tolerance: float = 0.5,
    save_path: str = None,
) -> str:
    """
    Plot the Poisson distribution of expected cancellations with the
    overbooking threshold marked. This is the key visual in the dashboard.
    """
    lambda_hat = np.sum(cancel_proba)
    dist = poisson(lambda_hat)
    x_range = range(max(0, int(lambda_hat - 4 * np.sqrt(lambda_hat))),
                    int(lambda_hat + 5 * np.sqrt(lambda_hat)) + 1)
    pmf_vals = [dist.pmf(k) for k in x_range]

    if save_path is None:
        save_path = "/tmp/cancellation_distribution.png"

    fig, ax = plt.subplots(figsize=(10, 5))

    # Colour bars: green (safe) vs red (walk risk)
    bar_colors = ["#4CAF50" if k <= recommended_buffer else "#F44336" for k in x_range]
    ax.bar(list(x_range), pmf_vals, color=bar_colors, alpha=0.8, edgecolor="white")
    ax.axvline(
        recommended_buffer,
        color="navy",
        linestyle="--",
        lw=2,
        label=f"Recommended buffer: {recommended_buffer}",
    )
    ax.axvline(
        lambda_hat,
        color="orange",
        linestyle=":",
        lw=2,
        label=f"Expected cancellations: {lambda_hat:.1f}",
    )

    # Walk risk annotation
    walk_risk = float(dist.cdf(recommended_buffer - 1)) if recommended_buffer > 0 else 0.0
    ax.text(
        recommended_buffer + 0.5,
        max(pmf_vals) * 0.9,
        f"Walk risk: {walk_risk*100:.1f}%",
        color="navy",
        fontsize=10,
    )

    ax.set_xlabel("Total Cancellations on This Date")
    ax.set_ylabel("Probability")
    ax.set_title(
        f"Expected Cancellation Distribution\n"
        f"λ={lambda_hat:.1f} | Buffer={recommended_buffer} | "
        f"Extra Revenue: €{recommended_buffer * mean_adr:,.0f}"
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)

    return save_path