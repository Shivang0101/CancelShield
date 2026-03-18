"""
Translates model predictions into EUR business impact.

Key concepts:
  Revenue at Risk = P(cancel) × ADR × total_nights
  This converts a probabilistic ML output into an actionable currency figure
  that a hotel revenue manager understands and can act on.
"""

import logging
from typing import Dict, Optional, Tuple
 
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Single Booking Revenue at Risk

def booking_revenue_at_risk(
    p_cancel: float,
    adr: float,
    total_nights: int,
    discount_rate: float = 1.0,
) -> float:
    """
    Compute revenue at risk for a single booking.

    Formula: Revenue at Risk = P(cancel) × ADR × total_nights × discount_rate

    Example:
      p_cancel=0.73, ADR=112 EUR, total_nights=4, discount_rate=0.9 (partial rebook)
      → Revenue at Risk = 0.73 × 112 × 4 × 0.9 = 293.90 EUR

    Parameters
    ----------
    p_cancel      : Predicted cancellation probability from Module 1
    adr           : Predicted or actual Average Daily Rate (EUR)
    total_nights  : stays_in_weekend_nights + stays_in_week_nights
    discount_rate : Expected rebooking recovery fraction (0-1). Default 1.0 = no rebook.

    Returns : Revenue at risk in EUR
    """
    rar = float(p_cancel) * float(adr) * float(total_nights) * float(discount_rate)
    return round(rar, 2)


def risk_level(p_cancel: float, threshold_low: float = 0.3, threshold_high: float = 0.6) -> str:
    """Categorise booking into LOW / MEDIUM / HIGH risk."""
    if p_cancel < threshold_low:
        return "LOW"
    elif p_cancel < threshold_high:
        return "MEDIUM"
    else:
        return "HIGH"

# Batch Revenue at Risk (All Bookings for a Date)

def daily_revenue_at_risk(
    bookings_df: pd.DataFrame,
    cancel_proba: np.ndarray,
    adr_predictions: np.ndarray = None,
    date_col: str = "arrival_date",
    target_date: str = None,
) -> pd.DataFrame:
    """
    Compute revenue at risk for all bookings on a given arrival date
    (or all dates if target_date is None).

    Parameters
    ----------
    bookings_df     : DataFrame with columns: arrival_date, adr,
                      stays_in_weekend_nights, stays_in_week_nights
    cancel_proba    : Array of P(cancel) from Module 1 (same row order)
    adr_predictions : Optional ADR predictions from Module 2 (overrides actual adr)
    target_date     : Filter to single date (format: 'YYYY-MM-DD')

    Returns
    -------
    DataFrame with columns: arrival_date, booking_index, p_cancel,
                              adr_used, total_nights, revenue_at_risk, risk_level
    Sorted by revenue_at_risk descending.
    """
    df = bookings_df.copy().reset_index(drop=True)
    df["p_cancel"] = cancel_proba
    df["adr_used"] = adr_predictions if adr_predictions is not None else df["adr"].values
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_nights"] = df["total_nights"].replace(0, 1)  # Guard divide-by-zero

    df["revenue_at_risk"] = df.apply(
        lambda row: booking_revenue_at_risk(
            row["p_cancel"], row["adr_used"], row["total_nights"]
        ),
        axis=1,
    )
    df["risk_level"] = df["p_cancel"].apply(risk_level)

    if target_date is not None and date_col in df.columns:
        df = df[df[date_col].astype(str).str[:10] == target_date]

    return df.sort_values("revenue_at_risk", ascending=False).reset_index(drop=True)


# Segment Revenue Risk Breakdown

def segment_revenue_risk(
    bookings_df: pd.DataFrame,
    cancel_proba: np.ndarray,
    adr_predictions: np.ndarray = None,
    segment_col: str = "distribution_channel",
) -> pd.DataFrame:
    """
    Break down total revenue at risk by a segment dimension.
    Useful for identifying which channel/customer_type drives most risk.

    Returns
    -------
    DataFrame with columns: segment, n_bookings, total_revenue_at_risk,
                             avg_p_cancel, avg_adr, pct_high_risk
    Sorted by total_revenue_at_risk descending.
    """
    detail = daily_revenue_at_risk(bookings_df, cancel_proba, adr_predictions)

    if segment_col not in detail.columns:
        logger.warning("Segment column '%s' not found in DataFrame.", segment_col)
        return pd.DataFrame()

    result = (
        detail.groupby(segment_col)
        .agg(
            n_bookings=(segment_col, "count"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
            avg_p_cancel=("p_cancel", "mean"),
            avg_adr=("adr_used", "mean"),
            pct_high_risk=("risk_level", lambda x: (x == "HIGH").mean() * 100),
        )
        .sort_values("total_revenue_at_risk", ascending=False)
        .reset_index()
    )

    result["total_revenue_at_risk"] = result["total_revenue_at_risk"].round(2)
    result["avg_p_cancel"] = result["avg_p_cancel"].round(3)
    result["avg_adr"] = result["avg_adr"].round(2)
    result["pct_high_risk"] = result["pct_high_risk"].round(1)

    logger.info(
        "Segment risk breakdown by '%s': top segment='%s', risk=€%.0f",
        segment_col,
        result.iloc[0][segment_col],
        result.iloc[0]["total_revenue_at_risk"],
    )

    return result


# Expected Loss Table (Ranked Booking List)

def expected_loss_table(
    bookings_df: pd.DataFrame,
    cancel_proba: np.ndarray,
    adr_predictions: np.ndarray = None,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Produce a ranked table of the top-N highest revenue-at-risk bookings.

    This is the primary output for the hotel revenue manager's dashboard.
    They see which specific bookings to prioritise for intervention.

    Returns columns:
      rank, arrival_date, distribution_channel, customer_type,
      p_cancel, adr_used, total_nights, revenue_at_risk, risk_level, action
    """
    detail = daily_revenue_at_risk(bookings_df, cancel_proba, adr_predictions)

    action_map = {
        "HIGH": " Send retention offer NOW",
        "MEDIUM": " Monitor — contact if no contact in 7 days",
        "LOW": " No action needed",
    }
    detail["action"] = detail["risk_level"].map(action_map)

    output_cols = [
        c for c in [
            "arrival_date", "hotel", "distribution_channel", "customer_type",
            "deposit_type", "lead_time", "p_cancel", "adr_used",
            "total_nights", "revenue_at_risk", "risk_level", "action",
        ] if c in detail.columns
    ]

    result = detail[output_cols].head(top_n).copy()
    result.index = range(1, len(result) + 1)
    result.index.name = "rank"
    return result


# Property-Level Revenue Summary

def property_revenue_summary(
    bookings_df: pd.DataFrame,
    cancel_proba: np.ndarray,
    adr_predictions: np.ndarray = None,
    date_range: Tuple = None,
) -> Dict:
    """
    Compute high-level KPIs for the property dashboard.

    Returns
    -------
    dict:
      total_bookings, total_revenue_at_risk, avg_cancel_probability,
      n_high_risk, n_medium_risk, n_low_risk,
      expected_cancellations, expected_revenue_loss,
      top_risk_date, top_risk_channel
    """
    detail = daily_revenue_at_risk(bookings_df, cancel_proba, adr_predictions)

    if date_range is not None:
        start, end = date_range
        if "arrival_date" in detail.columns:
            mask = (detail["arrival_date"] >= start) & (detail["arrival_date"] <= end)
            detail = detail[mask]

    n = len(detail)
    if n == 0:
        return {}

    summary = {
        "total_bookings": n,
        "total_revenue_at_risk": round(float(detail["revenue_at_risk"].sum()), 2),
        "avg_cancel_probability": round(float(detail["p_cancel"].mean()), 3),
        "n_high_risk": int((detail["risk_level"] == "HIGH").sum()),
        "n_medium_risk": int((detail["risk_level"] == "MEDIUM").sum()),
        "n_low_risk": int((detail["risk_level"] == "LOW").sum()),
        "expected_cancellations": round(float(detail["p_cancel"].sum()), 1),
        "expected_revenue_loss": round(float(detail["revenue_at_risk"].sum()), 2),
        "median_cancel_prob": round(float(detail["p_cancel"].median()), 3),
        "p90_revenue_at_risk": round(float(detail["revenue_at_risk"].quantile(0.9)), 2),
    }

    if "arrival_date" in detail.columns:
        by_date = detail.groupby("arrival_date")["revenue_at_risk"].sum()
        summary["top_risk_date"] = str(by_date.idxmax())
        summary["top_risk_date_eur"] = round(float(by_date.max()), 2)

    if "distribution_channel" in detail.columns:
        by_channel = detail.groupby("distribution_channel")["revenue_at_risk"].sum()
        summary["top_risk_channel"] = str(by_channel.idxmax())
        summary["top_risk_channel_eur"] = round(float(by_channel.max()), 2)

    return summary

# Intervention ROI Calculator

def intervention_roi(
    n_interventions: int,
    catch_rate: float,
    avg_adr: float,
    avg_nights: float,
    intervention_cost_per_booking: float = 15.0,
    retention_success_rate: float = 0.25,
) -> Dict:
    """
    Calculate ROI of a proactive retention campaign.

    Formula:
      Revenue saved = n_interventions × catch_rate × retention_success_rate × avg_adr × avg_nights
      Campaign cost = n_interventions × intervention_cost_per_booking
      Net ROI = revenue_saved / campaign_cost

    Parameters
    ----------
    n_interventions         : Number of high-risk bookings to contact
    catch_rate              : Fraction of actual cancellations caught by the model
    avg_adr                 : Mean ADR for the booking cohort
    avg_nights              : Mean total nights
    intervention_cost       : Cost per outreach action (email + agent time)
    retention_success_rate  : What fraction of contacted guests stay? (25% is realistic)
    """
    revenue_saved = (
        n_interventions * catch_rate * retention_success_rate * avg_adr * avg_nights
    )
    campaign_cost = n_interventions * intervention_cost_per_booking
    net_value = revenue_saved - campaign_cost
    roi = (revenue_saved / max(campaign_cost, 1)) if campaign_cost > 0 else 0.0

    return {
        "n_interventions": n_interventions,
        "revenue_saved_eur": round(revenue_saved, 2),
        "campaign_cost_eur": round(campaign_cost, 2),
        "net_value_eur": round(net_value, 2),
        "roi_ratio": round(roi, 2),
        "roi_pct": round((roi - 1) * 100, 1),
        "break_even_retention_rate": round(
            campaign_cost / max(n_interventions * catch_rate * avg_adr * avg_nights, 1), 3
        ),
    }