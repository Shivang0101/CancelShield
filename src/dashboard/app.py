"""
src/dashboard/app.py
=====================
CancelShield Plotly Dash Dashboard
Three-tab interactive hotel analytics interface calling the FastAPI backend.

Tab 1 — Booking Risk Analyser    : Input booking details → cancellation risk + SHAP
Tab 2 — Property Overview         : Date-range KPIs + risk heatmap + ranked table
Tab 3 — Overbooking Planner       : Date + risk tolerance → buffer recommendation

Runs at: http://localhost:8050
Calls:   http://api:8000 (or localhost:8000 in dev)
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yaml
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.exceptions import PreventUpdate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPES = ["Transient", "Contract", "Group", "Transient-Party"]
CHANNELS = ["TA/TO", "Direct", "Corporate", "GDS", "Complementary", "Undefined"]
ROOM_TYPES = list("ABCDEFGH")
MEALS = ["BB", "HB", "FB", "SC", "Undefined"]

# Colour palette
RISK_COLORS = {"HIGH": "#F44336", "MEDIUM": "#FF9800", "LOW": "#4CAF50"}
BRAND_BLUE = "#1565C0"
BRAND_LIGHT = "#E3F2FD"

# App Initialisation

app = dash.Dash(
    __name__,
    title="CancelShield — Hotel Intelligence",
    update_title="Loading...",
    suppress_callback_exceptions=True,
)
server = app.server  

def _kpi_card(title: str, value: str, subtitle: str = "", color: str = BRAND_BLUE):
    return html.Div(
        style={
            "backgroundColor": "white",
            "borderRadius": "8px",
            "padding": "20px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.12)",
            "borderTop": f"4px solid {color}",
            "minWidth": "200px",
            "flex": "1",
        },
        children=[
            html.P(title, style={"color": "#757575", "fontSize": "13px", "margin": "0 0 8px"}),
            html.H2(value, style={"color": color, "margin": "0", "fontSize": "28px"}),
            html.P(subtitle, style={"color": "#9E9E9E", "fontSize": "12px", "margin": "4px 0 0"}),
        ],
    )


def _section_header(text: str):
    return html.H3(
        text,
        style={
            "color": BRAND_BLUE,
            "borderBottom": f"2px solid {BRAND_BLUE}",
            "paddingBottom": "8px",
            "marginTop": "32px",
        },
    )

# Layout: Top Header
header = html.Div(
    style={
        "backgroundColor": BRAND_BLUE,
        "padding": "16px 32px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
    },
    children=[
        html.Div([
            html.H1(" CancelShield", style={"color": "white", "margin": "0", "fontSize": "24px"}),
            html.P("Booking Cancellation Intelligence & Revenue Protection",
                   style={"color": "#90CAF9", "margin": "2px 0 0", "fontSize": "13px"}),
        ]),
        html.Div(
            id="api-status",
            style={"color": "white", "fontSize": "13px"},
            children="Checking API...",
        ),
    ],
)

# Tab 1: Booking Risk Analyser

tab1_layout = html.Div(
    style={"padding": "24px"},
    children=[
        _section_header(" Booking Details"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "16px"},
            children=[
                html.Div([html.Label("Hotel"), dcc.Dropdown(
                    ["City Hotel", "Resort Hotel"], "City Hotel", id="t1-hotel", clearable=False)]),
                html.Div([html.Label("Arrival Month"), dcc.Dropdown(
                    MONTHS, "August", id="t1-month", clearable=False)]),
                html.Div([html.Label("Lead Time (days)"), dcc.Input(
                    id="t1-lead-time", type="number", value=30, min=0, max=737,
                    style={"width": "100%", "padding": "6px"})]),
                html.Div([html.Label("Deposit Type"), dcc.Dropdown(
                    DEPOSIT_TYPES, "No Deposit", id="t1-deposit", clearable=False)]),
                html.Div([html.Label("Distribution Channel"), dcc.Dropdown(
                    CHANNELS, "TA/TO", id="t1-channel", clearable=False)]),
                html.Div([html.Label("Customer Type"), dcc.Dropdown(
                    CUSTOMER_TYPES, "Transient", id="t1-customer-type", clearable=False)]),
                html.Div([html.Label("Reserved Room"), dcc.Dropdown(
                    ROOM_TYPES, "A", id="t1-room-type", clearable=False)]),
                html.Div([html.Label("Meal Plan"), dcc.Dropdown(
                    MEALS, "BB", id="t1-meal", clearable=False)]),
                html.Div([html.Label("Weekend Nights"), dcc.Input(
                    id="t1-weekend-nights", type="number", value=0, min=0, max=20,
                    style={"width": "100%", "padding": "6px"})]),
                html.Div([html.Label("Week Nights"), dcc.Input(
                    id="t1-week-nights", type="number", value=2, min=0, max=20,
                    style={"width": "100%", "padding": "6px"})]),
                html.Div([html.Label("Adults"), dcc.Input(
                    id="t1-adults", type="number", value=2, min=1, max=10,
                    style={"width": "100%", "padding": "6px"})]),
                html.Div([html.Label("Special Requests"), dcc.Slider(
                    0, 5, 1, value=0, marks={i: str(i) for i in range(6)},
                    id="t1-special-requests", tooltip={"placement": "bottom"})]),
                html.Div([html.Label("Previous Cancellations"), dcc.Slider(
                    0, 5, 1, value=0, marks={i: str(i) for i in range(6)},
                    id="t1-prev-cancellations", tooltip={"placement": "bottom"})]),
                html.Div([html.Label("Is Repeated Guest"), dcc.RadioItems(
                    [{"label": "  Yes", "value": 1}, {"label": "  No", "value": 0}],
                    value=0, id="t1-repeated-guest", inline=True)]),
            ],
        ),

        html.Button(
            " Analyse Booking",
            id="t1-predict-btn",
            n_clicks=0,
            style={
                "backgroundColor": BRAND_BLUE, "color": "white",
                "border": "none", "padding": "12px 32px",
                "borderRadius": "6px", "cursor": "pointer",
                "fontSize": "15px", "marginTop": "24px",
                "fontWeight": "bold",
            },
        ),

        # Results area
        html.Div(id="t1-results", style={"marginTop": "32px"}),
        dcc.Loading(id="t1-loading", children=[html.Div(id="t1-loading-output")], type="circle"),
    ],
)


# Tab 2: Property Overview

tab2_layout = html.Div(
    style={"padding": "24px"},
    children=[
        _section_header(" Property Overview"),
        html.P(
            "This tab simulates a batch of bookings for demonstration. "
            "In production, it calls the backend for each booking in the DB.",
            style={"color": "#757575"},
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "24px", "marginBottom": "16px"},
            children=[
                html.Div([
                    html.Label("Date Range"),
                    dcc.DatePickerRange(
                        id="t2-date-range",
                        start_date=date(2024, 8, 1),
                        end_date=date(2024, 8, 31),
                        display_format="YYYY-MM-DD",
                    ),
                ]),
                html.Div([
                    html.Label("Number of Bookings"),
                    dcc.Input(
                        id="t2-n-bookings", type="number", value=500,
                        min=10, max=2000,
                        style={"width": "100%", "padding": "8px", "fontSize": "15px"},
                    ),
                ]),
            ],
        ),

        html.Button(
            " Load Dashboard",
            id="t2-load-btn",
            n_clicks=0,
            style={
                "backgroundColor": BRAND_BLUE, "color": "white",
                "border": "none", "padding": "10px 24px",
                "borderRadius": "6px", "cursor": "pointer",
            },
        ),

        # KPI Row
        html.Div(id="t2-kpi-row", style={"display": "flex", "gap": "16px", "marginTop": "24px",
                                           "flexWrap": "wrap"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginTop": "24px"},
            children=[
                html.Div([
                    html.H4("Risk by Distribution Channel", style={"color": BRAND_BLUE}),
                    dcc.Graph(id="t2-channel-chart", style={"height": "350px"}),
                ]),
                html.Div([
                    html.H4("Cancellation Risk Heatmap (Month × Day)", style={"color": BRAND_BLUE}),
                    dcc.Graph(id="t2-heatmap", style={"height": "350px"}),
                ]),
            ],
        ),

        html.H4("Top Bookings by Revenue at Risk", style={"color": BRAND_BLUE, "marginTop": "32px"}),
        html.Div(id="t2-risk-table"),
        dcc.Loading(id="t2-loading", type="circle"),
    ],
)


# Tab 3: Overbooking Planner

tab3_layout = html.Div(
    style={"padding": "24px"},
    children=[
        _section_header(" Overbooking Planner"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "24px"},
            children=[
                html.Div([
                    html.Label("Arrival Date"),
                    dcc.DatePickerSingle(
                        id="t3-date",
                        date=date(2024, 8, 15),
                        display_format="YYYY-MM-DD",
                    ),
                ]),
                html.Div([
                    html.Label("Hotel Capacity (Rooms)"),
                    dcc.Input(
                        id="t3-capacity", type="number", value=200, min=10, max=5000,
                        style={"width": "100%", "padding": "8px", "fontSize": "16px"},
                    ),
                ]),
                html.Div([
                    html.Label("Current Bookings"),
                    dcc.Input(
                        id="t3-bookings", type="number", value=200, min=1, max=5000,
                        style={"width": "100%", "padding": "8px", "fontSize": "16px"},
                    ),
                ]),
            ],
        ),

        html.Div(
            style={"marginTop": "24px"},
            children=[
                html.Label(
                    "Risk Tolerance",
                    style={"fontWeight": "bold", "fontSize": "15px"},
                ),
                html.P(
                    "0 = Very Conservative (5× walk penalty) | 0.5 = Balanced (2× walk penalty) | 1 = Aggressive (1× walk penalty)",
                    style={"color": "#757575", "fontSize": "13px"},
                ),
                dcc.Slider(
                    0.0, 1.0, 0.1,
                    value=0.5,
                    marks={
                        0.0: {"label": "Conservative", "style": {"color": "#4CAF50"}},
                        0.5: {"label": "Balanced", "style": {"color": "#FF9800"}},
                        1.0: {"label": "Aggressive", "style": {"color": "#F44336"}},
                    },
                    id="t3-risk-tolerance",
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),

        html.Button(
            " Get Recommendation",
            id="t3-recommend-btn",
            n_clicks=0,
            style={
                "backgroundColor": BRAND_BLUE, "color": "white",
                "border": "none", "padding": "12px 32px",
                "borderRadius": "6px", "cursor": "pointer",
                "fontSize": "15px", "marginTop": "24px", "fontWeight": "bold",
            },
        ),

        html.Div(id="t3-results", style={"marginTop": "32px"}),
        dcc.Loading(id="t3-loading", type="circle"),
    ],
)


# Main Layout

app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "backgroundColor": "#F5F7FA", "minHeight": "100vh"},
    children=[
        header,
        dcc.Tabs(
            id="main-tabs",
            value="tab1",
            style={"backgroundColor": "white", "borderBottom": f"2px solid {BRAND_BLUE}"},
            children=[
                dcc.Tab(label=" Booking Risk Analyser", value="tab1",
                        style={"padding": "12px 24px"},
                        selected_style={"padding": "12px 24px", "borderTop": f"3px solid {BRAND_BLUE}", "fontWeight": "bold"}),
                dcc.Tab(label=" Property Overview", value="tab2",
                        style={"padding": "12px 24px"},
                        selected_style={"padding": "12px 24px", "borderTop": f"3px solid {BRAND_BLUE}", "fontWeight": "bold"}),
                dcc.Tab(label=" Overbooking Planner", value="tab3",
                        style={"padding": "12px 24px"},
                        selected_style={"padding": "12px 24px", "borderTop": f"3px solid {BRAND_BLUE}", "fontWeight": "bold"}),
            ],
        ),
        html.Div(id="tab-content", style={"backgroundColor": "#F5F7FA"}),
    ],
)


# Callbacks

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab1":
        return tab1_layout
    elif tab == "tab2":
        return tab2_layout
    elif tab == "tab3":
        return tab3_layout


@app.callback(Output("api-status", "children"), Input("main-tabs", "value"))
def check_api_status(_):
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            return " API Connected"
        return f" API Status: {r.status_code}"
    except Exception:
        return " API Offline"


# Tab 1 — Predict Cancellation
@app.callback(
    Output("t1-results", "children"),
    Input("t1-predict-btn", "n_clicks"),
    [
        State("t1-hotel", "value"),
        State("t1-month", "value"),
        State("t1-lead-time", "value"),
        State("t1-deposit", "value"),
        State("t1-channel", "value"),
        State("t1-customer-type", "value"),
        State("t1-room-type", "value"),
        State("t1-meal", "value"),
        State("t1-weekend-nights", "value"),
        State("t1-week-nights", "value"),
        State("t1-adults", "value"),
        State("t1-special-requests", "value"),
        State("t1-prev-cancellations", "value"),
        State("t1-repeated-guest", "value"),
    ],
    prevent_initial_call=True,
)
def predict_cancellation(
    n_clicks, hotel, month, lead_time, deposit, channel,
    customer_type, room_type, meal, weekend_nights, week_nights,
    adults, special_requests, prev_cancellations, is_repeated
):
    if not n_clicks:
        raise PreventUpdate

    payload = {
        "hotel": hotel or "City Hotel",
        "arrival_date_month": month or "August",
        "lead_time": int(lead_time or 30),
        "deposit_type": deposit or "No Deposit",
        "distribution_channel": channel or "TA/TO",
        "customer_type": customer_type or "Transient",
        "reserved_room_type": room_type or "A",
        "meal": meal or "BB",
        "stays_in_weekend_nights": int(weekend_nights or 0),
        "stays_in_week_nights": int(week_nights or 2),
        "adults": int(adults or 2),
        "children": 0, "babies": 0,
        "is_repeated_guest": int(is_repeated or 0),
        "previous_cancellations": int(prev_cancellations or 0),
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "adr": 0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": int(special_requests or 0),
        "arrival_date_year": 2024,
        "arrival_date_day_of_month": 15,
    }

    try:
        r = requests.post(f"{API_BASE}/predict-cancellation", json=payload, timeout=15)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        return html.Div(
            f" API Error: {e}",
            style={"color": "red", "padding": "16px", "backgroundColor": "#FFEBEE", "borderRadius": "8px"},
        )

    p = resp["cancel_probability"]
    risk = resp["risk_level"]
    rar = resp.get("revenue_at_risk_eur", 0)
    color = RISK_COLORS.get(risk, "#9E9E9E")

    # Gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(p * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Cancel Probability (%)", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#E8F5E9"},
                {"range": [30, 60], "color": "#FFF9C4"},
                {"range": [60, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": resp["threshold_used"] * 100,
            },
        },
    ))
    gauge.update_layout(height=300, margin={"t": 40, "b": 10, "l": 20, "r": 20})

    # SHAP bar chart
    shap_fig = go.Figure()
    if resp.get("top_shap_factors"):
        factors = resp["top_shap_factors"]
        shap_fig.add_trace(go.Bar(
            x=[f["importance"] for f in factors],
            y=[f["feature"] for f in factors],
            orientation="h",
            marker_color=[color] * len(factors),
        ))
        shap_fig.update_layout(
            title="Top Risk Factors",
            xaxis_title="Feature Importance",
            height=250,
            margin={"t": 40, "b": 10, "l": 10, "r": 10},
        )

    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                _kpi_card("Risk Level", risk, "Based on model threshold", color),
                _kpi_card("Cancel Probability", f"{p:.1%}", "Module 1 XGBoost", color),
                _kpi_card("Revenue at Risk", f"€{rar:,.0f}", "If cancelled, not rebooked", BRAND_BLUE),
                _kpi_card("Decision", "Cancel" if resp["cancel_prediction"] else "Keep",
                          f"Threshold: {resp['threshold_used']:.2f}", color),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginTop": "24px"},
            children=[
                dcc.Graph(figure=gauge),
                dcc.Graph(figure=shap_fig) if resp.get("top_shap_factors") else html.Div(),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": BRAND_LIGHT, "border": f"1px solid {BRAND_BLUE}",
                "borderRadius": "8px", "padding": "16px", "marginTop": "24px",
            },
            children=[
                html.H4("🤖 AI Explanation", style={"color": BRAND_BLUE, "marginTop": 0}),
                html.Pre(
                    resp.get("explanation_text", "No explanation available."),
                    style={"whiteSpace": "pre-wrap", "fontFamily": "inherit", "margin": 0},
                ),
            ],
        ),
    ])


# Tab 2 — Property Overview
@app.callback(
    [Output("t2-kpi-row", "children"),
     Output("t2-channel-chart", "figure"),
     Output("t2-heatmap", "figure"),
     Output("t2-risk-table", "children")],
    Input("t2-load-btn", "n_clicks"),
    [State("t2-date-range", "start_date"), 
     State("t2-date-range", "end_date"),
     State("t2-n-bookings", "value")],
    prevent_initial_call=True,
)
def load_property_overview(n_clicks, start_date, end_date,n_bookings):
    if not n_clicks:
        raise PreventUpdate

    # Simulate property-level data
    seed = hash(str(start_date) + str(end_date)) % (2**32)
    np.random.seed(seed)
    n = int(n_bookings or 500)
    channels = ["TA/TO", "Direct", "Corporate", "GDS", "Groups"]
    df = pd.DataFrame({
        "channel": np.random.choice(channels, n, p=[0.45, 0.25, 0.15, 0.1, 0.05]),
        "p_cancel": np.clip(np.random.beta(2, 3, n), 0.01, 0.99),
        "adr": np.random.normal(112, 30, n).clip(30, 400),
        "total_nights": np.random.randint(1, 8, n),
        "day_of_week": np.random.randint(0, 7, n),
        "month": np.random.randint(1, 13, n),
    })
    df["rar"] = df["p_cancel"] * df["adr"] * df["total_nights"]
    df["risk_level"] = pd.cut(df["p_cancel"], bins=[0, 0.3, 0.6, 1.0],
                               labels=["LOW", "MEDIUM", "HIGH"])

    # KPIs
    kpi_row = [
        _kpi_card("Total Bookings", str(n), "", BRAND_BLUE),
        _kpi_card("Expected Cancellations", f"{df['p_cancel'].sum():.0f}",
                  f"{df['p_cancel'].mean():.1%} avg rate", "#F44336"),
        _kpi_card("Total Revenue at Risk", f"€{df['rar'].sum():,.0f}", "All bookings", "#FF9800"),
        _kpi_card("High Risk Bookings", str((df["risk_level"] == "HIGH").sum()),
                  f"{(df['risk_level']=='HIGH').mean():.1%} of total", "#F44336"),
    ]

    # Channel chart
    channel_data = df.groupby("channel").agg(
        total_rar=("rar", "sum"),
        avg_cancel=("p_cancel", "mean"),
    ).reset_index()
    channel_fig = go.Figure([
        go.Bar(x=channel_data["channel"], y=channel_data["total_rar"],
               marker_color=BRAND_BLUE, name="Revenue at Risk (€)")
    ])
    channel_fig.update_layout(
        xaxis_title="Distribution Channel",
        yaxis_title="Total Revenue at Risk (€)",
        showlegend=False,
        plot_bgcolor="white",
        margin={"t": 20},
    )

    # Heatmap: day_of_week × month
    pivot = df.pivot_table(
        values="p_cancel", index="day_of_week", columns="month", aggfunc="mean"
    )
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [MONTHS[i - 1][:3] for i in pivot.columns]
    heat_fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="RdYlGn_r",
        colorbar={"title": "Avg P(cancel)"},
    ))
    heat_fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Day of Week",
        margin={"t": 20},
    )

    # Risk table
    top_bookings = df.nlargest(20, "rar")[
        ["channel", "p_cancel", "adr", "total_nights", "rar", "risk_level"]
    ].copy()
    top_bookings.columns = ["Channel", "P(Cancel)", "ADR (€)", "Nights", "Revenue at Risk (€)", "Risk"]
    top_bookings = top_bookings.round(2).reset_index(drop=True)
    table = dash_table.DataTable(
        data=top_bookings.to_dict("records"),
        columns=[{"name": c, "id": c} for c in top_bookings.columns],
        style_cell={"padding": "8px", "fontFamily": "inherit"},
        style_header={"backgroundColor": BRAND_BLUE, "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": '{Risk} = "HIGH"'}, "backgroundColor": "#FFEBEE"},
            {"if": {"filter_query": '{Risk} = "MEDIUM"'}, "backgroundColor": "#FFF9C4"},
            {"if": {"filter_query": '{Risk} = "LOW"'}, "backgroundColor": "#E8F5E9"},
        ],
        page_size=15,
        export_format="csv",
    )

    return kpi_row, channel_fig, heat_fig, table


# Tab 3 — Overbooking Planner
@app.callback(
    Output("t3-results", "children"),
    Input("t3-recommend-btn", "n_clicks"),
    [
        State("t3-date", "date"),
        State("t3-capacity", "value"),
        State("t3-bookings", "value"),
        State("t3-risk-tolerance", "value"),
    ],
    prevent_initial_call=True,
)
def get_overbooking_recommendation(n_clicks, arrival_date, capacity, bookings, risk_tolerance):
    if not n_clicks:
        raise PreventUpdate

    payload = {
        "arrival_date": str(arrival_date),
        "hotel_capacity": int(capacity or 200),
        "current_bookings": int(bookings or 200),
        "risk_tolerance": float(risk_tolerance if risk_tolerance is not None else 0.5),
    }

    try:
        r = requests.post(f"{API_BASE}/overbooking-recommendation", json=payload, timeout=15)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        return html.Div(f" API Error: {e}", style={"color": "red", "padding": "16px"})

    buffer = resp["recommended_overbooking_buffer"]
    walk_risk = resp["probability_of_walking_guest_pct"]
    extra_rev = resp["expected_extra_revenue_eur"]
    net_gain = resp["net_expected_gain_eur"]
    walk_color = "#F44336" if walk_risk > 30 else "#FF9800" if walk_risk > 15 else "#4CAF50"

    # Poisson distribution visualisation
    lam = resp["predicted_cancellations_mean"]
    std = resp["predicted_cancellations_std"]
    from scipy.stats import poisson as sp_poisson
    x_range = range(max(0, int(lam - 3.5 * std)), int(lam + 4 * std) + 1)
    pmf_vals = [sp_poisson(lam).pmf(k) for k in x_range]
    bar_colors = ["#4CAF50" if k <= buffer else "#F44336" for k in x_range]

    dist_fig = go.Figure()
    dist_fig.add_trace(go.Bar(
        x=list(x_range),
        y=pmf_vals,
        marker_color=bar_colors,
        name="P(total cancellations = k)",
    ))
    dist_fig.add_vline(
        x=lam, line_dash="dot", line_color="orange",
        annotation_text=f"Expected: {lam:.1f}", annotation_position="top right",
    )
    dist_fig.add_vline(
        x=buffer, line_dash="dash", line_color="navy",
        annotation_text=f"Buffer: {buffer}", annotation_position="top left",
    )
    dist_fig.update_layout(
        title=f"Cancellation Distribution — {arrival_date}",
        xaxis_title="Total Cancellations on This Date",
        yaxis_title="Probability",
        plot_bgcolor="white",
        showlegend=False,
        height=350,
    )

    # Sensitivity table
    sensitivity_df = pd.DataFrame(resp.get("sensitivity_table", []))

    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                _kpi_card("Recommended Buffer", str(buffer),
                          "Extra bookings to accept", BRAND_BLUE),
                _kpi_card("Accept Up To", str(int(bookings or 200) + buffer),
                          f"Capacity: {capacity}", BRAND_BLUE),
                _kpi_card("Expected Extra Revenue", f"€{extra_rev:,.0f}",
                          "If cancellations behave as predicted", "#4CAF50"),
                _kpi_card("Walk Risk", f"{walk_risk:.1f}%",
                          "P(need to walk a guest)", walk_color),
                _kpi_card("Net Expected Gain", f"€{net_gain:,.0f}",
                          "Revenue minus expected walk cost", "#4CAF50" if net_gain > 0 else "#F44336"),
            ],
        ),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginTop": "24px"},
            children=[
                dcc.Graph(figure=dist_fig),
                html.Div([
                    html.H4("Sensitivity Analysis (Risk Tolerance)", style={"color": BRAND_BLUE}),
                    dash_table.DataTable(
                        data=sensitivity_df.round(2).to_dict("records") if not sensitivity_df.empty else [],
                        columns=[{"name": c.replace("_", " ").title(), "id": c}
                                 for c in sensitivity_df.columns] if not sensitivity_df.empty else [],
                        style_cell={"padding": "6px 12px", "fontSize": "13px"},
                        style_header={"backgroundColor": BRAND_BLUE, "color": "white"},
                        style_data_conditional=[
                            {"if": {"row_index": list(sensitivity_df[
                                sensitivity_df["risk_tolerance"] == float(risk_tolerance or 0.5)
                            ].index)},
                             "backgroundColor": BRAND_LIGHT, "fontWeight": "bold"},
                        ] if not sensitivity_df.empty else [],
                        page_size=12,
                    ),
                ]),
            ],
        ), 
        html.Div(
            style={
                "backgroundColor": BRAND_LIGHT, "border": f"1px solid {BRAND_BLUE}",
                "borderRadius": "8px", "padding": "16px", "marginTop": "24px",
            },
            children=[
                html.H4(" Algorithm Details", style={"color": BRAND_BLUE, "marginTop": 0}),
                html.P(
                    f"λ = {lam:.1f} expected cancellations | "
                    f"C_walk / (C_walk + C_empty) = {resp['threshold_ratio']:.3f} threshold | "
                    f"Walk cost multiplier: {resp['walk_cost_multiplier']:.1f}× ADR",
                    style={"margin": "0", "color": "#424242"},
                ),
            ],
        ),
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)