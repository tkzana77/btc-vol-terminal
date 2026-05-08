import requests
import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from datetime import datetime, timezone
from scipy.stats import norm

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="BTC Volatility Terminal",
    layout="wide"
)
# =====================================================
# AUTO REFRESH
# =====================================================

st_autorefresh(
    interval=15 * 60 * 1000,
    key="btc_refresh"
)

st.title("BTC Volatility Terminal")

# =====================================================
# DATA DOWNLOAD
# =====================================================

url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"

params = {
    "currency": "BTC",
    "kind": "option"
}

response = requests.get(url, params=params)

data = response.json()

df = pd.DataFrame(data["result"])

# =====================================================
# BTC SPOT
# =====================================================

index_url = "https://www.deribit.com/api/v2/public/get_index_price"

index_params = {
    "index_name": "btc_usd"
}

index_response = requests.get(
    index_url,
    params=index_params
)

btc_spot = index_response.json()["result"]["index_price"]

# =====================================================
# TIME
# =====================================================

current_time = datetime.now(timezone.utc)

# =====================================================
# PARSER
# =====================================================

def parse_instrument(instrument_name):

    parts = instrument_name.split("-")

    underlying = parts[0]

    expiry_raw = parts[1]

    expiry_date = datetime.strptime(
        expiry_raw,
        "%d%b%y"
    )

    strike = float(parts[2])

    option_type = parts[3]

    return pd.Series([
        underlying,
        expiry_raw,
        expiry_date,
        strike,
        option_type
    ])

df[[
    "underlying",
    "expiry_raw",
    "expiry_date",
    "strike",
    "option_type"
]] = df["instrument_name"].apply(parse_instrument)

# =====================================================
# DTE
# =====================================================

df["dte"] = (
    df["expiry_date"] - current_time.replace(tzinfo=None)
).dt.days

# =====================================================
# STRIKE DISTANCE
# =====================================================

df["strike_distance"] = abs(
    df["strike"] - btc_spot
)

# =====================================================
# BLACK-SCHOLES DELTA
# =====================================================

def bs_delta(
    spot,
    strike,
    iv,
    dte,
    option_type
):

    T = max(dte / 365, 0.0001)

    sigma = iv / 100

    r = 0

    d1 = (
        np.log(spot / strike)
        + (r + sigma**2 / 2) * T
    ) / (sigma * np.sqrt(T))

    if option_type == "C":
        return norm.cdf(d1)

    else:
        return norm.cdf(d1) - 1

df["delta"] = df.apply(
    lambda row: bs_delta(
        btc_spot,
        row["strike"],
        row["mark_iv"],
        row["dte"],
        row["option_type"]
    ),
    axis=1
)

# =====================================================
# ATM OPTIONS
# =====================================================

atm_options = df.loc[
    df.groupby("expiry_date")["strike_distance"].idxmin()
]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Controls")

selected_expiry = st.sidebar.selectbox(
    "Select Expiry",
    sorted(df["expiry_date"].astype(str).unique())
)

# =====================================================
# METRICS
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "BTC Spot",
        f"{btc_spot:,.2f}"
    )

with col2:
    atm_iv = atm_options.iloc[0]["mark_iv"]

    st.metric(
        "Front ATM IV",
        f"{atm_iv:.2f}"
    )

# =====================================================
# TERM STRUCTURE
# =====================================================

fig_term = px.line(
    atm_options.sort_values("dte"),
    x="dte",
    y="mark_iv",
    title="ATM IV TERM STRUCTURE",
    markers=True
)

st.plotly_chart(
    fig_term,
    use_container_width=True
)

# =====================================================
# SMILE
# =====================================================

smile_df = df[
    df["expiry_date"].astype(str) == selected_expiry
]

fig_smile = px.line(
    smile_df.sort_values("strike"),
    x="strike",
    y="mark_iv",
    color="option_type",
    title=f"IV SMILE ({selected_expiry})",
    markers=True
)

st.plotly_chart(
    fig_smile,
    use_container_width=True
)
# =====================================================
# RR / SKEW DATA
# =====================================================

rr_data = []

for expiry in sorted(df["expiry_date"].unique()):

    expiry_df = df[
        df["expiry_date"] == expiry
    ].copy()

    if len(expiry_df) < 20:
        continue

    # -----------------------------
    # 25 DELTA CALL
    # -----------------------------

    call_df = expiry_df[
        expiry_df["option_type"] == "C"
    ].copy()

    call_df["delta_distance"] = abs(
        call_df["delta"] - 0.25
    )

    call_25 = call_df.loc[
        call_df["delta_distance"].idxmin()
    ]

    # -----------------------------
    # 25 DELTA PUT
    # -----------------------------

    put_df = expiry_df[
        expiry_df["option_type"] == "P"
    ].copy()

    put_df["delta_distance"] = abs(
        put_df["delta"] + 0.25
    )

    put_25 = put_df.loc[
        put_df["delta_distance"].idxmin()
    ]

    # -----------------------------
    # ATM IV
    # -----------------------------

    atm_row = atm_options[
        atm_options["expiry_date"] == expiry
    ].iloc[0]

    atm_iv = atm_row["mark_iv"]

    call_iv = call_25["mark_iv"]

    put_iv = put_25["mark_iv"]

    # -----------------------------
    # RR
    # -----------------------------

    rr = call_iv - put_iv

    # -----------------------------
    # SKEW
    # -----------------------------

    skew = (
        put_iv - call_iv
    ) / atm_iv

    rr_data.append({

        "expiry": str(expiry.date()),
        "dte": atm_row["dte"],
        "rr": rr,
        "skew": skew

    })

# =====================================================
# RR DATAFRAME
# =====================================================

rr_df = pd.DataFrame(rr_data)
# =====================================================
# HISTORICAL DATABASE
# =====================================================

rr_df["timestamp"] = datetime.utcnow()

history_columns = [
    "timestamp",
    "expiry",
    "dte",
    "rr",
    "skew"
]

history_df = rr_df[history_columns]

csv_file = "vol_history.csv"

# =====================================================
# SAVE CSV
# =====================================================

if os.path.exists(csv_file):

    history_df.to_csv(

        csv_file,

        mode="a",

        header=False,

        index=False
    )

else:

    history_df.to_csv(

        csv_file,

        index=False
    )

# =====================================================
# SELECT METRIC
# =====================================================

metric_choice = st.selectbox(

    "Select Volatility Metric",

    [
        "25Δ Risk Reversal",
        "25Δ Skew"
    ]
)

# =====================================================
# CHART
# =====================================================

if metric_choice == "25Δ Risk Reversal":

    fig_rr = px.line(

        rr_df.sort_values("dte"),

        x="dte",
        y="rr",

        markers=True,

        title="BTC 25Δ RISK REVERSAL"
    )

    st.plotly_chart(
        fig_rr,
        use_container_width=True
    )

else:

    fig_skew = px.line(

        rr_df.sort_values("dte"),

        x="dte",
        y="skew",

        markers=True,

        title="BTC 25Δ SKEW"
    )

    st.plotly_chart(
        fig_skew,
        use_container_width=True
    )