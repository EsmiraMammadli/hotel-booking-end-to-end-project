import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Time-Series Demand Trend", layout="wide")
st.title("Time-Series Demand Trend")
st.caption("Demand curves, moving averages, and STL decomposition by hotel type and market segment.")

# ─────────────────────────────────────────────
# Load & cache data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hotel_bookings.csv")

    # ── same cleaning your notebook does ──────
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])
    df = df.drop(columns=["company", "reservation_status"], errors="ignore")
    df["agent"] = df["agent"].fillna(0)
    df["country"] = df["country"].fillna(df["country"].mode()[0])
    df = df.dropna(subset=["children"])
    df = df.drop_duplicates()
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[df["adults"] > 0]
    df = df[(df["adr"] > 0) & (df["adr"] < 455)]
    df = df[df["adults"] <= 10]
    df = df[df["lead_time"] <= 600]

    # ── feature engineering ───────────────────
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["revenue_potential"] = df["adr"] * df["total_nights"]

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    df["arrival_date_month_num"] = df["arrival_date_month"].map(month_map)

    # ── build arrival_date (first of each month) ──
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month_num"].astype(str).str.zfill(2) + "-01"
    )
    return df


@st.cache_data
def build_monthly(df):
    monthly = (
        df.groupby(["arrival_date", "hotel", "market_segment"])
        .agg(
            bookings=("hotel", "count"),
            cancellations=("is_canceled", "sum"),
            avg_adr=("adr", "mean"),
            revenue_pot=("revenue_potential", "sum"),
        )
        .reset_index()
    )
    monthly["confirmed"] = monthly["bookings"] - monthly["cancellations"]
    monthly["cancel_rate"] = (monthly["cancellations"] / monthly["bookings"]).round(3)
    return monthly.sort_values("arrival_date").reset_index(drop=True)


df = load_data()
monthly = build_monthly(df)

# ─────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────
st.sidebar.header("Filters")

hotel_options = ["Both"] + sorted(df["hotel"].unique().tolist())
selected_hotel = st.sidebar.selectbox("Hotel type", hotel_options)

all_segments = sorted(df["market_segment"].unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Market segments", all_segments, default=all_segments
)

ma_window = st.sidebar.select_slider(
    "Moving average window (months)", options=[3, 6, 12], value=6
)

show_stl = st.sidebar.toggle("Show STL decomposition", value=True)

# ─────────────────────────────────────────────
# Filter data
# ─────────────────────────────────────────────
filtered = monthly.copy()
if selected_hotel != "Both":
    filtered = filtered[filtered["hotel"] == selected_hotel]
if selected_segments:
    filtered = filtered[filtered["market_segment"].isin(selected_segments)]

# ─────────────────────────────────────────────
# KPI row
# ─────────────────────────────────────────────
total_bookings  = int(filtered["bookings"].sum())
total_confirmed = int(filtered["confirmed"].sum())
avg_cancel_rate = filtered["cancel_rate"].mean()
peak_month      = (
    filtered.groupby("arrival_date")["confirmed"].sum().idxmax()
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total bookings",    f"{total_bookings:,}")
k2.metric("Confirmed bookings", f"{total_confirmed:,}")
k3.metric("Avg cancellation rate", f"{avg_cancel_rate:.1%}")
k4.metric("Peak month", peak_month.strftime("%b %Y"))

st.divider()

# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Demand curves",
    "Moving average",
    "STL decomposition",
    "Heatmaps",
])

# ── TAB 1 — Demand curves ─────────────────────
with tab1:
    st.subheader("Monthly confirmed bookings")

    col_a, col_b = st.columns(2)

    with col_a:
        hotel_monthly = (
            filtered.groupby(["arrival_date", "hotel"])["confirmed"]
            .sum().reset_index()
        )
        fig1 = px.line(
            hotel_monthly,
            x="arrival_date", y="confirmed", color="hotel",
            markers=True,
            labels={"arrival_date": "Arrival month", "confirmed": "Confirmed bookings", "hotel": "Hotel"},
            title="By hotel type",
        )
        fig1.update_layout(hovermode="x unified", height=380)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        seg_monthly = (
            filtered.groupby(["arrival_date", "market_segment"])["confirmed"]
            .sum().reset_index()
        )
        fig2 = px.line(
            seg_monthly,
            x="arrival_date", y="confirmed", color="market_segment",
            labels={"arrival_date": "Arrival month", "confirmed": "Confirmed bookings", "market_segment": "Segment"},
            title="By market segment",
        )
        fig2.update_layout(hovermode="x unified", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Revenue potential line
    st.subheader("Revenue potential trend")
    rev_monthly = (
        filtered.groupby("arrival_date")["revenue_pot"].sum().reset_index()
    )
    fig_rev = px.area(
        rev_monthly,
        x="arrival_date", y="revenue_pot",
        labels={"arrival_date": "Arrival month", "revenue_pot": "Revenue potential (€)"},
    )
    fig_rev.update_layout(hovermode="x unified", height=300)
    st.plotly_chart(fig_rev, use_container_width=True)


# ── TAB 2 — Moving average ────────────────────
with tab2:
    st.subheader(f"Booking demand with {ma_window}-month moving average")

    total_monthly = (
        filtered.groupby("arrival_date")["confirmed"]
        .sum().reset_index()
        .sort_values("arrival_date")
    )
    total_monthly["ma"] = (
        total_monthly["confirmed"].rolling(ma_window, min_periods=1).mean().round()
    )
    total_monthly["ma_3"]  = total_monthly["confirmed"].rolling(3,  min_periods=1).mean().round()
    total_monthly["ma_12"] = total_monthly["confirmed"].rolling(12, min_periods=1).mean().round()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=total_monthly["arrival_date"], y=total_monthly["confirmed"],
        mode="lines", name="Actual", opacity=0.35, line=dict(width=1, color="#888780"),
    ))
    fig_ma.add_trace(go.Scatter(
        x=total_monthly["arrival_date"], y=total_monthly["ma_3"],
        mode="lines", name="3-month MA", line=dict(width=1.5, dash="dot", color="#85B7EB"),
    ))
    fig_ma.add_trace(go.Scatter(
        x=total_monthly["arrival_date"], y=total_monthly["ma"],
        mode="lines", name=f"{ma_window}-month MA", line=dict(width=2.5, color="#378ADD"),
    ))
    fig_ma.add_trace(go.Scatter(
        x=total_monthly["arrival_date"], y=total_monthly["ma_12"],
        mode="lines", name="12-month MA", line=dict(width=1.5, dash="dash", color="#1D9E75"),
    ))
    fig_ma.update_layout(hovermode="x unified", height=400,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_ma, use_container_width=True)

    with st.expander("How to read moving averages"):
        st.markdown("""
- **3-month MA** — short-term trend, reactive to sudden changes.
- **6-month MA** — balances noise vs responsiveness; good for seasonal shape.
- **12-month MA** — long-run trend only; seasonal peaks are fully smoothed out.
- Where the 3-month MA crosses above the 12-month MA is a demand uptick signal.
        """)


# ── TAB 3 — STL decomposition ─────────────────
with tab3:
    st.subheader("STL decomposition")
    st.caption("Separates your demand series into Trend + Seasonal + Residual components.")

    if not show_stl:
        st.info("Enable 'Show STL decomposition' in the sidebar to view this tab.")
    else:
        decomp_by = st.radio("Decompose by", ["Overall", "Hotel type"], horizontal=True)

        def run_stl(series: pd.Series, label: str):
            series = series.sort_index().dropna()
            if len(series) < 24:
                st.warning(f"'{label}' has fewer than 24 months — STL needs at least 2 full cycles. Try a wider filter.")
                return
            stl_result = STL(series, period=12, robust=True).fit()

            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                vertical_spacing=0.07,
            )
            colors = ["#378ADD", "#1D9E75", "#EF9F27", "#D85A30"]
            components = [series.values, stl_result.trend, stl_result.seasonal, stl_result.resid]
            for i, (comp, color) in enumerate(zip(components, colors), start=1):
                fig.add_trace(go.Scatter(
                    x=series.index, y=comp,
                    mode="lines", line=dict(width=1.5, color=color),
                    showlegend=False,
                ), row=i, col=1)
                if i in (3, 4):
                    fig.add_hline(y=0, line_width=0.5, line_dash="dash",
                                  line_color="gray", row=i, col=1)

            fig.update_layout(height=600, title_text=f"STL decomposition — {label}",
                               hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Strength metrics
            var_resid    = np.var(stl_result.resid)
            var_seas_res = np.var(stl_result.seasonal + stl_result.resid)
            var_trnd_res = np.var(stl_result.trend    + stl_result.resid)
            f_s = max(0, 1 - var_resid / var_seas_res) if var_seas_res > 0 else 0
            f_t = max(0, 1 - var_resid / var_trnd_res) if var_trnd_res > 0 else 0

            m1, m2 = st.columns(2)
            m1.metric("Seasonal strength", f"{f_s:.2f}",
                      help="0 = no seasonality, 1 = perfectly seasonal")
            m2.metric("Trend strength",    f"{f_t:.2f}",
                      help="0 = no trend, 1 = perfectly trending")

        if decomp_by == "Overall":
            ts = (
                filtered.groupby("arrival_date")["confirmed"]
                .sum()
            )
            run_stl(ts, "All hotels")
        else:
            for hotel_name in filtered["hotel"].unique():
                st.markdown(f"##### {hotel_name}")
                ts = (
                    filtered[filtered["hotel"] == hotel_name]
                    .groupby("arrival_date")["confirmed"]
                    .sum()
                )
                run_stl(ts, hotel_name)
                st.divider()


# ── TAB 4 — Heatmaps ──────────────────────────
with tab4:
    st.subheader("Year × month heatmaps")

    heat_src = df.copy()
    if selected_hotel != "Both":
        heat_src = heat_src[heat_src["hotel"] == selected_hotel]
    if selected_segments:
        heat_src = heat_src[heat_src["market_segment"].isin(selected_segments)]

    month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        pivot_bookings = (
            heat_src.groupby(["arrival_date_year", "arrival_date_month_num"])
            .agg(confirmed=("is_canceled", lambda x: (x == 0).sum()))
            .reset_index()
            .pivot(index="arrival_date_year", columns="arrival_date_month_num", values="confirmed")
        )
        pivot_bookings.columns = [month_labels[c] for c in pivot_bookings.columns]
        fig_h1 = px.imshow(
            pivot_bookings, color_continuous_scale="Blues",
            labels=dict(x="Month", y="Year", color="Confirmed"),
            title="Confirmed bookings", aspect="auto",
        )
        fig_h1.update_layout(height=320)
        st.plotly_chart(fig_h1, use_container_width=True)

    with col_h2:
        pivot_adr = (
            heat_src.groupby(["arrival_date_year", "arrival_date_month_num"])["adr"]
            .mean().round(1)
            .reset_index()
            .pivot(index="arrival_date_year", columns="arrival_date_month_num", values="adr")
        )
        pivot_adr.columns = [month_labels[c] for c in pivot_adr.columns]
        fig_h2 = px.imshow(
            pivot_adr, color_continuous_scale="RdYlGn",
            labels=dict(x="Month", y="Year", color="Avg ADR (€)"),
            title="Average ADR (€)", aspect="auto",
        )
        fig_h2.update_layout(height=320)
        st.plotly_chart(fig_h2, use_container_width=True)

    # Cancel rate heatmap (full width)
    st.markdown("##### Cancellation rate")
    pivot_cancel = (
        heat_src.groupby(["arrival_date_year", "arrival_date_month_num"])
        .agg(cancel_rate=("is_canceled", "mean"))
        .reset_index()
        .assign(cancel_rate=lambda d: d["cancel_rate"].round(3))
        .pivot(index="arrival_date_year", columns="arrival_date_month_num", values="cancel_rate")
    )
    pivot_cancel.columns = [month_labels[c] for c in pivot_cancel.columns]
    fig_h3 = px.imshow(
        pivot_cancel, color_continuous_scale="OrRd",
        labels=dict(x="Month", y="Year", color="Cancel rate"),
        title="Cancellation rate by year & month", aspect="auto",
        zmin=0, zmax=1,
    )
    fig_h3.update_layout(height=280)
    st.plotly_chart(fig_h3, use_container_width=True)