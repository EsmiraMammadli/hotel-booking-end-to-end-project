import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import mannwhitneyu, chi2_contingency

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Repeat Guest Loyalty Analysis", layout="wide")

REPEAT_COLOR   = "#1D9E75"
FIRSTTM_COLOR  = "#378ADD"
COLOR_MAP      = {"Repeat guest": REPEAT_COLOR, "First-time guest": FIRSTTM_COLOR}

# ─────────────────────────────────────────────────────────────
# Data loading — exact cleaning pipeline from the notebook
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(path: str = "hotel_bookings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # --- datetime ---
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    # --- drop leaky / unused columns ---
    df = df.drop(columns=["company", "reservation_status"], errors="ignore")

    # --- fill nulls ---
    df["agent"]   = df["agent"].fillna(0)
    df["country"] = df["country"].fillna(df["country"].mode()[0])
    df = df.dropna(subset=["children"])

    # --- dedup ---
    df = df.drop_duplicates()

    # --- remove zero-guest rows ---
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

    # --- outlier caps (from notebook) ---
    df = df[df["adults"] > 0]
    df = df[(df["adr"] > 0) & (df["adr"] < 455)]
    df = df[df["adults"] <= 10]
    df = df[df["lead_time"] <= 600]
    df = df[df["children"] <= 4]
    df = df[df["stays_in_weekend_nights"] <= 5]
    df = df[df["stays_in_week_nights"] <= 10]
    df = df[df["required_car_parking_spaces"] <= 3]

    # --- engineered features (from notebook) ---
    df["total_nights"]   = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"]   = df["adults"] + df["children"] + df["babies"]
    df["revenue_potential"] = df["adr"] * df["total_nights"]
    df["price_per_person"]  = df["adr"] / df["total_guests"].clip(lower=1)
    df["net_bookings"]      = df["previous_bookings_not_canceled"] - df["previous_cancellations"]
    df["room_mismatch"]     = (df["assigned_room_type"] != df["reserved_room_type"]).astype(int)
    df["is_new_guest"]      = (df["previous_bookings_not_canceled"] == 0).astype(int)
    df["deposit_given"]     = (df["deposit_type"] != "No Deposit").astype(int)
    df["is_family"]         = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["request_density"]   = df["total_of_special_requests"] / df["total_nights"].clip(lower=1)
    df["booking_lead_time_bucket"] = np.where(df["lead_time"] <= 30, "Short-term", "Long-term")

    month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                 "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    df["arrival_date_month_num"] = df["arrival_date_month"].map(month_map)

    def season(m):
        if m in [12, 1, 2]: return "Winter"
        if m in [3, 4, 5]:  return "Spring"
        if m in [6, 7, 8]:  return "Summer"
        return "Autumn"
    df["arrival_season"] = df["arrival_date_month_num"].apply(season)

    # --- readable guest type label ---
    df["guest_type"] = df["is_repeated_guest"].map({1: "Repeat guest", 0: "First-time guest"})

    return df


@st.cache_data
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    repeat  = df[df["is_repeated_guest"] == 1]
    firsttm = df[df["is_repeated_guest"] == 0]

    continuous = [
        ("adr",                       "Avg ADR (€)"),
        ("lead_time",                 "Avg lead time (days)"),
        ("total_of_special_requests", "Avg special requests"),
        ("total_nights",              "Avg total nights"),
        ("revenue_potential",         "Avg revenue potential (€)"),
        ("price_per_person",          "Avg price per person (€)"),
        ("room_mismatch",             "Room mismatch rate"),
    ]

    rows = []
    for col, label in continuous:
        stat, p = mannwhitneyu(
            repeat[col].dropna(), firsttm[col].dropna(), alternative="two-sided"
        )
        rows.append({
            "Metric":      label,
            "Repeat":      round(repeat[col].mean(), 2),
            "First-time":  round(firsttm[col].mean(), 2),
            "p-value":     round(p, 4),
            "Significant": "Yes ✓" if p < 0.05 else "No",
        })

    # cancellation: chi-square (binary outcome)
    ct = pd.crosstab(df["is_repeated_guest"], df["is_canceled"])
    chi2, p_cancel, _, _ = chi2_contingency(ct)
    rows.append({
        "Metric":      "Cancellation rate",
        "Repeat":      round(repeat["is_canceled"].mean(), 3),
        "First-time":  round(firsttm["is_canceled"].mean(), 3),
        "p-value":     round(p_cancel, 4),
        "Significant": "Yes ✓" if p_cancel < 0.05 else "No",
    })

    stats_df = pd.DataFrame(rows)
    stats_df["Diff %"] = (
        (stats_df["Repeat"] - stats_df["First-time"]) / stats_df["First-time"] * 100
    ).round(1)
    return stats_df


# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
df       = load_and_clean()
repeat   = df[df["is_repeated_guest"] == 1]
firsttm  = df[df["is_repeated_guest"] == 0]
stats_df = compute_stats(df)

# ─────────────────────────────────────────────────────────────
# Pre-compute headline numbers
# ─────────────────────────────────────────────────────────────
repeat_pct   = len(repeat) / len(df)
adr_lift     = (repeat["adr"].mean() / firsttm["adr"].mean() - 1) * 100
lead_diff    = repeat["lead_time"].mean() - firsttm["lead_time"].mean()
cancel_mult  = firsttm["is_canceled"].mean() / max(repeat["is_canceled"].mean(), 0.001)
req_diff     = repeat["total_of_special_requests"].mean() - firsttm["total_of_special_requests"].mean()
rev_lift     = (repeat["revenue_potential"].mean() / firsttm["revenue_potential"].mean() - 1) * 100

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("Repeat Guest Loyalty Analysis")
st.caption(
    "How repeat guests differ from first-timers across ADR, lead time, "
    "special requests, room preference, and cancellation behaviour."
)

# ─────────────────────────────────────────────────────────────
# Sidebar filter
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    hotel_filter = st.selectbox(
        "Hotel type",
        ["Both", "City Hotel", "Resort Hotel"]
    )
    season_filter = st.multiselect(
        "Arrival season",
        ["Spring", "Summer", "Autumn", "Winter"],
        default=["Spring", "Summer", "Autumn", "Winter"]
    )

fdf = df.copy()
if hotel_filter != "Both":
    fdf = fdf[fdf["hotel"] == hotel_filter]
if season_filter:
    fdf = fdf[fdf["arrival_season"].isin(season_filter)]

f_repeat  = fdf[fdf["is_repeated_guest"] == 1]
f_firsttm = fdf[fdf["is_repeated_guest"] == 0]

# ─────────────────────────────────────────────────────────────
# KPI row (filtered)
# ─────────────────────────────────────────────────────────────
f_repeat_pct  = len(f_repeat) / max(len(fdf), 1)
f_adr_lift    = (f_repeat["adr"].mean() / max(f_firsttm["adr"].mean(), 0.001) - 1) * 100
f_cancel_mult = (f_firsttm["is_canceled"].mean() /
                 max(f_repeat["is_canceled"].mean(), 0.001))
f_req_diff    = (f_repeat["total_of_special_requests"].mean() -
                 f_firsttm["total_of_special_requests"].mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total bookings",    f"{len(fdf):,}")
k2.metric("Repeat guest share", f"{f_repeat_pct:.1%}",
          help="% of bookings from guests who have stayed before")
k3.metric("ADR premium",
          f"{f_adr_lift:+.1f}%",
          delta=f"€{f_repeat['adr'].mean() - f_firsttm['adr'].mean():.0f} more per night")
k4.metric("Cancels fewer",
          f"{f_cancel_mult:.1f}×",
          delta=f"{f_firsttm['is_canceled'].mean():.0%} vs {f_repeat['is_canceled'].mean():.0%}")
k5.metric("Extra special requests",
          f"{f_req_diff:+.2f}",
          help="Average difference in number of special requests per booking")

st.divider()

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Distributions",
    "Room & segment",
    "Cancellation deep-dive",
    "Statistical tests",
    "Business insight",
])

# ── TAB 1 — Distributions ────────────────────────────────────
with tab1:
    st.subheader("Metric distributions: repeat vs first-time")

    metric_options = {
        "ADR (€)":                  "adr",
        "Lead time (days)":         "lead_time",
        "Special requests":         "total_of_special_requests",
        "Total nights":             "total_nights",
        "Revenue potential (€)":    "revenue_potential",
        "Price per person (€)":     "price_per_person",
        "Request density":          "request_density",
    }

    selected_label = st.selectbox("Choose metric", list(metric_options.keys()))
    col_name = metric_options[selected_label]

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig_hist = px.histogram(
            fdf, x=col_name, color="guest_type",
            barmode="overlay", opacity=0.65,
            marginal="box", nbins=60,
            color_discrete_map=COLOR_MAP,
            labels={col_name: selected_label, "guest_type": "Guest type"},
            title=f"{selected_label} — distribution comparison",
        )
        fig_hist.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        r_val = f_repeat[col_name].mean()
        f_val = f_firsttm[col_name].mean()
        diff  = r_val - f_val
        pct   = diff / max(f_val, 0.001) * 100

        st.markdown("##### Summary")
        st.metric("Repeat mean",     f"{r_val:.2f}")
        st.metric("First-time mean", f"{f_val:.2f}")
        st.metric("Difference",      f"{diff:+.2f}", delta=f"{pct:+.1f}%")

        stat, p = mannwhitneyu(
            f_repeat[col_name].dropna(),
            f_firsttm[col_name].dropna(),
            alternative="two-sided"
        )
        sig = p < 0.05
        if sig:
            st.success(f"Statistically significant (p = {p:.4f})")
        else:
            st.warning(f"Not significant (p = {p:.4f})")

    # Side-by-side violin for all metrics
    st.markdown("##### All metrics at a glance")
    violin_cols  = ["adr", "lead_time", "total_of_special_requests",
                    "total_nights", "revenue_potential"]
    violin_names = ["ADR", "Lead time", "Special req.", "Nights", "Revenue pot."]

    fig_vio = make_subplots(rows=1, cols=len(violin_cols),
                             subplot_titles=violin_names)
    for i, (col, name) in enumerate(zip(violin_cols, violin_names), start=1):
        for gtype, color in [(0, FIRSTTM_COLOR), (1, REPEAT_COLOR)]:
            sub = fdf[fdf["is_repeated_guest"] == gtype][col].dropna()
            label = "Repeat guest" if gtype == 1 else "First-time guest"
            fig_vio.add_trace(go.Violin(
                y=sub, name=label, legendgroup=label,
                showlegend=(i == 1),
                line_color=color, fillcolor=color,
                opacity=0.55, box_visible=True, meanline_visible=True,
                points=False,
            ), row=1, col=i)
    fig_vio.update_layout(height=380, violinmode="overlay",
                           legend=dict(orientation="h", yanchor="bottom", y=1.05))
    st.plotly_chart(fig_vio, use_container_width=True)


# ── TAB 2 — Room & segment ───────────────────────────────────
with tab2:
    st.subheader("Booking behaviour: room types and market segments")

    col_r, col_s = st.columns(2)

    with col_r:
        room_df = (
            fdf.groupby(["guest_type", "reserved_room_type"])
            .size().reset_index(name="count")
        )
        room_df["pct"] = (
            room_df.groupby("guest_type")["count"]
            .transform(lambda x: x / x.sum() * 100)
            .round(1)
        )
        fig_room = px.bar(
            room_df, x="reserved_room_type", y="pct",
            color="guest_type", barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"reserved_room_type": "Room type",
                    "pct": "Share (%)", "guest_type": ""},
            title="Room type preference (%)",
        )
        fig_room.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_room, use_container_width=True)

    with col_s:
        seg_df = (
            fdf.groupby(["guest_type", "market_segment"])
            .size().reset_index(name="count")
        )
        seg_df["pct"] = (
            seg_df.groupby("guest_type")["count"]
            .transform(lambda x: x / x.sum() * 100)
            .round(1)
        )
        fig_seg = px.bar(
            seg_df, x="market_segment", y="pct",
            color="guest_type", barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"market_segment": "Market segment",
                    "pct": "Share (%)", "guest_type": ""},
            title="Market segment distribution (%)",
        )
        fig_seg.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_seg, use_container_width=True)

    # Deposit type breakdown
    st.markdown("##### Deposit type behaviour")
    dep_df = (
        fdf.groupby(["guest_type", "deposit_type"])
        .size().reset_index(name="count")
    )
    dep_df["pct"] = (
        dep_df.groupby("guest_type")["count"]
        .transform(lambda x: x / x.sum() * 100)
        .round(1)
    )
    fig_dep = px.bar(
        dep_df, x="deposit_type", y="pct",
        color="guest_type", barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"deposit_type": "Deposit type",
                "pct": "Share (%)", "guest_type": ""},
        title="Deposit type preference (%)",
    )
    fig_dep.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02),
                          height=340)
    st.plotly_chart(fig_dep, use_container_width=True)

    # Meal plan
    st.markdown("##### Meal plan preference")
    meal_df = (
        fdf.groupby(["guest_type", "meal"])
        .size().reset_index(name="count")
    )
    meal_df["pct"] = (
        meal_df.groupby("guest_type")["count"]
        .transform(lambda x: x / x.sum() * 100)
        .round(1)
    )
    fig_meal = px.bar(
        meal_df, x="meal", y="pct",
        color="guest_type", barmode="group",
        color_discrete_map=COLOR_MAP,
        labels={"meal": "Meal plan", "pct": "Share (%)", "guest_type": ""},
        title="Meal plan preference (%)",
    )
    fig_meal.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           height=340)
    st.plotly_chart(fig_meal, use_container_width=True)


# ── TAB 3 — Cancellation deep-dive ───────────────────────────
with tab3:
    st.subheader("Cancellation behaviour")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        cancel_summary = (
            fdf.groupby("guest_type")["is_canceled"]
            .agg(["mean", "sum", "count"])
            .reset_index()
            .rename(columns={"mean": "rate", "sum": "cancelled", "count": "total"})
        )
        cancel_summary["rate_pct"] = (cancel_summary["rate"] * 100).round(1)

        fig_cancel = go.Figure()
        for _, row in cancel_summary.iterrows():
            color = REPEAT_COLOR if row["guest_type"] == "Repeat guest" else FIRSTTM_COLOR
            fig_cancel.add_trace(go.Bar(
                x=[row["guest_type"]], y=[row["rate_pct"]],
                marker_color=color, name=row["guest_type"],
                text=f"{row['rate_pct']}%", textposition="outside",
                showlegend=False,
            ))
        fig_cancel.update_layout(
            title="Cancellation rate (%)",
            yaxis_title="Cancellation rate (%)",
            height=360,
        )
        st.plotly_chart(fig_cancel, use_container_width=True)

    with col_c2:
        # Cancellation rate by lead time bucket
        cancel_lead = (
            fdf.groupby(["guest_type", "booking_lead_time_bucket"])["is_canceled"]
            .mean().reset_index()
        )
        cancel_lead["rate_pct"] = (cancel_lead["is_canceled"] * 100).round(1)

        fig_lead = px.bar(
            cancel_lead,
            x="booking_lead_time_bucket", y="rate_pct",
            color="guest_type", barmode="group",
            color_discrete_map=COLOR_MAP,
            labels={"booking_lead_time_bucket": "Lead time bucket",
                    "rate_pct": "Cancellation rate (%)", "guest_type": ""},
            title="Cancellation rate by lead time",
            text="rate_pct",
        )
        fig_lead.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_lead.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02),
                               height=360)
        st.plotly_chart(fig_lead, use_container_width=True)

    # Cancellation rate by season
    st.markdown("##### Cancellation rate by arrival season")
    cancel_season = (
        fdf.groupby(["guest_type", "arrival_season"])["is_canceled"]
        .mean().reset_index()
    )
    cancel_season["rate_pct"] = (cancel_season["is_canceled"] * 100).round(1)
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    cancel_season["arrival_season"] = pd.Categorical(
        cancel_season["arrival_season"], categories=season_order, ordered=True
    )
    cancel_season = cancel_season.sort_values("arrival_season")

    fig_season = px.line(
        cancel_season,
        x="arrival_season", y="rate_pct",
        color="guest_type", markers=True,
        color_discrete_map=COLOR_MAP,
        labels={"arrival_season": "Season",
                "rate_pct": "Cancellation rate (%)", "guest_type": ""},
        title="Cancellation rate across seasons",
    )
    fig_season.update_layout(height=340)
    st.plotly_chart(fig_season, use_container_width=True)


# ── TAB 4 — Statistical tests ─────────────────────────────────
with tab4:
    st.subheader("Mann-Whitney U & chi-square significance tests")
    st.caption(
        "ADR is non-normal (confirmed by Shapiro-Wilk in the notebook), "
        "so Mann-Whitney U is used for all continuous features. "
        "Cancellation rate uses chi-square (binary outcome)."
    )

    stats_display = compute_stats(fdf).copy()

    def highlight_sig(row):
        if row["Significant"] == "Yes ✓":
            return ["background-color: rgba(29,158,117,0.12)"] * len(row)
        return [""] * len(row)

    def color_diff(val):
        try:
            v = float(val)
            if v > 0:   return "color: #1D9E75; font-weight:500"
            elif v < 0: return "color: #D85A30; font-weight:500"
        except:
            pass
        return ""

    styled = (
        stats_display.style
        .apply(highlight_sig, axis=1)
        .map(color_diff, subset=["Diff %"])
        .format({"Repeat": "{:.3f}", "First-time": "{:.3f}",
                 "p-value": "{:.4f}", "Diff %": "{:+.1f}%"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("How to read this table"):
        st.markdown("""
- **Repeat / First-time** — mean value for each group (filtered by sidebar).
- **Diff %** — how much higher/lower the repeat guest mean is vs first-time. Green = repeat guests score higher.
- **p-value** — probability of observing this difference by chance if there were no real effect.
- **Significant** — p < 0.05 threshold. Highlighted rows indicate reliable differences.
- **Test used** — Mann-Whitney U for continuous metrics (non-parametric, robust to non-normality); Chi-square for cancellation rate (binary).
        """)


# ── TAB 5 — Business insight ─────────────────────────────────
with tab5:
    st.subheader("Business insight summary")

    f_adr_abs   = f_repeat["adr"].mean() - f_firsttm["adr"].mean()
    f_lead_diff = f_repeat["lead_time"].mean() - f_firsttm["lead_time"].mean()
    f_cancel_r  = f_repeat["is_canceled"].mean()
    f_cancel_f  = f_firsttm["is_canceled"].mean()
    f_cancel_x  = f_cancel_f / max(f_cancel_r, 0.001)
    f_rev_lift  = (f_repeat["revenue_potential"].mean() /
                   max(f_firsttm["revenue_potential"].mean(), 0.001) - 1) * 100
    f_req_d     = (f_repeat["total_of_special_requests"].mean() -
                   f_firsttm["total_of_special_requests"].mean())

    lead_word = "later" if f_lead_diff < 0 else "earlier"
    adr_word  = "more"  if f_adr_abs  > 0 else "less"

    st.info(
        f"**Repeat guests book {lead_word}** "
        f"({abs(f_lead_diff):.0f} days {'closer to' if f_lead_diff < 0 else 'further from'} arrival), "
        f"**pay {abs(f_adr_lift):.0f}% {adr_word}** (€{abs(f_adr_abs):.0f} ADR gap), "
        f"and **cancel {f_cancel_x:.1f}× less** than first-time guests — "
        f"making them the hotel's most valuable segment."
    )

    # Visual scorecard
    st.markdown("##### Scorecard")
    metrics_card = [
        ("Repeat share",         f"{f_repeat_pct:.1%}",          None),
        ("ADR premium",          f"+{f_adr_lift:.1f}%",           f"€{f_adr_abs:+.0f}/night"),
        ("Books later by",       f"{abs(f_lead_diff):.0f} days",  None),
        ("Cancels fewer",        f"{f_cancel_x:.1f}×",            f"{f_cancel_f:.0%} vs {f_cancel_r:.0%}"),
        ("Extra special reqs",   f"{f_req_d:+.2f}",               "per booking"),
        ("Revenue uplift",       f"+{f_rev_lift:.1f}%",           "revenue potential"),
    ]
    cols = st.columns(3)
    for i, (label, value, sub) in enumerate(metrics_card):
        with cols[i % 3]:
            st.metric(label, value, sub)

    st.markdown("##### Revenue opportunity")
    total_repeat_rev  = f_repeat["revenue_potential"].sum()
    total_firsttm_rev = f_firsttm["revenue_potential"].sum()
    total_rev         = total_repeat_rev + total_firsttm_rev

    fig_pie = px.pie(
        values=[total_repeat_rev, total_firsttm_rev],
        names=["Repeat guests", "First-time guests"],
        color_discrete_sequence=[REPEAT_COLOR, FIRSTTM_COLOR],
        title=f"Revenue potential split (total: €{total_rev:,.0f})",
        hole=0.45,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(height=360,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.1))
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("##### Strategic recommendation")
    st.markdown(f"""
Despite representing only **{f_repeat_pct:.1%}** of all bookings, repeat guests:

- Pay **{abs(f_adr_lift):.0f}% higher ADR** — no discounting needed to attract them
- Cancel **{f_cancel_x:.1f}× less often** — reducing overbooking risk and revenue leakage
- Make **{abs(f_req_d):.2f} more special requests per stay** — signalling higher engagement and spend intent
- Generate **{f_rev_lift:.1f}% more revenue potential** per booking

**Recommendation:** A targeted loyalty programme that converts even 1–2% of first-time guests into repeat visitors would deliver outsized revenue impact with minimal acquisition cost.
    """)