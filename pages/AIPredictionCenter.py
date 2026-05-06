import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Baku Hotel AI Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

THRESHOLD = 0.35
RAW_COLS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
    "agent", "days_in_waiting_list", "adr", "required_car_parking_spaces",
    "total_of_special_requests", "total_nights", "total_guests", "arrival_date_month_num",
    "is_holiday", "net_bookings", "room_mismatch", "is_new_guest", "deposit_given",
    "reservation_year", "reservation_month", "reservation_day", "reservation_day_of_week",
    "is_family", "price_per_person", "request_density", "revenue_potential", "actual_revenue",
]
ENCODED_COLS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "agent",
    "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "total_nights", "total_guests", "arrival_date_month_num", "is_holiday", "net_bookings",
    "room_mismatch", "is_new_guest", "deposit_given", "reservation_year", "reservation_month",
    "reservation_day", "reservation_day_of_week", "is_family", "price_per_person",
    "request_density", "revenue_potential", "actual_revenue", "meal_HB", "meal_Rare", "meal_SC",
    "market_segment_Direct", "market_segment_Groups", "market_segment_Offline TA/TO",
    "market_segment_Online TA", "market_segment_Rare", "distribution_channel_Direct",
    "distribution_channel_Rare", "distribution_channel_TA/TO", "reserved_room_type_C",
    "reserved_room_type_D", "reserved_room_type_E", "reserved_room_type_F", "reserved_room_type_G",
    "reserved_room_type_Rare", "assigned_room_type_B", "assigned_room_type_C", "assigned_room_type_D",
    "assigned_room_type_E", "assigned_room_type_F", "assigned_room_type_G", "assigned_room_type_Rare",
    "deposit_type_Non Refund", "deposit_type_Rare", "customer_type_Rare", "customer_type_Transient",
    "customer_type_Transient-Party", "arrival_season_Spring", "arrival_season_Summer",
    "arrival_season_Winter", "revenue_category_Mid Revenue", "revenue_category_High Revenue",
    "res_year", "res_month", "res_day", "country_BRA", "country_DEU", "country_ESP", "country_FRA",
    "country_GBR", "country_IRL", "country_ITA", "country_Other", "country_PRT", "country_Rare",
    "waitlist_cat_Rare", "booking_lead_time_bucket_Short-term",
]


@st.cache_data
def load_assets_placeholder():
    """
    Placeholder section:
    Replace this with your own loading logic for:
      - df (cleaned data)
      - new_df (encoded data)
      - xgb_clf (classifier model)
      - xgb_reg (regressor model)
    """
    # Keep placeholders, but auto-load local CSVs if present.
    df = pd.DataFrame(columns=RAW_COLS)
    for candidate in ["hotel_bookings_cleaned.csv", "hotel_bookings.csv"]:
        try:
            loaded = pd.read_csv(candidate)
            if not loaded.empty:
                df = loaded.copy()
                break
        except Exception:
            continue

    if not df.empty:
        if "children" in df.columns:
            df["children"] = df["children"].fillna(0)
        if "total_nights" not in df.columns and {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(df.columns):
            df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
        if "total_guests" not in df.columns and {"adults", "children", "babies"}.issubset(df.columns):
            df["total_guests"] = df["adults"] + df["children"] + df["babies"]
        if "request_density" not in df.columns and {"total_of_special_requests", "total_nights"}.issubset(df.columns):
            df["request_density"] = df["total_of_special_requests"] / df["total_nights"].replace(0, 1)
        if "room_mismatch" not in df.columns and {"reserved_room_type", "assigned_room_type"}.issubset(df.columns):
            df["room_mismatch"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

    new_df = pd.DataFrame(columns=ENCODED_COLS)
    if not df.empty:
        # Lightweight encoded frame placeholder aligned to your notebook columns.
        encoded = pd.get_dummies(
            df,
            columns=[
                c for c in [
                    "meal", "market_segment", "distribution_channel", "reserved_room_type",
                    "assigned_room_type", "deposit_type", "customer_type", "arrival_season", "country"
                ] if c in df.columns
            ],
            drop_first=False,
        )
        for col in ENCODED_COLS:
            if col not in encoded.columns:
                encoded[col] = 0
        new_df = encoded[ENCODED_COLS].copy()

    xgb_clf = None
    xgb_reg = None
    return df, new_df, xgb_clf, xgb_reg


df, new_df, xgb_clf, xgb_reg = load_assets_placeholder()

st.markdown(
    """
<style>
    .stApp { background-color: #0b1220; color: #e5e7eb; }
    section[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1f2937; }
    .hero {
        border: 1px solid #334155; border-radius: 16px; padding: 20px;
        background: linear-gradient(120deg, #0f172a 0%, #111827 60%, #1e293b 100%);
        margin-bottom: 16px;
    }
    .tag {
        display: inline-block; padding: 4px 10px; border: 1px solid #334155;
        border-radius: 999px; font-size: 12px; color: #93c5fd; margin-right: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1 style="margin:0;">🏨 Baku Hotel AI Dashboard</h1>
  <p style="margin:8px 0 12px 0; color:#9ca3af;">
    Otel rezervasiyası üçün ikili AI engine: Ləğv riski (Classification) + ADR qiymət təxmini (Regression)
  </p>
  <span class="tag">Threshold: 0.35</span>
  <span class="tag">F1: 0.71</span>
  <span class="tag">R²: 0.88</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎛️ Input Parametrlər")
    hotel = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    lead_time = st.slider("Lead Time (days)", 0, 730, 60)
    stays_weekend = st.slider("Weekend Nights", 0, 10, 1)
    stays_week = st.slider("Week Nights", 0, 20, 2)
    adults = st.number_input("Adults", min_value=1, max_value=10, value=2)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Groups"])
    deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"])
    customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
    special_requests = st.slider("Total Special Requests", 0, 5, 1)
    parking_spaces = st.selectbox("Parking Spaces", [0, 1, 2])
    predict_clicked = st.button("🚀 Proqnozu İşlət", use_container_width=True)


def build_input_row():
    total_nights = stays_weekend + stays_week
    total_guests = adults + children + babies
    row = {
        "hotel": 1 if hotel == "City Hotel" else 0,
        "lead_time": lead_time,
        "stays_in_weekend_nights": stays_weekend,
        "stays_in_week_nights": stays_week,
        "adults": adults,
        "children": children,
        "babies": babies,
        "total_of_special_requests": special_requests,
        "required_car_parking_spaces": parking_spaces,
        "total_nights": total_nights,
        "total_guests": total_guests,
        "is_family": 1 if (children + babies) > 0 else 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "booking_changes": 0,
        "days_in_waiting_list": 0,
        "is_holiday": 0,
        "room_mismatch": 0,
        "request_density": special_requests / max(total_nights, 1),
        "price_per_person": 0,
        "deposit_given": 1 if deposit_type == "Non Refund" else 0,
        "is_new_guest": 1,
        "net_bookings": 1,
        "adr": 0,
    }

    encoded_input = pd.DataFrame(np.zeros((1, len(ENCODED_COLS))), columns=ENCODED_COLS)
    for key, val in row.items():
        if key in encoded_input.columns:
            encoded_input.at[0, key] = val

    segment_col = f"market_segment_{market_segment}"
    customer_col = f"customer_type_{customer_type}"
    deposit_col = "deposit_type_Non Refund" if deposit_type == "Non Refund" else None
    if segment_col in encoded_input.columns:
        encoded_input.at[0, segment_col] = 1
    if customer_col in encoded_input.columns:
        encoded_input.at[0, customer_col] = 1
    if deposit_col and deposit_col in encoded_input.columns:
        encoded_input.at[0, deposit_col] = 1

    return row, encoded_input


tab1, tab2, tab3 = st.tabs(
    ["🔮 Prediction (AI Proqnoz)", "📊 Business Analytics (EDA)", "⚙️ Model Performance"]
)

with tab1:
    st.subheader("AI Proqnoz Mərkəzi")
    if predict_clicked:
        row, encoded_input = build_input_row()

        if xgb_clf is not None:
            prob_cancel = float(xgb_clf.predict_proba(encoded_input)[0][1])
        else:
            prob_cancel = min(0.95, 0.18 + (lead_time / 1000) + (0.08 if deposit_type == "No Deposit" else -0.05))

        cancel_flag = int(prob_cancel >= THRESHOLD)

        if "is_canceled" in encoded_input.columns:
            encoded_input.at[0, "is_canceled"] = cancel_flag
        if xgb_reg is not None:
            pred_adr = float(xgb_reg.predict(encoded_input)[0])
        else:
            pred_adr = 65 + (lead_time * 0.07) + (special_requests * 4.5) + (10 if hotel == "Resort Hotel" else 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Cancellation Probability", f"{prob_cancel:.1%}", "Threshold 35%")
        c2.metric("Predicted ADR", f"${pred_adr:.2f}")
        c3.metric("Total Nights", f"{row['total_nights']}")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_cancel * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444" if cancel_flag else "#22c55e"},
                    "steps": [
                        {"range": [0, 35], "color": "rgba(34,197,94,0.25)"},
                        {"range": [35, 100], "color": "rgba(239,68,68,0.25)"},
                    ],
                    "threshold": {"line": {"color": "#f59e0b", "width": 3}, "value": THRESHOLD * 100},
                },
                title={"text": "Cancellation Risk Meter"},
            )
        )
        gauge.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
        st.plotly_chart(gauge, use_container_width=True)

        if cancel_flag:
            st.error("🔴 Ləğv riski yüksəkdir. (Threshold 0.35 keçildi)")
        else:
            st.success("🟢 Ləğv riski aşağıdır. Rezervasiya stabil görünür.")

    else:
        st.info("Sidebar-dan inputları seçib proqnozu başlat.")

with tab2:
    st.subheader("Business Analytics (EDA)")
    if df.empty:
        st.warning("`df` placeholder vəziyyətindədir. Data yükləyən kimi qrafiklər avtomatik işləyəcək.")
    else:
        left, right = st.columns(2)
        with left:
            fig_lead = px.histogram(
                df,
                x="lead_time",
                color="is_canceled",
                nbins=50,
                barmode="overlay",
                title="Lead Time vs Cancellation",
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
            )
            fig_lead.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
            st.plotly_chart(fig_lead, use_container_width=True)

        with right:
            cancel_by_segment = df.groupby("market_segment", as_index=False)["is_canceled"].mean()
            fig_segment = px.bar(
                cancel_by_segment,
                x="market_segment",
                y="is_canceled",
                color="is_canceled",
                title="Market Segment üzrə Ləğv Payı",
                color_continuous_scale="Reds",
            )
            fig_segment.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
            st.plotly_chart(fig_segment, use_container_width=True)

        low1, low2 = st.columns(2)
        with low1:
            adr_hotel = df.groupby("hotel", as_index=False)["adr"].mean()
            fig_adr = px.bar(
                adr_hotel,
                x="hotel",
                y="adr",
                color="hotel",
                title="Hotel Type üzrə Ortalama ADR",
                color_discrete_sequence=["#38bdf8", "#f59e0b"],
            )
            fig_adr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
            st.plotly_chart(fig_adr, use_container_width=True)

        with low2:
            req_impact = df.groupby("total_of_special_requests", as_index=False)["is_canceled"].mean()
            fig_req = px.line(
                req_impact,
                x="total_of_special_requests",
                y="is_canceled",
                markers=True,
                title="Special Requests-in Ləğvə Təsiri",
            )
            fig_req.update_traces(line_color="#a78bfa", marker_color="#f43f5e")
            fig_req.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
            st.plotly_chart(fig_req, use_container_width=True)

with tab3:
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classifier F1", "0.71", "Strong")
    c2.metric("Classifier Accuracy", "0.84")
    c3.metric("Regressor R²", "0.88", "Excellent")
    c4.metric("Regressor RMSE", "21.50")

    perf_curve = pd.DataFrame(
        {
            "Model": ["XGBoost Classifier", "XGBoost Classifier", "XGBoost Regressor", "XGBoost Regressor"],
            "Metric": ["F1", "Accuracy", "R2", "RMSE"],
            "Score": [0.71, 0.84, 0.88, 0.215],
        }
    )
    fig_perf = px.bar(
        perf_curve,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Performance Overview (Normalized RMSE)",
        color_discrete_sequence=["#38bdf8", "#22c55e"],
    )
    fig_perf.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
    st.plotly_chart(fig_perf, use_container_width=True)

    f_left, f_right = st.columns(2)
    with f_left:
        st.markdown("#### Classifier Feature Importance")
        if xgb_clf is not None and hasattr(xgb_clf, "feature_importances_"):
            clf_imp = pd.DataFrame({"feature": ENCODED_COLS, "importance": xgb_clf.feature_importances_}).sort_values(
                "importance", ascending=False
            ).head(15)
        else:
            clf_imp = pd.DataFrame(
                {
                    "feature": ["lead_time", "deposit_type_Non Refund", "market_segment_Online TA", "total_of_special_requests", "room_mismatch"],
                    "importance": [0.28, 0.19, 0.14, 0.13, 0.09],
                }
            )
        fig_clf_imp = px.bar(clf_imp.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance")
        fig_clf_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        st.plotly_chart(fig_clf_imp, use_container_width=True)

    with f_right:
        st.markdown("#### Regressor Feature Importance")
        if xgb_reg is not None and hasattr(xgb_reg, "feature_importances_"):
            reg_imp = pd.DataFrame({"feature": ENCODED_COLS, "importance": xgb_reg.feature_importances_}).sort_values(
                "importance", ascending=False
            ).head(15)
        else:
            reg_imp = pd.DataFrame(
                {
                    "feature": ["lead_time", "market_segment_Groups", "total_nights", "is_holiday", "customer_type_Transient"],
                    "importance": [0.24, 0.18, 0.16, 0.12, 0.11],
                }
            )
        fig_reg_imp = px.bar(reg_imp.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance")
        fig_reg_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e5e7eb")
        st.plotly_chart(fig_reg_imp, use_container_width=True)