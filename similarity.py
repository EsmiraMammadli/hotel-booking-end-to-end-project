import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Bu funksiya faylları bir dəfə oxuyur və yaddaşda (cache) saxlayır
@st.cache_resource
def load_artefacts():
    scaler = joblib.load('similarity_scaler.pkl')
    sim_matrix = joblib.load('similarity_matrix.npy')
    sim_cols = joblib.load('similarity_cols.pkl')
    df = pd.read_parquet('bookings_clean.parquet')
    return scaler, sim_matrix, sim_cols, df

# Faylları dəyişənlərə mənimsədirik
scaler, sim_matrix, sim_cols, new_df = load_artefacts()

# 2. Similarity
def find_similar_bookings(booking_id, top_n=5, show_outcome=True):
    query_vec = sim_matrix[booking_id].reshape(1, -1)
    scores = cosine_similarity(query_vec, sim_matrix)[0]
    scores[booking_id] = -1
    top_indices = np.argsort(scores)[::-1][:top_n]

    cols_to_show = ['hotel', 'lead_time', 'adr', 'total_nights', 'total_guests']
    if show_outcome:
        cols_to_show.append('is_canceled')

    result = new_df.iloc[top_indices][cols_to_show].copy()
    result['similarity_score'] = scores[top_indices].round(4)
    return result


# 3. Saytın interfeysi
st.title("🏨 Bron Bənzərlik Sistemi")

booking_id = st.number_input("Bron ID-sini daxil edin:", min_value=0, max_value=len(new_df) - 1, value=0)
top_n = st.slider("Neçə bənzər bron göstərilsin?", 1, 20, 5)
show_outcome = st.checkbox("Ləğv statusunu göstər", value=True)

if st.button("Bənzərləri Tap"):
    st.write(f"### Seçilən Bron: {booking_id}")
    st.dataframe(new_df.iloc[[booking_id]])

    st.write("---")
    st.write("### Tapılan Ən Bənzər Bronlar:")
    recommendations = find_similar_bookings(booking_id, top_n, show_outcome)
    st.dataframe(recommendations)


def lookup_by_profile(profile_dict, top_n=5):
    # 1. Cədvəl yarat
    profile_df = pd.DataFrame([profile_dict])
    # 2. Kategorik dataları kodlaşdır (One-hot encode)
    profile_df = pd.get_dummies(profile_df)
    # 3. Sütunları modelin bildiyi sütunlarla eyniləşdir
    profile_df = profile_df.reindex(columns=sim_cols, fill_value=0)
    # 4. Tərəzidən keçir (Scale)
    profile_vec = scaler.transform(profile_df.fillna(0))
    # 5. Bənzərliyi hesabla
    scores = cosine_similarity(profile_vec, sim_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]

    # Nəticəni hazırla
    result = new_df.iloc[top_indices][
        ['hotel', 'lead_time', 'adr', 'total_nights', 'total_guests', 'is_canceled']].copy()
    result['similarity_score'] = scores[top_indices].round(4)
    result['outcome'] = result['is_canceled'].map({0: 'Completed ✅', 1: 'Cancelled ❌'})
    return result

st.write("---")
st.header("✨ Yeni Bron üçün Təxmin")

# İstifadəçidən məlumatları alırıq
col1, col2 = st.columns(2)
with col1:
    l_time = st.number_input("Lead Time (gün):", value=30)
    price = st.number_input("ADR (Qiymət):", value=95.0)
with col2:
    nights = st.number_input("Gecələmə sayı:", value=3)
    guests = st.number_input("Qonaq sayı:", value=2)

# Bu məlumatları funksiyanın istədiyi formata (dict) salırıq
custom_profile = {
    'lead_time': l_time,
    'adr': price,
    'total_nights': nights,
    'total_guests': guests,
    'hotel': 0, # Məsələn City Hotel
    'customer_type': 'Transient'
}

if st.button("Bu profilə uyğun keçmiş bronları tap"):
    res = lookup_by_profile(custom_profile)
    st.table(res)


import streamlit as st

st.title("Booking Similarity Engine")
st.caption("Find the 5 most similar historical bookings to any profile")

col1, col2 = st.columns(2)
with col1:
    lead_time = st.slider("Lead time (days)", 0, 600, 30)
    adr       = st.slider("ADR (€)", 0, 455, 95)
    nights    = st.slider("Total nights", 1, 15, 3)
with col2:
    guests    = st.slider("Total guests", 1, 10, 2)
    hotel     = st.selectbox("Hotel", ["City Hotel", "Resort Hotel"])
    ctype     = st.selectbox("Customer type",
                    ["Transient", "Contract", "Group", "Transient-Party"])

if st.button("Find similar bookings ↗"):
    profile = {
        'lead_time': lead_time,
        'adr': adr,
        'total_nights': nights,
        'total_guests': guests,
        'hotel': 0 if hotel == "City Hotel" else 1,
        'customer_type': ctype,
        'is_family': 0,
        'deposit_given': 0,
    }
    results = lookup_by_profile(profile)
    st.dataframe(
        results.style.map(
            lambda v: 'color: red' if v == 'Cancelled ❌' else 'color: green',
            subset=['outcome']
        ),
        use_container_width=True
    )