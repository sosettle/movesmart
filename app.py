"""
streamlit app
"""
import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
from src import recommender
from src import visualizations

# --- Page Config ---
st.set_page_config(
    page_title="MoveSmart",
    layout="wide"
)

# -------------------------
# Dummy Data Functions
# -------------------------
def dummy_similar_cities(city_name):
    """Return similar cities with scores and text"""
    return [
        {
            "name": "Boulder",
            "affordability": 0.45,
            "jobs": 0.72,
            "health": 0.88,
            "text": "Boulder is known for its outdoor lifestyle, tech startups, and high quality of life.",
            "cluster": "cluster1"
        },
        {
            "name": "Salt Lake City",
            "affordability": 0.55,
            "jobs": 0.75,
            "health": 0.80,
            "text": "Salt Lake City offers strong job growth, access to mountains, and a growing tech sector.",
            "cluster": "cluster2"
        },
        {
            "name": "Portland",
            "affordability": 0.50,
            "jobs": 0.68,
            "health": 0.82,
            "text": "Portland is known for its creative culture, green spaces, and walkable neighborhoods.",
            "cluster": "cluster3"
        }
    ]

# -------------------------
# Load Data
# -------------------------
standardized_indicies_df = pd.read_csv("data/processed/Final_Enriched_Dataset.csv")
# standardized_indicies_df = standardized_indicies_df.drop(columns=['Unnamed: 0'])

sample_cbsa = [12420, 19100, 19740, 31080, 39580, 41860, 41940, 42660]
standardized_indicies_df = standardized_indicies_df[
    standardized_indicies_df['cbsa_code'].isin(sample_cbsa)
]

print(len(standardized_indicies_df))

# -------------------------
# Initialize Session State
# -------------------------
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "similar_cities" not in st.session_state:
    st.session_state.similar_cities = []

if "page" not in st.session_state:
    st.session_state.page = "home"

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None

if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = None

if "results_df" not in st.session_state:
    st.session_state.results_df = None

# -------------------------
# CSS Styling
# -------------------------
st.markdown("""
<style>
.stApp { background-color: #F1F5F9 !important; }

.navbar {
    background: #0F172A;
    padding: 1rem 2.5rem;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.logo {
    font-weight: 700;
    font-size: 1.5rem;
}

.city-card {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 6px solid #0F172A;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    transition: transform 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.city-card:hover {
    transform: translateY(-4px);
}

.city-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0F172A;
}

.city-desc {
    color: #64748B;
    font-size: 0.95rem;
}

.badge {
    background: #F1F5F9;
    color: #475569;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
}

.stat-column {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stat-item {
    font-size: 0.85rem;
    color: #1E293B;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# UI Components
# -------------------------
def city_card_html(city):
    return f"""
    <div class="city-card">
        <div>
            <span class="badge">{city['sub_cluster_text']}</span>
            <div class="city-name">{city['cbsa_name_y']}</div>
            <p class="city-desc">{city['tagline']}</p>
        </div>
        <div class="stat-column">
            <span class="stat-item">💰 Affordability: {city['affordability_score']}</span>
            <span class="stat-item">💼 Jobs: {city['job_growth_score']}</span>
            <span class="stat-item">❤️ Health: {city['health_score']}</span>
            <span class="stat-item">🌍 Diversity: {city['diversity_score']}</span>
            <span class="stat-item">🌡️ Climate (Warmth): {city['weather_warmth_score']}</span>
            <span class="stat-item">🌤️ Climate (Mildness): {city['weather_mildness_score']}</span>
            <span class="stat-item">🛡️ Safety: {city['safety_score']}</span>
            <span class="stat-item">🎓 Education: {city['education_score']}</span>
            <span class="stat-item">🚶 Walkability: {city['walkability_score']}</span>
            <span class="stat-item">🏙️ Urban: {city['urban_score']}</span>
        </div>
    </div>
    """

# -------------------------
# Navbar
# -------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">🏙 MoveSmart</div>
    <div style="font-size: 0.9rem; opacity: 0.8;">
        <span style="margin-left: 20px;">Methodology</span>
        <span style="margin-left: 20px;">About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# HOME PAGE
# -------------------------
if st.session_state.page == "home":

    col1, col2 = st.columns([1, 4], gap="large")

    # ----------------- INPUTS -----------------
    with col1:
        st.subheader("Preferences")

        user_income = st.number_input("Enter your annual income", value=None, step=1000, format="%d")

        affordability_score = st.slider("Affordability", 0.0, 5.0, 1.0)
        job_growth_score = st.slider("Job Growth", 0.0, 5.0, 1.0)
        health_score = st.slider("Health", 0.0, 5.0, 1.0)
        safety_score = st.slider("Safety", 0.0, 5.0, 1.0)
        education_score = st.slider("Education", 0.0, 5.0, 1.0)
        walkability_score = st.slider("Walkability", 0.0, 5.0, 1.0)
        diversity_score = st.slider("Diversity", 0.0, 5.0, 1.0)
        urban_score = st.slider("Urban", 0.0, 5.0, 1.0)
        weather_warmth_score = st.slider("Weather Warmth", 0.0, 5.0, 1.0)
        weather_mildness_score = st.slider("Weather Mildness", 0.0, 5.0, 1.0)

        # ----------------- BUTTON -----------------
        if st.button("Find My City", use_container_width=True, type="primary"):

            user_inputs = {
                "affordability_score": affordability_score,
                "safety_score": safety_score,
                "job_growth_score": job_growth_score,
                "education_score": education_score,
                "health_score": health_score,
                "walkability_score": walkability_score,
                "diversity_score": diversity_score,
                "urban_score": urban_score,
                "weather_warmth_score": weather_warmth_score,
                "weather_mildness_score": weather_mildness_score,
            }

            results_df = recommender.recommend_cities(
                df=standardized_indicies_df,
                user_inputs=user_inputs,
                user_income=user_income,
                housing_mode="rent",
                top_n=10
            )

            print(results_df)

            results = recommender.add_text_to_cbsa(results_df).to_dict(orient='records')

            # Persist for reruns
            st.session_state.recommendations = results
            st.session_state.user_inputs = user_inputs
            st.session_state.results_df = results_df

    # ----------------- RIGHT PANEL -----------------
    with col2:
        st.subheader("Map View")

        m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron")

        if st.session_state.recommendations:
            for city in st.session_state.recommendations:
                folium.Marker(
                    location=[city["centroid_lat"], city["centroid_lon"]],
                    popup=city["cbsa_name_y"],
                    icon=folium.Icon(color="cadetblue", icon="info-sign")
                ).add_to(m)

        st_folium(m, width="100%", height=400, key="main_map")

        # ----------------- TABS -----------------
        if st.session_state.user_inputs and st.session_state.results_df is not None:

            viz = visualizations.Visualization(st.session_state.user_inputs)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Radar", "Contributions", "Map", "Cluster", "Table", "Dropdown"
            ])

            with tab1:
                st.plotly_chart(viz.plot_radar(st.session_state.results_df), use_container_width=True)

            with tab2:
                st.plotly_chart(viz.plot_contributions(st.session_state.results_df), use_container_width=True)

            with tab3:
                st.plotly_chart(viz.plot_map(st.session_state.results_df), use_container_width=True)

            with tab4:
                st.plotly_chart(viz.plot_cluster_profile(st.session_state.results_df), use_container_width=True)

            with tab5:
                st.plotly_chart(viz.plot_table(st.session_state.results_df), use_container_width=True)

            with tab6:
                st.plotly_chart(viz.plot_dropdown_map(st.session_state.results_df), use_container_width=True)

    # -------------------------
    # Recommendation Cards
    # -------------------------
    if st.session_state.recommendations:
        st.markdown("---")
        st.subheader("Recommended Matches")

        rec = st.session_state.recommendations

        for i in range(0, len(rec), 3):
            cols = st.columns(3)
            for j, city in enumerate(rec[i:i+3]):
                with cols[j]:
                    st.markdown(city_card_html(city), unsafe_allow_html=True)

                    with st.expander("📖 About this metro"):
                        st.write(city['summary'])
                        st.write("\n----------------------\n")
                        st.write(city['city_wiki_wikivoyage_text'])

                    if st.button("Explore Similar Cluster", key=f"cluster_{city['cbsa_name_y']}"):
                        st.session_state.selected_cluster = city["cluster"]
                        st.session_state.page = "cluster_page"
                        st.rerun()

# -------------------------
# CLUSTER PAGE
# -------------------------
elif st.session_state.page == "cluster_page":

    st.title("Cities with Similar Profiles")

    cluster = st.session_state.selected_cluster
    cities = dummy_similar_cities(cluster)

    st.write(f"Cluster: {cluster}")

    for i in range(0, len(cities), 3):
        cols = st.columns(3)
        for j, city in enumerate(cities[i:i+3]):
            with cols[j]:
                st.markdown(city_card_html(city), unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = "home"