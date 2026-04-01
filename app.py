"""
streamlit app
"""
import streamlit as st
import pandas as pd

from src import recommender
from src import visualizations

SLIDER_MIN = 0.0
SLIDER_MAX = 5.0
SLIDER_DEFAULT = 2.5
SLIDER_STEP = 0.01


def round_prefs(d: dict) -> dict:
    return {k: round(float(v), 2) for k, v in d.items()}


def metro_label(city: dict) -> str:
    return str(city.get("cbsa_name_y") or city.get("cbsa_name") or "Unknown")


def fmt_score(x) -> str:
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "—"


# --- Page Config ---
st.set_page_config(
    page_title="MoveSmart",
    layout="wide"
)

# -------------------------
# Dummy Data Functions
# -------------------------
def dummy_similar_cities(city_name):
    """Placeholder rows matching recommendation card fields (0–5 scale)."""
    return [
        {
            "sub_cluster_text": "Similar A",
            "cbsa_name_y": "Boulder, CO Metro Area",
            "tagline": "Outdoor lifestyle and strong tech scene.",
            "affordability_score": 2.25,
            "job_growth_score": 3.60,
            "health_score": 4.40,
            "diversity_score": 3.10,
            "weather_warmth_score": 2.80,
            "weather_mildness_score": 3.20,
            "safety_score": 4.10,
            "education_score": 4.50,
            "walkability_score": 3.30,
            "urban_score": 3.00,
            "cluster": "cluster1",
        },
        {
            "sub_cluster_text": "Similar B",
            "cbsa_name_y": "Salt Lake City, UT Metro Area",
            "tagline": "Mountains, growth, and a diversifying economy.",
            "affordability_score": 2.75,
            "job_growth_score": 3.75,
            "health_score": 4.00,
            "diversity_score": 2.90,
            "weather_warmth_score": 2.50,
            "weather_mildness_score": 3.40,
            "safety_score": 3.85,
            "education_score": 3.95,
            "walkability_score": 2.85,
            "urban_score": 3.25,
            "cluster": "cluster2",
        },
        {
            "sub_cluster_text": "Similar C",
            "cbsa_name_y": "Portland, OR Metro Area",
            "tagline": "Creative culture and walkable neighborhoods.",
            "affordability_score": 2.50,
            "job_growth_score": 3.40,
            "health_score": 4.10,
            "diversity_score": 3.55,
            "weather_warmth_score": 2.60,
            "weather_mildness_score": 3.50,
            "safety_score": 3.20,
            "education_score": 4.20,
            "walkability_score": 3.80,
            "urban_score": 3.60,
            "cluster": "cluster3",
        },
    ]

# -------------------------
# Load Data
# -------------------------
standardized_indicies_df = pd.read_csv("data/final/Final_Enriched_Dataset.csv")
# standardized_indicies_df = standardized_indicies_df.drop(columns=['Unnamed: 0'])

# sample_cbsa = [12420, 19100, 19740, 31080, 39580, 41860, 41940, 42660]
# standardized_indicies_df = standardized_indicies_df[
#     standardized_indicies_df['cbsa_code'].isin(sample_cbsa)
# ]

# print(len(standardized_indicies_df))

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
st.markdown(
    """
<style>
/*
 * Do not set .stApp background with !important — it forces a light grey shell while Streamlit
 * dark mode still applies light-colored text, yielding unreadable contrast. Let the app theme
 * control the main surface; custom blocks below carry their own colors.
 */

.navbar {
    background: #0F172A;
    padding: 1rem 2.5rem;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #F8FAFC;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.navbar .logo {
    font-weight: 700;
    font-size: 1.5rem;
    color: #F8FAFC;
}

.navbar .nav-links,
.navbar .nav-links span {
    font-size: 0.9rem;
    color: rgba(248, 250, 252, 0.88);
}

.city-card {
    background-color: #FFFFFF;
    color: #0F172A;
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
    color: #475569;
    font-size: 0.95rem;
}

.badge {
    background: #F1F5F9;
    color: #334155;
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

/* Prefer Streamlit’s theme for native widgets (sliders, tabs, subheaders). */
[data-testid="stAppViewContainer"] .main {
    color: inherit;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# UI Components
# -------------------------
def city_card_html(city):
    name = metro_label(city)
    badge = city.get("sub_cluster_text") or "—"
    tag = city.get("tagline") or ""
    return f"""
    <div class="city-card">
        <div>
            <span class="badge">{badge}</span>
            <div class="city-name">{name}</div>
            <p class="city-desc">{tag}</p>
        </div>
        <div class="stat-column">
            <span class="stat-item">💰 Affordability: {fmt_score(city.get("affordability_score"))}</span>
            <span class="stat-item">💼 Jobs: {fmt_score(city.get("job_growth_score"))}</span>
            <span class="stat-item">❤️ Health: {fmt_score(city.get("health_score"))}</span>
            <span class="stat-item">🌍 Diversity: {fmt_score(city.get("diversity_score"))}</span>
            <span class="stat-item">🌡️ Climate (Warmth): {fmt_score(city.get("weather_warmth_score"))}</span>
            <span class="stat-item">🌤️ Climate (Mildness): {fmt_score(city.get("weather_mildness_score"))}</span>
            <span class="stat-item">🛡️ Safety: {fmt_score(city.get("safety_score"))}</span>
            <span class="stat-item">🎓 Education: {fmt_score(city.get("education_score"))}</span>
            <span class="stat-item">🚶 Walkability: {fmt_score(city.get("walkability_score"))}</span>
            <span class="stat-item">🏙️ Urban: {fmt_score(city.get("urban_score"))}</span>
        </div>
    </div>
    """

# -------------------------
# Navbar
# -------------------------
st.markdown(
    """
<div class="navbar">
    <div class="logo">🏙 MoveSmart</div>
    <div class="nav-links">
        <span style="margin-left: 20px;">Methodology</span>
        <span style="margin-left: 20px;">About</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------
# HOME PAGE
# -------------------------
if st.session_state.page == "home":

    col1, col2 = st.columns([1, 4], gap="large")

    # ----------------- INPUTS -----------------
    with col1:
        st.subheader("Preferences")

        user_income = st.number_input("Enter your annual income", value=None, step=1000, format="%d")

        affordability_score = st.slider(
            "Affordability", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        job_growth_score = st.slider(
            "Job Growth", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        health_score = st.slider(
            "Health", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        safety_score = st.slider(
            "Safety", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        education_score = st.slider(
            "Education", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        walkability_score = st.slider(
            "Walkability", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        diversity_score = st.slider(
            "Diversity", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        urban_score = st.slider(
            "Urban", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        weather_warmth_score = st.slider(
            "Weather Warmth", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )
        weather_mildness_score = st.slider(
            "Weather Mildness", SLIDER_MIN, SLIDER_MAX, SLIDER_DEFAULT, step=SLIDER_STEP, format="%.2f"
        )

        # ----------------- BUTTON -----------------
        if st.button("Find My City", use_container_width=True, type="primary"):

            user_inputs = round_prefs(
                {
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
            )

            results_df = recommender.recommend_cities(
                df=standardized_indicies_df,
                user_inputs=user_inputs,
                user_income=user_income,
                housing_mode="rent",
                top_n=15,
            )

            results = recommender.add_text_to_cbsa(results_df).to_dict(orient="records")

            # Persist for reruns
            st.session_state.recommendations = results
            st.session_state.user_inputs = user_inputs
            st.session_state.results_df = results_df

    # ----------------- RIGHT PANEL (2×2 charts) -----------------
    with col2:
        if st.session_state.user_inputs and st.session_state.results_df is not None:
            viz = visualizations.Visualization(st.session_state.user_inputs)

            map_color_labels = [lab for lab, _ in visualizations.MAP_COLOR_COLUMN_OPTIONS]
            map_label_to_col = visualizations.MAP_COLOR_LABEL_TO_COLUMN
            map_color_label = st.selectbox(
                "Color map by",
                map_color_labels,
                index=0,
                key="viz_map_color_by",
                help="Cluster: categorical colors. Scores: 0–5 Viridis gradient on the map pane.",
            )
            map_color_col = map_label_to_col[map_color_label]

            row1a, row1b = st.columns(2, gap="medium")
            row2a, row2b = st.columns(2, gap="medium")

            with row1a:
                st.markdown("##### Reccomendation vs Input Radar Chart")
                st.plotly_chart(
                    viz.plot_radar(st.session_state.results_df),
                    use_container_width=True,
                    theme="streamlit",
                )

            with row1b:
                st.markdown("##### Ranked Contributions")
                st.plotly_chart(
                    viz.plot_contributions(st.session_state.results_df),
                    use_container_width=True,
                    theme="streamlit",
                )

            with row2a:
                st.markdown("##### Map")
                st.plotly_chart(
                    viz.plot_map(st.session_state.results_df, color_column=map_color_col),
                    use_container_width=True,
                    theme="streamlit",
                )

            with row2b:
                st.markdown("##### Details")
                st.plotly_chart(
                    viz.plot_table(st.session_state.results_df),
                    use_container_width=True,
                    theme="streamlit",
                )

    # -------------------------
    # Recommendation Cards
    # -------------------------
    if st.session_state.recommendations:
        st.markdown("---")
        st.subheader("Recommended Matches")

        rec = st.session_state.recommendations

        for i in range(0, len(rec), 3):
            cols = st.columns(3)
            for j, city in enumerate(rec[i : i + 3]):
                idx = i + j
                with cols[j]:
                    st.markdown(city_card_html(city), unsafe_allow_html=True)
                    st.caption(f"Match score: **{fmt_score(city.get('recommendation_score'))}** / {SLIDER_MAX:.2f}")

                    with st.expander("📖 About this metro"):
                        st.write(city.get("summary", ""))
                        st.write("\n----------------------\n")
                        st.write(city.get("city_wiki_wikivoyage_text", ""))

                    if st.button("Explore Similar Cluster", key=f"explore_cluster_{idx}"):
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