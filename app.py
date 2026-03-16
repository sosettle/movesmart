import streamlit as st
from streamlit_folium import st_folium
import folium

# --- Page Config ---
st.set_page_config(
    page_title="MoveSmart",
    layout="wide"
)

# -------------------------
# Dummy Data Functions
# -------------------------

def dummy_recommendations(user_weights):
    return [
        {"name": "Denver", "affordability": 0.5, "jobs": 0.7, "health": 0.9, "text": "Outdoor lifestyle, strong economy, and great access to mountains.", "lat": 39.7392, "lon":-104.9903, "cluster": "Mountain Hub"},
        {"name": "Austin", "affordability": 0.6, "jobs": 0.9, "health": 0.8, "text": "Tech hub with music culture and fast population growth.", "lat":30.2672, "lon":-97.7431, "cluster": "Tech Oasis"},
        {"name": "Seattle", "affordability": 0.4, "jobs": 0.85, "health": 0.9, "text": "Major tech jobs with beautiful nature and waterfront.", "lat": 47.6062, "lon": -122.3321, "cluster": "Coastal Elite"}
    ]

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

def dummy_city(city_name):
    """Return similar cities with scores and text"""
    return {
            "name": "Fremont",
            "affordability_score": 0.45,
            "job_growth_score": 0.72,
            "health_score": 0.88,
            "text": "Happiest City. Home to Telsa.",
            "lat": 37.55,
            "long": 121.99
        }

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

# -------------------------
# CSS
# -------------------------

st.markdown("""
<style>
    /* 1. Page Background */
    .stApp {
        background-color: #F1F5F9 !important;
    }

    /* 2.  Navy Navbar */
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
        letter-spacing: -0.5px;
    }

    /* 3. The City Card */
    .city-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        /* Navy Left Border */
        border-left: 6px solid #0F172A;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
        # height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .city-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .city-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0.5rem 0;
    }

    .city-desc {
        color: #64748B;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .badge {
        background: #F1F5F9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        border-top: 1px solid #F1F5F9;
        padding-top: 1rem;
    }

    .stat-item {
        font-size: 0.85rem;
        color: #1E293B;
        font-weight: 600;
    }
    .stat-column {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .stat-item {
        display: block;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------
# FUNCTIONS
# -------------------------

def city_card_html2(city):
        return f"""
        <div class="city-card">
            <div>
                <span class="badge">{city['cluster']}</span>
                <div class="city-name">{city['name']}</div>
                <p class="city-desc">{city['text']}</p>
            </div>
            <div class="stat-row">
                <span class="stat-item">💰 {city['affordability']}</span>
                <span class="stat-item">💼 {city['jobs']}</span>
                <span class="stat-item">❤️ {city['health']}</span>
            </div>
        </div>
        """
def city_card_html(city):
        return f"""
        <div class="city-card">
            <div>
                <span class="badge">{city['cluster']}</span>
                <div class="city-name">{city['name']}</div>
                <p class="city-desc">{city['text']}</p>
            </div>
            <div class="stat-column">
                <span class="stat-item">💰 Affordability: {city['affordability']}</span>
                <span class="stat-item">💼 Jobs: {city['jobs']}</span>
                <span class="stat-item">❤️ Health: {city['health']}</span>
                <span class="stat-item">🌍 Diversity: {city['health']}</span>
                <span class="stat-item">🌡️ Climate: {city['health']}</span>
                <span class="stat-item">🛡️ Safety: {city['health']}</span>
                <span class="stat-item">🎓 Education: {city['health']}</span>
             </div>
        </div>
        """



# -------------------------
# UI Header (Navbar)
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
# Sidebar/Input Logic
# -------------------------
if st.session_state.page == "home":

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Preferences")
        f_afford = st.slider("Affordability", 0.0, 1.0, 0.5)
        f_jobs = st.slider("Job Opportunities", 0.0, 1.0, 0.5)
        f_health = st.slider("Health", 0.0, 1.0, 0.5)
        f_safety = st.slider("Safety", 0.0, 1.0, 0.5)
        
        if st.button("Find My City", use_container_width=True, type="primary"):
            st.session_state.recommendations = dummy_recommendations([f_afford, f_jobs, f_health, f_safety])

    with col2:
        st.subheader("Map View")
        # Clean map with Positron tiles
        m = folium.Map(location=[39.5, -98.35], zoom_start=4, tiles="cartodbpositron")
        
        if st.session_state.recommendations:
            for city in st.session_state.recommendations:
                folium.Marker(
                    location=[city["lat"], city["lon"]],
                    popup=city["name"],
                    icon=folium.Icon(color="cadetblue", icon="info-sign")
                ).add_to(m)
        
        st_folium(m, width="100%", height=400, key="main_map")

    # -------------------------
    # Results Cards
    # -------------------------
    

    if st.session_state.recommendations:
        st.markdown("---")
        st.subheader("Recommended Matches")
        
        rec = st.session_state.recommendations
        # Display in a grid
        for i in range(0, len(rec), 3):
            cols = st.columns(3)
            for j, city in enumerate(rec[i:i+3]):
                with cols[j]:
                    st.markdown(city_card_html(city), unsafe_allow_html=True)
                    if st.button("Explore Similar Cluster", key=f"cluster_{city['name']}"):
                        st.session_state.selected_cluster = city["cluster"]
                        st.session_state.page = "cluster_page"
                        st.rerun()


elif st.session_state.page == "cluster_page":

    st.title("Cities with Similar Profiles")

    cluster = st.session_state.selected_cluster

    cities = dummy_similar_cities(cluster)

    st.write(f"Cluster: {cluster}")

    for i in range(0, len(cities), 3):
        cols = st.columns(3)
        for j, city in enumerate(cities[i:i+3]):
            with cols[j]:
                st.markdown(city_card_html(city),unsafe_allow_html=True)

    if st.button("⬅ Back"):
        st.session_state.page = "home"