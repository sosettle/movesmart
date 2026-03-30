# 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# standardized_indicies_df = pd.read_csv("data/processed/standardized_indicies_df.csv")
# standardized_indicies_df = standardized_indicies_df.drop(columns=['Unnamed: 0'])
# df = recommender.recommended_cities(
#     df,
#     user_inputs,
#     user_income=1000000,
#     housing_mode="rent",
#     top_n=50
# )

class Visualization:
    def __init__(self, user_inputs):
        self.user_inputs_scaled = {k: v * 10 for k, v in user_inputs.items()}
        self.RADAR_COLS = list(self.user_inputs_scaled.keys())

        self.DISPLAY_LABELS = {
            "affordability_score": "Affordability",
            "job_growth_score": "Job Growth",
            "safety_score": "Safety",
            "education_score": "Education",
            "health_score": "Health",
            "walkability_score": "Walkability",
            "diversity_score": "Diversity",
            "urban_score": "Urban",
            "weather_warmth_score": "Warmth",
            "weather_mildness_score": "Mildness",
        }

    # =========================
    # HELPERS
    # =========================
    def round_df_numeric(self, df_in, decimals=2):
        df_out = df_in.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        df_out[numeric_cols] = df_out[numeric_cols].round(decimals)
        return df_out

    def prepare_plot_df(self, df):
        df = df.copy()
        df["city_state"] = df["city"].astype(str) + ", " + df["state"].astype(str)

        if "cluster_label" not in df.columns:
            df["cluster_label"] = "Cluster " + df["sub_cluster_text"].astype(str)

        return self.round_df_numeric(df, 2)

    def get_top_n(self, df, n=25):
        return df.sort_values("recommendation_score", ascending=False).head(n).copy()

    # =========================
    # 1. RADAR
    # =========================
    def plot_radar(self, df):
        df = self.prepare_plot_df(df)
        row = self.get_top_n(df, 1).iloc[0]

        categories = [self.DISPLAY_LABELS[c] for c in self.RADAR_COLS]
        user_vals = [round(self.user_inputs_scaled[c], 2) for c in self.RADAR_COLS]
        city_vals = [round(row[c], 2) for c in self.RADAR_COLS]

        categories += [categories[0]]
        user_vals += [user_vals[0]]
        city_vals += [city_vals[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(r=city_vals, theta=categories, fill="toself", name=row["city_state"]))
        fig.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill="toself", name="User Preferences", opacity=0.4))

        return fig

    # =========================
    # 2. CONTRIBUTIONS
    # =========================
    def plot_contributions(self, df):
        df = self.prepare_plot_df(df)
        row = self.get_top_n(df, 1).iloc[0]

        data = []
        for col in self.RADAR_COLS:
            gap = abs(row[col] - self.user_inputs_scaled[col])
            score = max(0, 100 - gap)
            contribution = score * self.user_inputs_scaled[col] / 100

            data.append({
                "feature": self.DISPLAY_LABELS[col],
                "contribution": round(contribution, 2),
                "gap": round(gap, 2),
                "user_preference": round(self.user_inputs_scaled[col], 2),
                "city_value": round(row[col], 2)
            })

        cdf = pd.DataFrame(data).sort_values("contribution", ascending=False)

        fig = px.bar(
            cdf,
            x="contribution",
            y="feature",
            orientation="h"
        )

        return fig

    # =========================
    # 3. MAP
    # =========================
    def plot_map(self, df):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, 5)

        size_vals = np.clip(df["recommendation_score"] / 12, 4, 10)

        fig = px.scatter_geo(
            df,
            lat="centroid_lat",
            lon="centroid_lon",
            color="cluster_label",
            size=size_vals,
            scope="usa"
        )

        fig.add_trace(go.Scattergeo(
            lat=top["centroid_lat"],
            lon=top["centroid_lon"],
            text=top["city_state"],
            mode="markers+text",
            marker=dict(size=14, symbol="star", color="yellow")
        ))

        return fig

    # =========================
    # 4. CLUSTER PROFILE
    # =========================
    def plot_cluster_profile(self, df, top_n=25):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, top_n)

        cluster = top.iloc[0]["cluster_label"]
        cluster_df = top[top["cluster_label"] == cluster]

        cluster_med = cluster_df[self.RADAR_COLS].median().round(2)

        compare = pd.DataFrame({
            "feature": [self.DISPLAY_LABELS[c] for c in self.RADAR_COLS],
            "Cluster Median": [cluster_med[c] for c in self.RADAR_COLS],
            "User Preferences": [self.user_inputs_scaled[c] for c in self.RADAR_COLS]
        })

        long = compare.melt(id_vars="feature", var_name="series", value_name="value")

        fig = px.bar(long, x="value", y="feature", color="series", orientation="h")

        return fig

    # =========================
    # 5. TABLE
    # =========================
    def plot_table(self, df, top_n=10):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, top_n)

        table_df = top[[
            "city_state", "cluster_label", "recommendation_score"
        ]].copy()

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(table_df.columns)),
            cells=dict(values=[table_df[col] for col in table_df.columns])
        )])

        return fig

    # =========================
    # 6. DROPDOWN MAP
    # =========================
    def plot_dropdown_map(self, df):
        df = self.prepare_plot_df(df)

        features = self.RADAR_COLS + ["recommendation_score"]

        fig = go.Figure()

        for i, f in enumerate(features):
            fig.add_trace(go.Scattergeo(
                lat=df["centroid_lat"],
                lon=df["centroid_lon"],
                marker=dict(color=df[f]),
                visible=(i == 0),
                name=self.DISPLAY_LABELS.get(f, f)
            ))

        return fig

