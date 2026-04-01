import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# (selectbox label, dataframe column) for map coloring
MAP_COLOR_COLUMN_OPTIONS: list[tuple[str, str]] = [
    ("Cluster", "cluster_label"),
    ("Recommendation score", "recommendation_score"),
    ("Affordability", "affordability_score"),
    ("Job Growth", "job_growth_score"),
    ("Safety", "safety_score"),
    ("Education", "education_score"),
    ("Health", "health_score"),
    ("Walkability", "walkability_score"),
    ("Diversity", "diversity_score"),
    ("Urban", "urban_score"),
    ("Warmth", "weather_warmth_score"),
    ("Mildness", "weather_mildness_score"),
]

MAP_COLOR_LABEL_TO_COLUMN: dict[str, str] = dict(MAP_COLOR_COLUMN_OPTIONS)
MAP_COLUMN_TO_LABEL: dict[str, str] = {col: lab for lab, col in MAP_COLOR_COLUMN_OPTIONS}

MAP_SCORE_GRADIENT_COLUMNS: frozenset = frozenset(
    {
        "affordability_score",
        "job_growth_score",
        "safety_score",
        "education_score",
        "health_score",
        "walkability_score",
        "diversity_score",
        "urban_score",
        "weather_warmth_score",
        "weather_mildness_score",
        "recommendation_score",
    }
)


class Visualization:
    def __init__(self, user_inputs):
        # Same 0–5 scale as sliders and `recommend_cities` output
        self.user_inputs_scaled = {k: round(float(v), 2) for k, v in user_inputs.items()}
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

    def round_df_numeric(self, df_in, decimals=2):
        df_out = df_in.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        df_out[numeric_cols] = df_out[numeric_cols].round(decimals)
        return df_out

    def prepare_plot_df(self, df):
        df = df.copy()
        df["city_state"] = df["city"].astype(str) + ", " + df["state"].astype(str)

        if "sub_cluster_final_name" in df.columns:
            df["cluster_label"] = df["sub_cluster_final_name"].fillna("").astype(str).str.strip()
            if "sub_cluster_text" in df.columns:
                mask = df["cluster_label"].eq("")
                df.loc[mask, "cluster_label"] = df.loc[mask, "sub_cluster_text"].astype(str)
        elif "cluster_label" not in df.columns:
            df["cluster_label"] = "Cluster " + df["sub_cluster_text"].astype(str)

        df["cluster_label"] = df["cluster_label"].replace("", "—")

        return self.round_df_numeric(df, 2)

    def get_top_n(self, df, n=25):
        return df.sort_values("recommendation_score", ascending=False).head(n).copy()

    def plot_radar(self, df):
        df = self.prepare_plot_df(df)
        row = self.get_top_n(df, 1).iloc[0]

        categories = [self.DISPLAY_LABELS[c] for c in self.RADAR_COLS]
        user_vals = [round(self.user_inputs_scaled[c], 2) for c in self.RADAR_COLS]
        city_vals = [round(row[c], 2) for c in self.RADAR_COLS]

        categories += [categories[0]]
        user_vals += [user_vals[0]]
        city_vals += [city_vals[0]]

        ht = "<b>%{fullData.name}</b><br>%{theta}: %{r:.2f}<extra></extra>"
        # Strong contrast: deep blue (metro) vs vivid amber (user); distinct fills and heavy strokes
        city_line, city_fill = "#1D4ED8", "rgba(29,78,216,0.22)"
        user_line, user_fill = "#C026D3", "rgba(192,38,211,0.18)"
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=city_vals,
                theta=categories,
                fill="toself",
                name=row["city_state"],
                hovertemplate=ht,
                line=dict(color=city_line, width=3.5),
                fillcolor=city_fill,
                marker=dict(color=city_line, size=11, line=dict(color="#0f172a", width=1.5)),
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=user_vals,
                theta=categories,
                fill="toself",
                name="User preferences",
                hovertemplate=ht,
                line=dict(color=user_line, width=3.5, dash="dash"),
                fillcolor=user_fill,
                marker=dict(color=user_line, size=11, line=dict(color="#4a044e", width=1.5)),
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 5], tickformat=".2f", gridcolor="rgba(15,23,42,0.12)"),
                angularaxis=dict(linecolor="rgba(15,23,42,0.25)", gridcolor="rgba(15,23,42,0.08)"),
                bgcolor="rgba(248,250,252,0.6)",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        return fig

    def plot_contributions(self, df):
        df = self.prepare_plot_df(df)
        row = self.get_top_n(df, 1).iloc[0]

        data = []
        for col in self.RADAR_COLS:
            gap = abs(row[col] - self.user_inputs_scaled[col])
            score = max(0.0, 100.0 * (1.0 - gap / 5.0))
            contribution = score * self.user_inputs_scaled[col] / 100.0

            data.append(
                {
                    "feature": self.DISPLAY_LABELS[col],
                    "contribution": round(contribution, 2),
                    "gap": round(gap, 2),
                    "user_preference": round(self.user_inputs_scaled[col], 2),
                    "city_value": round(row[col], 2),
                }
            )

        cdf = pd.DataFrame(data).sort_values("contribution", ascending=True)
        feat_order = cdf["feature"].tolist()

        fig = px.bar(
            cdf,
            x="contribution",
            y="feature",
            orientation="h",
            category_orders={"feature": feat_order},
        )
        fig.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=feat_order),
        )
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>contribution: %{x:.2f}<br>"
            + "gap: %{customdata[0]:.2f}<br>"
            + "user: %{customdata[1]:.2f}<br>city: %{customdata[2]:.2f}<extra></extra>",
            customdata=cdf[["gap", "user_preference", "city_value"]].values,
        )
        fig.update_xaxes(tickformat=".2f")
        fig.update_yaxes(title=None)
        return fig

    def plot_map(self, df, color_column: str = "cluster_label"):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, 5)

        map_marker_px = 10
        size_vals = np.full(len(df), map_marker_px, dtype=float)

        col = color_column if color_column in df.columns else "cluster_label"
        is_cluster = col == "cluster_label"

        hover_extra = {"city_state": True, "recommendation_score": True}
        if col not in hover_extra and col in df.columns:
            hover_extra[col] = True

        if is_cluster:
            fig = px.scatter_geo(
                df,
                lat="centroid_lat",
                lon="centroid_lon",
                color="cluster_label",
                size=size_vals,
                size_max=int(map_marker_px),
                scope="usa",
                hover_data=hover_extra,
            )
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            if col in MAP_SCORE_GRADIENT_COLUMNS:
                rng: tuple[float, float] | list[float] = (0.0, 5.0)
            else:
                lo = float(np.nanmin(series.to_numpy()))
                hi = float(np.nanmax(series.to_numpy()))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    lo, hi = 0.0, 1.0
                if hi <= lo:
                    hi = lo + 1e-6
                rng = (lo, hi)

            fig = px.scatter_geo(
                df,
                lat="centroid_lat",
                lon="centroid_lon",
                color=series,
                color_continuous_scale="Viridis",
                range_color=rng,
                size=size_vals,
                size_max=int(map_marker_px),
                scope="usa",
                hover_data=hover_extra,
            )
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=MAP_COLUMN_TO_LABEL.get(col, col),
                    tickformat=".2f",
                ),
            )

        fig.add_trace(
            go.Scattergeo(
                lat=top["centroid_lat"],
                lon=top["centroid_lon"],
                text=top["city_state"],
                mode="markers+text",
                marker=dict(size=map_marker_px, symbol="star", color="#ca8a04", line=dict(width=0.6, color="#292524")),
                textfont=dict(color="#020617", size=12, family="system-ui, sans-serif"),
                textposition="top center",
                name="Top picks",
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

        # px maps `size` through sizeref; pin marker pixel size to match scattergeo `marker.size` (10).
        for tr in fig.data:
            if getattr(tr, "name", None) != "Top picks":
                tr.marker.size = map_marker_px

        return fig

    def plot_cluster_profile(self, df, top_n=25):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, top_n)

        cluster = top.iloc[0]["cluster_label"]
        cluster_df = top[top["cluster_label"] == cluster]

        cluster_med = cluster_df[self.RADAR_COLS].median().round(2)

        compare = pd.DataFrame(
            {
                "feature": [self.DISPLAY_LABELS[c] for c in self.RADAR_COLS],
                "Cluster Median": [cluster_med[c] for c in self.RADAR_COLS],
                "User Preferences": [round(self.user_inputs_scaled[c], 2) for c in self.RADAR_COLS],
            }
        )

        long = compare.melt(id_vars="feature", var_name="series", value_name="value")

        fig = px.bar(long, x="value", y="feature", color="series", orientation="h")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.2f}<extra></extra>")
        fig.update_xaxes(tickformat=".2f")
        return fig

    def plot_table(self, df, top_n=10):
        df = self.prepare_plot_df(df)
        top = self.get_top_n(df, top_n)

        table_df = top[["city_state", "cluster_label", "recommendation_score"]].copy()
        table_df["recommendation_score"] = table_df["recommendation_score"].map(lambda x: f"{float(x):.2f}")
        display_df = table_df.rename(
            columns={
                "city_state": "Metro",
                "cluster_label": "Cluster",
                "recommendation_score": "Recommendation Score",
            }
        )

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(display_df.columns),
                        fill_color="#0f172a",
                        font=dict(color="#f8fafc", size=13),
                        align="left",
                    ),
                    cells=dict(
                        values=[display_df[col] for col in display_df.columns],
                        fill_color="#f8fafc",
                        font=dict(color="#0f172a", size=12),
                        align="left",
                    ),
                )
            ]
        )
        return fig

    def plot_dropdown_map(self, df):
        df = self.prepare_plot_df(df)

        features = self.RADAR_COLS + ["recommendation_score"]

        fig = go.Figure()

        for i, f in enumerate(features):
            label = self.DISPLAY_LABELS.get(f, f)
            marker_kw: dict = dict(
                color=df[f],
                colorscale="Viridis",
                cmin=0,
                cmax=5,
                size=9,
            )
            if i == 0:
                marker_kw["colorbar"] = dict(title="", tickformat=".2f")
            fig.add_trace(
                go.Scattergeo(
                    lat=df["centroid_lat"],
                    lon=df["centroid_lon"],
                    marker=marker_kw,
                    visible=(i == 0),
                    name=label,
                    customdata=df[[f]],
                    hovertemplate=f"<b>{label}</b><br>%{{customdata[0]:.2f}}<extra></extra>",
                )
            )

        buttons = []
        for i, f in enumerate(features):
            vis = [j == i for j in range(len(features))]
            buttons.append(
                dict(
                    label=self.DISPLAY_LABELS.get(f, f),
                    method="update",
                    args=[{"visible": vis}],
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=1.02,
                    yanchor="bottom",
                )
            ]
        )
        return fig
