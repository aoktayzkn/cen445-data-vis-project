"""
Member 2 - Machine Learning Models & Advanced Multivariate Visualizations
Models:
  1) Unsupervised: K-Means Clustering + 3D PCA Dimensionality Reduction
  2) Supervised  : Random Forest Classifier (Viral Engagement) & Regressor (View Potential)
Visualizations:
  1) Interactive Multi-Dimensional Parallel Coordinates Plot (go.Parcoords)
  2) 3D PCA Behavioral Cluster Manifold with Centroids (px.scatter_3d + go.Scatter3d)
  3) Random Forest 2D Non-Linear Decision Surface Contour Map (go.Contour + go.Scatter)
     with interactive toggle for Gini Feature Importance & Tree Variance
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLUSTER_PALETTE = [
    "#ef4444",  # Red
    "#38bdf8",  # Sky Blue
    "#f59e0b",  # Amber
    "#10b981",  # Emerald
    "#a855f7",  # Purple
    "#ec4899",  # Pink
]


# ---------------------------------------------------------------------------
# 1) FEATURE ENGINEERING FOR MACHINE LEARNING
# ---------------------------------------------------------------------------
def engineer_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates log-transformed, ratio, and temporal features for ML modeling."""
    ml_df = df.copy()

    ml_df["log_views"] = np.log10(ml_df["views"].clip(lower=1))
    ml_df["log_likes"] = np.log10(ml_df["likes"].clip(lower=0) + 1)
    ml_df["log_comments"] = np.log10(ml_df["comment_count"].clip(lower=0) + 1)

    raw_eng = (ml_df["likes"] + ml_df["comment_count"]) / (ml_df["views"] + 1) * 100.0
    ml_df["engagement_rate"] = raw_eng.clip(upper=raw_eng.quantile(0.995))

    raw_com_rate = ml_df["comment_count"] / (ml_df["views"] + 1) * 1000.0
    ml_df["comments_per_1k"] = raw_com_rate.clip(upper=raw_com_rate.quantile(0.995))

    if "dislikes" in ml_df.columns:
        ml_df["approval_rate"] = (
            ml_df["likes"] / (ml_df["likes"] + ml_df["dislikes"] + 1) * 100.0
        )
    else:
        ml_df["approval_rate"] = 95.0

    ml_df["upper_ratio"] = (
        ml_df["title_upper_count"] / ml_df["title_length"].replace(0, 1) * 100.0
    ).clip(upper=100.0)

    if "publish_time" in ml_df.columns:
        pub_dt = pd.to_datetime(ml_df["publish_time"], errors="coerce", utc=True)
        ml_df["publish_hour"] = pub_dt.dt.hour.fillna(14).astype(int)
    else:
        ml_df["publish_hour"] = 14

    ml_df["days_to_trend"] = ml_df["days_to_trend"].clip(lower=0, upper=30)

    return ml_df


# ---------------------------------------------------------------------------
# 2) UNSUPERVISED ML: K-MEANS CLUSTERING + 3D PCA
# ---------------------------------------------------------------------------
def _assign_archetype_names(summary_df: pd.DataFrame) -> dict:
    """Assigns data-driven behavioral archetype labels based on cluster centroids."""
    names = {}
    remaining = set(summary_df.index)

    if remaining:
        c_viral = summary_df.loc[list(remaining), "log_views"].idxmax()
        names[c_viral] = "Mega-Viral Blockbusters"
        remaining.remove(c_viral)

    if remaining:
        c_eng = summary_df.loc[list(remaining), "engagement_rate"].idxmax()
        names[c_eng] = "High-Engagement Community"
        remaining.remove(c_eng)

    if remaining:
        c_fast = summary_df.loc[list(remaining), "upper_ratio"].idxmax()
        names[c_fast] = "Fast-Burn Click-Drivers"
        remaining.remove(c_fast)

    if remaining:
        c_seo = summary_df.loc[list(remaining), "tag_count"].idxmax()
        names[c_seo] = "SEO & Tag-Heavy Performers"
        remaining.remove(c_seo)

    fallback_labels = ["Mainstream Steady Reach", "Emerging / Niche Content"]
    for idx, c_id in enumerate(sorted(remaining)):
        names[c_id] = fallback_labels[idx % len(fallback_labels)]

    return {int(c_id): f"C{int(c_id) + 1}: {label}" for c_id, label in names.items()}


@st.cache_data(show_spinner=False)
def run_kmeans_pca_pipeline(df: pd.DataFrame, n_clusters: int = 4):
    """Executes K-Means clustering and 3D PCA projection with silhouette evaluation."""
    ml_df = engineer_ml_features(df)

    feature_cols = [
        "log_views",
        "engagement_rate",
        "comments_per_1k",
        "days_to_trend",
        "title_length",
        "upper_ratio",
        "tag_count",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(ml_df[feature_cols])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    ml_df["cluster_id"] = clusters.astype(int)

    sample_size = min(3500, len(ml_df))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(ml_df), size=sample_size, replace=False)
    sil_score = float(silhouette_score(X_scaled[sample_idx], clusters[sample_idx]))

    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    ml_df["PC1"] = pca_coords[:, 0]
    ml_df["PC2"] = pca_coords[:, 1]
    ml_df["PC3"] = pca_coords[:, 2]

    centroids_pca = pca.transform(kmeans.cluster_centers_)
    var_ratios = pca.explained_variance_ratio_ * 100.0

    centroid_summary = ml_df.groupby("cluster_id")[feature_cols].mean()
    archetype_map = _assign_archetype_names(centroid_summary)
    ml_df["archetype"] = ml_df["cluster_id"].map(archetype_map)

    centroids_df = pd.DataFrame(centroids_pca, columns=["PC1", "PC2", "PC3"])
    centroids_df["cluster_id"] = list(range(n_clusters))
    centroids_df["archetype"] = centroids_df["cluster_id"].map(archetype_map)
    centroids_df["video_count"] = [
        int((ml_df["cluster_id"] == i).sum()) for i in range(n_clusters)
    ]
    centroids_df["avg_views"] = [
        float(ml_df.loc[ml_df["cluster_id"] == i, "views"].mean())
        for i in range(n_clusters)
    ]
    centroids_df["avg_engagement"] = [
        float(ml_df.loc[ml_df["cluster_id"] == i, "engagement_rate"].mean())
        for i in range(n_clusters)
    ]

    return {
        "df": ml_df,
        "centroids_df": centroids_df,
        "silhouette": sil_score,
        "inertia": float(kmeans.inertia_),
        "var_ratios": var_ratios,
        "total_var": float(var_ratios.sum()),
        "archetype_map": archetype_map,
    }


# ---------------------------------------------------------------------------
# 3) SUPERVISED ML: RANDOM FOREST CLASSIFIER & REGRESSOR
# ---------------------------------------------------------------------------
RF_CONTINUOUS_FEATURES = {
    "tag_count": "Tag Count (SEO)",
    "title_length": "Title Length (Chars)",
    "upper_ratio": "Uppercase Ratio (%)",
    "publish_hour": "Publish Hour (0-23 UTC)",
    "days_to_trend": "Days to Trend (Velocity)",
}


@st.cache_data(show_spinner=False)
def run_random_forest_pipeline(df: pd.DataFrame):
    """
    Trains Random Forest models using pre-publication metadata and velocity features
    to predict High-Engagement Virality (Classification) and Engagement Rate (Regression).
    """
    ml_df = engineer_ml_features(df)

    median_eng = float(ml_df["engagement_rate"].median())
    y_clf = (ml_df["engagement_rate"] >= median_eng).astype(int)
    y_reg = ml_df["engagement_rate"].values

    cat_freq = ml_df["category"].value_counts(normalize=True)
    ml_df["category_tier"] = ml_df["category"].map(cat_freq).fillna(0.05)

    base_features = list(RF_CONTINUOUS_FEATURES.keys()) + ["category_tier"]
    country_dummies = pd.get_dummies(
        ml_df["country_code"], prefix="market", drop_first=False
    )
    X_df = pd.concat([ml_df[base_features], country_dummies], axis=1)

    feature_display_names = {
        **RF_CONTINUOUS_FEATURES,
        "category_tier": "Category Popularity Weight",
        **{col: f"Market: {col.split('_')[1]}" for col in country_dummies.columns},
    }

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X_df, y_clf, y_reg, test_size=0.25, random_state=42, stratify=y_clf
    )

    rf_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=9,
        min_samples_leaf=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_clf_train)

    y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]
    y_pred_cls = (y_pred_proba >= 0.5).astype(int)
    clf_auc = float(roc_auc_score(y_clf_test, y_pred_proba))
    clf_acc = float(accuracy_score(y_clf_test, y_pred_cls))

    rf_reg = RandomForestRegressor(
        n_estimators=100,
        max_depth=9,
        min_samples_leaf=12,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_reg_train)
    reg_r2 = float(r2_score(y_reg_test, rf_reg.predict(X_test)))

    importances = rf_clf.feature_importances_
    tree_std = np.std(
        [tree.feature_importances_ for tree in rf_clf.estimators_], axis=0
    )

    imp_df = pd.DataFrame(
        {
            "feature": X_df.columns,
            "display_name": [feature_display_names.get(c, c) for c in X_df.columns],
            "importance": importances * 100.0,
            "std": tree_std * 100.0,
        }
    ).sort_values("importance", ascending=True)

    return {
        "rf_clf": rf_clf,
        "X_columns": list(X_df.columns),
        "X_medians": X_df.median().to_dict(),
        "ml_df": ml_df,
        "median_eng": median_eng,
        "roc_auc": clf_auc,
        "accuracy": clf_acc,
        "reg_r2": reg_r2,
        "importance_df": imp_df,
    }


# ---------------------------------------------------------------------------
# 4) ADVANCED VISUALIZATION 1: INTERACTIVE PARALLEL COORDINATES (go.Parcoords)
# ---------------------------------------------------------------------------
def draw_parallel_coordinates(kmeans_res: dict, max_samples: int = 2500):
    """
    Draws an interactive multi-dimensional Parallel Coordinates plot for K-Means clusters.
    """
    ml_df = kmeans_res["df"]
    archetype_map = kmeans_res["archetype_map"]
    n_clusters = len(archetype_map)

    if len(ml_df) > max_samples:
        plot_df = ml_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        plot_df = ml_df.copy()

    selected_colors = CLUSTER_PALETTE[:n_clusters]
    if n_clusters == 1:
        colorscale = [[0.0, selected_colors[0]], [1.0, selected_colors[0]]]
    else:
        colorscale = [
            [i / (n_clusters - 1), selected_colors[i]] for i in range(n_clusters)
        ]

    eng_cap = float(np.percentile(plot_df["engagement_rate"], 99))
    com_cap = float(np.percentile(plot_df["comments_per_1k"], 99))

    dimensions = [
        dict(
            range=[float(plot_df["log_views"].min()), float(plot_df["log_views"].max())],
            label="Log10(Views)",
            values=plot_df["log_views"],
        ),
        dict(
            range=[0, eng_cap],
            label="Engagement (%)",
            values=plot_df["engagement_rate"].clip(upper=eng_cap),
        ),
        dict(
            range=[0, com_cap],
            label="Comments / 1K",
            values=plot_df["comments_per_1k"].clip(upper=com_cap),
        ),
        dict(
            range=[0, float(plot_df["days_to_trend"].max())],
            label="Days to Trend",
            values=plot_df["days_to_trend"],
        ),
        dict(
            range=[float(plot_df["title_length"].min()), float(plot_df["title_length"].max())],
            label="Title Length",
            values=plot_df["title_length"],
        ),
        dict(
            range=[0, 100],
            label="Uppercase (%)",
            values=plot_df["upper_ratio"],
        ),
        dict(
            range=[0, float(plot_df["tag_count"].max())],
            label="Tag Count",
            values=plot_df["tag_count"],
        ),
        dict(
            range=[0, max(1, n_clusters - 1)],
            tickvals=list(range(n_clusters)),
            ticktext=[f"C{i+1}" for i in range(n_clusters)],
            label="Cluster ID",
            values=plot_df["cluster_id"],
        ),
    ]

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=plot_df["cluster_id"],
                colorscale=colorscale,
                showscale=True,
                cmin=-0.25,
                cmax=max(1, n_clusters - 1) + 0.25,
                colorbar=dict(
                    title=dict(text="Cluster", font=dict(size=11)),
                    tickvals=list(range(n_clusters)),
                    ticktext=[f"C{i+1}" for i in range(n_clusters)],
                    thickness=14,
                    len=0.72,
                    y=0.46,
                ),
            ),
            dimensions=dimensions,
            labelfont=dict(size=12, color="#f8fafc", family="sans-serif"),
            tickfont=dict(size=10, color="#cbd5e1", family="sans-serif"),
            # Hide redundant range bounds so they never collide with top/bottom ticks
            rangefont=dict(size=1, color="rgba(0,0,0,0)"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Multi-Dimensional Behavioral Cluster Signatures (Interactive Parallel Coordinates)",
            font=dict(size=15),
            y=0.96,
            x=0,
            xanchor="left",
        ),
        height=460,
        margin=dict(l=55, r=65, t=95, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color="#e2e8f0"),
    )

    return fig


# ---------------------------------------------------------------------------
# 5) ADVANCED VISUALIZATION 2: 3D PCA CLUSTER MANIFOLD & CENTROIDS
# ---------------------------------------------------------------------------
def draw_pca_3d_clusters(kmeans_res: dict, max_samples: int = 2200):
    """
    Projects 7D behavioral space into 3D Principal Components (PC1, PC2, PC3)
    with K-Means archetype coloring and 3D diamond centroid markers.
    """
    ml_df = kmeans_res["df"]
    centroids_df = kmeans_res["centroids_df"]
    var_ratios = kmeans_res["var_ratios"]
    archetype_map = kmeans_res["archetype_map"]

    if len(ml_df) > max_samples:
        plot_df = ml_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        plot_df = ml_df.copy()

    ordered_archetypes = [archetype_map[i] for i in sorted(archetype_map.keys())]
    color_map = {
        archetype_map[i]: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        for i in sorted(archetype_map.keys())
    }

    plot_df["short_title"] = plot_df["title"].apply(
        lambda t: t if len(str(t)) <= 45 else str(t)[:42] + "..."
    )
    plot_df["marker_size"] = np.clip(plot_df["log_views"] * 2.2, 4, 14)

    fig = px.scatter_3d(
        plot_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="archetype",
        size="marker_size",
        size_max=13,
        opacity=0.75,
        category_orders={"archetype": ordered_archetypes},
        color_discrete_map=color_map,
        custom_data=[
            "short_title",
            "channel_title",
            "country",
            "category",
            "views",
            "engagement_rate",
            "days_to_trend",
        ],
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Channel: %{customdata[1]} (%{customdata[2]})<br>"
            "Category: %{customdata[3]}<br>"
            "Views: %{customdata[4]:,.0f}<br>"
            "Engagement Rate: %{customdata[5]:.2f}%<br>"
            "Days to Trend: %{customdata[6]}d<extra></extra>"
        ),
        marker=dict(line=dict(width=0)),
    )

    fig.add_trace(
        go.Scatter3d(
            x=centroids_df["PC1"],
            y=centroids_df["PC2"],
            z=centroids_df["PC3"],
            mode="markers+text",
            name="Cluster Centroids",
            text=[f"C{i+1}" for i in centroids_df["cluster_id"]],
            textposition="top center",
            textfont=dict(size=11, color="#ffffff"),
            customdata=centroids_df[
                ["archetype", "video_count", "avg_views", "avg_engagement"]
            ].values,
            hovertemplate=(
                "<b>🎯 %{customdata[0]} (Centroid)</b><br>"
                "Videos in Cluster: %{customdata[1]:,}<br>"
                "Avg Views: %{customdata[2]:,.0f}<br>"
                "Avg Engagement: %{customdata[3]:.2f}%<extra></extra>"
            ),
            marker=dict(
                size=8,
                symbol="diamond",
                color="#ffffff",
                line=dict(color="#0f172a", width=2),
                opacity=1.0,
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"3D PCA Cluster Manifold ({kmeans_res['total_var']:.1f}% Total Variance Captured)",
            font=dict(size=14),
            y=0.97,
            x=0,
            xanchor="left",
        ),
        height=520,
        margin=dict(l=5, r=5, t=95, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.86,
            xanchor="left",
            x=0,
            font=dict(size=10),
            title=dict(text=""),
        ),
        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 0.84]),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.55, y=1.55, z=1.2)),
            xaxis=dict(
                title=f"PC1 ({var_ratios[0]:.1f}% Var)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                gridcolor="#334155",
                showbackground=True,
            ),
            yaxis=dict(
                title=f"PC2 ({var_ratios[1]:.1f}% Var)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                gridcolor="#334155",
                showbackground=True,
            ),
            zaxis=dict(
                title=f"PC3 ({var_ratios[2]:.1f}% Var)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                gridcolor="#334155",
                showbackground=True,
            ),
        ),
        font=dict(family="sans-serif", size=12),
    )

    return fig


# ---------------------------------------------------------------------------
# 6) ADVANCED VISUALIZATION 3: RANDOM FOREST DECISION SURFACE & IMPORTANCE
# ---------------------------------------------------------------------------
def draw_rf_decision_surface(
    rf_res: dict,
    x_feat: str = "tag_count",
    y_feat: str = "title_length",
    view_mode: str = "surface",
):
    """
    Draws either:
      - 'surface': 2D Non-Linear Random Forest Probability Response Surface (go.Contour)
      - 'importance': Non-Linear Gini Feature Importance with Tree Variance error bars.
    """
    if view_mode == "importance":
        imp_df = rf_res["importance_df"].tail(10)
        fig = go.Figure(
            go.Bar(
                x=imp_df["importance"],
                y=imp_df["display_name"],
                orientation="h",
                error_x=dict(
                    type="data",
                    array=imp_df["std"],
                    color="#94a3b8",
                    thickness=1.2,
                    width=4,
                ),
                marker=dict(
                    color=imp_df["importance"],
                    colorscale="YlOrRd",
                    line=dict(color="#1e293b", width=1),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Gini Importance: %{x:.2f}%<br>"
                    "Tree Std Dev: ±%{error_x.array:.2f}%<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=dict(
                text=f"Random Forest Feature Importance (ROC-AUC: {rf_res['roc_auc']:.3f})",
                font=dict(size=14),
                y=0.96,
                x=0,
                xanchor="left",
            ),
            xaxis_title="Relative Predictive Importance (%)",
            yaxis_title="",
            height=520,
            margin=dict(l=10, r=15, t=65, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", size=12),
        )
        return fig

    rf_clf = rf_res["rf_clf"]
    ml_df = rf_res["ml_df"]
    X_cols = rf_res["X_columns"]
    X_medians = rf_res["X_medians"]
    median_eng = rf_res["median_eng"]

    x_min, x_max = np.percentile(ml_df[x_feat], [1, 99])
    y_min, y_max = np.percentile(ml_df[y_feat], [1, 99])
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    x_grid = np.linspace(x_min, x_max, 42)
    y_grid = np.linspace(y_min, y_max, 42)
    xx, yy = np.meshgrid(x_grid, y_grid)

    grid_df = pd.DataFrame({col: X_medians[col] for col in X_cols}, index=range(xx.size))
    grid_df[x_feat] = xx.ravel()
    grid_df[y_feat] = yy.ravel()

    z_prob = rf_clf.predict_proba(grid_df[X_cols])[:, 1].reshape(xx.shape) * 100.0

    sample_n = min(450, len(ml_df))
    sample_df = ml_df.sample(n=sample_n, random_state=42).copy()
    sample_df = sample_df[
        (sample_df[x_feat].between(x_min, x_max))
        & (sample_df[y_feat].between(y_min, y_max))
    ]
    sample_df["tier"] = (sample_df["engagement_rate"] >= median_eng).map(
        {True: "High Engagement (Actual)", False: "Standard Engagement (Actual)"}
    )
    sample_df["short_title"] = sample_df["title"].apply(
        lambda t: t if len(str(t)) <= 40 else str(t)[:37] + "..."
    )

    x_label = RF_CONTINUOUS_FEATURES.get(x_feat, x_feat)
    y_label = RF_CONTINUOUS_FEATURES.get(y_feat, y_feat)

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_prob,
            colorscale="YlOrRd",
            opacity=0.88,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=10, color="#ffffff"),
            ),
            colorbar=dict(
                title=dict(text="Viral Prob (%)", font=dict(size=11)),
                thickness=14,
                len=0.78,
            ),
            hovertemplate=(
                f"{x_label}: %{{x:.1f}}<br>"
                f"{y_label}: %{{y:.1f}}<br>"
                "<b>RF Predicted High-Engagement Prob: %{z:.1f}%</b><extra></extra>"
            ),
        )
    )

    for tier_name, color_hex, symbol_str in [
        ("High Engagement (Actual)", "#22c55e", "circle"),
        ("Standard Engagement (Actual)", "#94a3b8", "x"),
    ]:
        sub = sample_df[sample_df["tier"] == tier_name]
        fig.add_trace(
            go.Scatter(
                x=sub[x_feat],
                y=sub[y_feat],
                mode="markers",
                name=tier_name,
                customdata=sub[
                    ["short_title", "category", "engagement_rate", "views"]
                ].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Category: %{customdata[1]}<br>"
                    f"{x_label}: %{{x}}<br>"
                    f"{y_label}: %{{y}}<br>"
                    "Actual Engagement: %{customdata[2]:.2f}%<br>"
                    "Views: %{customdata[3]:,.0f}<extra></extra>"
                ),
                marker=dict(
                    size=6.5,
                    color=color_hex,
                    symbol=symbol_str,
                    line=dict(width=0.8, color="#0f172a"),
                    opacity=0.85,
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"RF Virality Response Surface ({x_label} vs. {y_label})",
            font=dict(size=14),
            y=0.97,
            x=0,
            xanchor="left",
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=520,
        margin=dict(l=10, r=10, t=90, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        font=dict(family="sans-serif", size=12),
    )

    return fig