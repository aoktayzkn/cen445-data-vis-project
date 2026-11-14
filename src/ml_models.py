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
from sklearn.model_selection import GroupShuffleSplit
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
    """Assign short, descriptive names using the strongest measured cluster features.

    The labels are descriptive, not causal. When k changes, names are recalculated.
    """
    names = {}
    remaining = set(summary_df.index)

    # Use distinct clusters for the main four measurable characteristics.
    rules = [
        ("log_views", "High Views"),
        ("engagement_rate", "High Engagement"),
        ("upper_ratio", "Uppercase Titles"),
        ("title_length", "Long Titles"),
    ]
    for feature, label in rules:
        if not remaining:
            break
        cluster_id = summary_df.loc[sorted(remaining), feature].idxmax()
        names[cluster_id] = label
        remaining.remove(cluster_id)

    # Extra clusters with k=5 or 6 do not receive unsupported claims.
    for cluster_id in sorted(remaining):
        names[cluster_id] = "Other Videos"

    return {
        int(cluster_id): f"C{int(cluster_id) + 1}: {label}"
        for cluster_id, label in names.items()
    }


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


    # PCA feature contributions
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1", "PC2", "PC3"]
    )

    print("\n--- PCA FEATURE LOADINGS ---")
    print(pca_loadings.round(3))


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


    # Numerical features
    base_features = list(RF_CONTINUOUS_FEATURES.keys())

    # One-hot encoding for video categories
    category_dummies = pd.get_dummies(
        ml_df["category"].fillna("Unknown"),
        prefix="category",
        dtype=int
    )

    # One-hot encoding for countries
    country_dummies = pd.get_dummies(
        ml_df["country_code"],
        prefix="market",
        dtype=int
    )

    # Combine all model features
    X_df = pd.concat(
        [ml_df[base_features], category_dummies, country_dummies],
        axis=1
    )




    feature_display_names = {
        **RF_CONTINUOUS_FEATURES,
        **{
            col: f"Category: {col.removeprefix('category_')}"
            for col in category_dummies.columns
        },
        **{
            col: f"Market: {col.removeprefix('market_')}"
            for col in country_dummies.columns
        },
    }



    # Split data by video ID to prevent data leakage
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X_df,
            y_clf,
            groups=ml_df["video_id"]
        )
    )

    # Training and test features
    X_train = X_df.iloc[train_idx]
    X_test = X_df.iloc[test_idx]

    # Classification targets
    y_clf_train = y_clf.iloc[train_idx]
    y_clf_test = y_clf.iloc[test_idx]

    # Regression targets
    y_reg_train = y_reg[train_idx]
    y_reg_test = y_reg[test_idx]


    # Check if video IDs overlap between train and test
    train_ids = set(ml_df["video_id"].iloc[train_idx])
    test_ids = set(ml_df["video_id"].iloc[test_idx])

    common_ids = train_ids.intersection(test_ids)

    print("Shared Video IDs:", len(common_ids))
    print("Training Rows:", len(X_train))
    print("Test Rows:", len(X_test))

    assert len(common_ids) == 0, "Data leakage detected!"


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
def draw_parallel_coordinates(kmeans_res: dict, max_samples: int = 220):
    """Interactive overview of the existing K-Means clusters.

    For clarity, only the displayed lines are sampled; K-Means and PCA are
    unchanged. The C1/C2/... labels are a display mapping, not new clusters.
    """
    ml_df = kmeans_res["df"]
    archetype_map = kmeans_res["archetype_map"]
    n_clusters = len(archetype_map)

    plot_df = (
        ml_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        if len(ml_df) > max_samples else ml_df.copy().reset_index(drop=True)
    )

    # One colour per cluster; transparent lines help avoid visual overplotting.
    colors = CLUSTER_PALETTE[:n_clusters]
    colorscale = []
    for i, color in enumerate(colors):
        rgb = tuple(int(color[j:j + 2], 16) for j in (1, 3, 5))
        rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.58)"
        a, b = i / n_clusters, (i + 1) / n_clusters
        colorscale.extend([[a, rgba], [b - 1e-6, rgba]])
    colorscale.append([1.0, colorscale[-1][1]])

    low = float(plot_df["log_views"].min())
    high = float(plot_df["log_views"].max())
    ticks = list(range(int(np.ceil(low)), int(np.floor(high)) + 1))

    def format_views(power):
        count = 10 ** power
        if count >= 1_000_000:
            return f"{count / 1_000_000:g}M"
        if count >= 1_000:
            return f"{count / 1_000:g}K"
        return f"{count:g}"

    engagement_cap = max(0.1, float(np.percentile(plot_df["engagement_rate"], 99)))
    dimensions = [
        dict(label="Views", values=plot_df["log_views"],
             range=[low, high] if low != high else [low - 0.5, high + 0.5],
             tickvals=ticks, ticktext=[format_views(t) for t in ticks]),
        dict(label="Engagement (%)", values=plot_df["engagement_rate"].clip(upper=engagement_cap),
             range=[0, engagement_cap]),
        dict(label="Trend Record Age (d)", values=plot_df["days_to_trend"],
             range=[0, max(1, float(plot_df["days_to_trend"].max()))]),
        dict(label="Title Length", values=plot_df["title_length"],
             range=[0, max(1, float(plot_df["title_length"].max()))]),
        dict(label="Uppercase (%)", values=plot_df["upper_ratio"], range=[0, 100]),
        dict(label="Tag Count", values=plot_df["tag_count"],
             range=[0, max(1, float(plot_df["tag_count"].max()))]),
        dict(label="Cluster", values=plot_df["cluster_id"],
             range=[0, max(1, n_clusters - 1)],
             tickvals=list(range(n_clusters)),
             ticktext=[f"C{i + 1}" for i in range(n_clusters)]),
    ]

    fig = go.Figure(go.Parcoords(
        line=dict(color=plot_df["cluster_id"], colorscale=colorscale,
                  showscale=False, cmin=-0.5, cmax=n_clusters - 0.5),
        dimensions=dimensions,
        domain=dict(x=[0.04, 0.98], y=[0.055, 0.66]),
        labelfont=dict(size=13, color="#f8fafc"),
        tickfont=dict(size=11, color="#cbd5e1"),
        rangefont=dict(size=1, color="rgba(0,0,0,0)"),
    ))

    # The legend is ABOVE the axes in its own reserved area.
    # It does not compete with axis titles or take extra space below the chart.
    # Two columns keep long cluster names readable on desktop.
    for cid in range(n_clusters):
        row, col = divmod(cid, 2)
        fig.add_annotation(
            x=0.025 + 0.50 * col, y=0.905 - 0.08 * row,
            xref="paper", yref="paper", showarrow=False,
            text=f"● {archetype_map[cid]}",
            xanchor="left", yanchor="middle", align="left",
            font=dict(size=13, color=colors[cid]),
        )

    rows = (n_clusters + 1) // 2
    # Preserve a gap between the bottom legend row and axis labels,
    # including the optional fifth and sixth clusters.
    if rows > 2:
        fig.data[0].domain.y = [0.055, 0.66 - (rows - 2) * 0.08]
    fig.update_layout(
        title=dict(text="Video Cluster Profiles", font=dict(size=17), x=0.01, y=0.995,
                   xanchor="left", yanchor="top"),
        height=550 + 24 * max(0, rows - 2),
        margin=dict(l=65, r=25, t=65, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#e2e8f0"),
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
                title=f"PC1 ({var_ratios[0]:.1f}%)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                gridcolor="#334155",
                showbackground=True,
            ),
            yaxis=dict(
                title=f"PC2 ({var_ratios[1]:.1f}%)",
                backgroundcolor="rgba(15, 23, 42, 0.45)",
                gridcolor="#334155",
                showbackground=True,
            ),
            zaxis=dict(
                title=f"PC3 ({var_ratios[2]:.1f}%)",
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
    country_code: str | None = None,
    category: str | None = None,
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
                text="Feature Importance",
                font=dict(size=14),
                y=0.89,
                x=0,
                xanchor="left",
            ),
            xaxis_title="Importance (%)",
            yaxis_title="",
            height=520,
            margin=dict(l=10, r=15, t=85, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="sans-serif", size=12),
        )
        return fig

    rf_clf = rf_res["rf_clf"]
    ml_df = rf_res["ml_df"]
    X_cols = rf_res["X_columns"]
    # The decision surface uses valid category/country flags and a real video baseline.
    median_eng = rf_res["median_eng"]

    countries = sorted(ml_df["country_code"].dropna().unique().tolist())
    if not countries:
        raise ValueError("No countries available for the Random Forest surface.")
    country_code = country_code or countries[0]

    country_rows = ml_df[ml_df["country_code"] == country_code]
    if country_rows.empty:
        raise ValueError(f"No data available for country: {country_code}")

    category = category or country_rows["category"].mode().iloc[0]
    matching_rows = country_rows[country_rows["category"] == category]
    if matching_rows.empty:
        raise ValueError(f"No videos for {country_code} / {category}")

    # Choose a real reference video close to typical metadata values.
    numeric_cols = list(RF_CONTINUOUS_FEATURES.keys())
    numeric_values = matching_rows[numeric_cols].astype(float)
    medians = numeric_values.median()
    scales = (numeric_values.quantile(0.75) - numeric_values.quantile(0.25)).replace(0, 1)
    distances = (((numeric_values - medians) / scales) ** 2).sum(axis=1)
    reference_video = matching_rows.iloc[int(np.argmin(distances.to_numpy()))]

    x_min, x_max = np.percentile(matching_rows[x_feat], [1, 99])
    y_min, y_max = np.percentile(matching_rows[y_feat], [1, 99])
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    x_grid = np.linspace(x_min, x_max, 42)
    y_grid = np.linspace(y_min, y_max, 42)
    xx, yy = np.meshgrid(x_grid, y_grid)

    # One-hot columns must describe exactly one real country and one category.
    reference_features = {col: 0.0 for col in X_cols}
    for feat in numeric_cols:
        reference_features[feat] = float(reference_video[feat])

    reference_features[f"market_{country_code}"] = 1
    reference_features[f"category_{category}"] = 1

    grid_df = pd.DataFrame([reference_features] * xx.size, columns=X_cols)
    grid_df[x_feat] = xx.ravel()
    grid_df[y_feat] = yy.ravel()

    z_prob = rf_clf.predict_proba(grid_df[X_cols])[:, 1].reshape(xx.shape) * 100.0

    # Overlay observed videos from the same country and category only.
    sample_n = min(140, len(matching_rows))
    sample_df = matching_rows.sample(n=sample_n, random_state=42).copy()
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
                title=dict(text="High Engagement (%)", font=dict(size=11)),
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
            text=f"RF High-Engagement Probability: {country_code} / {category}",
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
