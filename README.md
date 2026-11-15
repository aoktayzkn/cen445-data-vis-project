# 📊 YouTube Trending Video Analytics Dashboard

An interactive, multi-module data visualization and machine learning web application built with **Streamlit**, **Plotly**, **Scikit-Learn**, and **NetworkX** for the **CEN445 Data Visualization** term project. The dashboard explores cross-country YouTube trending dynamics across **25,000+ videos** from five major global markets (**US, GB, DE, FR, RU**).

---

## 👥 Team Members & Module Ownership

The dashboard is architected around a unified **1 Top + 2 Bottom symmetric grid layout** with persistent sidebar filtering, `@st.cache_data` and `@st.cache_resource` high-speed state caching, KPI summary cards, and expandable technical documentation (`st.expander`).

| Module / Tab | Team Member | Core Responsibilities & Visuals |
| :--- | :--- | :--- |
| **1. Exploratory Data Analysis (EDA)** | **Fatih AKSÜT** | Multi-country dataset ingestion, 99th-percentile outlier filtering, category JSON mapping, Distribution Histogram with Marginal Boxplots, Pearson Correlation Matrix, and Category-to-Channel Hierarchical Treemap (`uye1_veri_isleme.py`, `veri_hazirla.py`). |
| **2. Machine Learning Models** | **Yusuf YALÇIN** | Unsupervised behavioral segmentation (**K-Means Clustering + 3D PCA Manifold**), Multi-Dimensional Parallel Coordinates with interactive brushing, and supervised non-linear virality prediction (**Random Forest Classifier & Regressor** with 2D response surfaces and feature importance diagnostics). Includes global `@st.cache_resource` implementation for real-time model evaluation (`src/ml_models.py`). |
| **3. Dashboard Architecture & Advanced Visualizations** | **Ali Oktay ÖZKAN** | Core Streamlit layout orchestration, sidebar filter pipeline, `@st.cache_data` optimization, Multi-Metric Northern-Hemisphere Choropleth Map, Velocity-to-Engagement Sankey Flow Diagram, and Graph-Theoretic Tag Co-occurrence Network with Degree & Betweenness Centrality diagnostics (`app.py`, `src/visualizations.py`). |

---

## ✨ Key Features & Interactive Capabilities

### Global Sidebar Controls
* **Country Multiselect:** Dynamically filter any combination of **United States (`US`)**, **Great Britain (`GB`)**, **Germany (`DE`)**, **France (`FR`)**, and **Russia (`RU`)**.
* **View Count Range Slider:** Isolate specific audience reach tiers across the 99th-percentile cleaned dataset.
* **Fast-Trending Velocity Toggle (`<3 Days`):** Filter videos that achieved trending status within 48 hours of publication.

### Module 1 — Exploratory Data Analysis (Fatih AKSÜT)
1. **Distribution Histogram & Marginal Boxplot:** Compares right-skewed `Views` and `Likes` distributions across countries while highlighting medians and interquartile ranges (IQR).
2. **Feature Correlation Heatmap:** Evaluates linear relationships across `Views`, `Likes`, `Comments`, `Title Length`, and `Uppercase Character Count`.
3. **Hierarchical Category & Channel Treemap:** Maps proportional view volume across content categories down to top-performing creator channels.

### Module 2 — Machine Learning & Behavioral Clustering (Yusuf YALÇIN)
1. **Interactive ML Controls & Live Evaluation KPIs:** Adjust K-Means cluster count ($k \in [2, 6]$), select Random Forest decision surface axes, toggle between *2D Decision Surface* and *Feature Importance*, and monitor real-time **Silhouette Score**, **3D PCA Variance Captured (%)**, **RF Classifier ROC-AUC**, and **RF Regressor $R^2$**.
2. **Parallel Coordinates Plot:** Visualizes 7-dimensional behavioral cluster signatures across log-views, engagement rate, comment intensity, days to trend, title length, uppercase ratio, and tag count.
3. **3D PCA Cluster Manifold & Random Forest Virality Surface:** Projects standardized features onto the top 3 orthogonal Principal Components alongside a non-linear contour response surface predicting high-engagement virality.

### Module 3 — Geospatial, Flow & Network Analytics (Ali Oktay ÖZKAN)
1. **Interactive Graph/Geo Controls & Analytical KPIs:** Switch geospatial metrics (`Average Views`, `Engagement Rate %`, `Comment Intensity per 1K`, `Trending Velocity Days`), adjust Sankey category depth (`Top 5–10`), control Network node count (`15–35`), and scale graph nodes by **Tag Frequency**, **Degree Centrality** ($C_D(v)$), or **Betweenness Centrality** ($C_B(v)$).
2. **Multi-Metric Geospatial Choropleth Map:** Natural Earth Northern-Hemisphere projection with custom 6-metric hover tooltips and dynamic color scales.
3. **Categorical Velocity-to-Engagement Sankey Diagram:** Traces how dominant content categories transition into *Fast Trend (<3d)* vs. *Slow Trend (3d+)* regimes and convert into *High* vs. *Standard Engagement*.
4. **Semantic Tag Co-occurrence Network Graph:** Built with NetworkX **Fruchterman-Reingold** force-directed layout (`k=0.85`), coloring nodes by dominant video category and computing graph density ($D = \frac{2E}{V(V-1)}$) and shortest-path bridge tags.

---

## 📂 Repository Structure

```text
├── app.py                  # Main Streamlit dashboard entry point, caching & layout orchestration
├── uye1_veri_isleme.py     # Data cleaning pipeline & Module 1 EDA visualizations
├── veri_hazirla.py         # Raw multi-country CSV & JSON category preparation script
├── src/
│   ├── visualizations.py   # Module 3 Choropleth map, Sankey diagram & NetworkX graph functions
│   └── ml_models.py        # Module 2 K-Means, 3D PCA, and Random Forest pipelines & charts
├── data/                   # Cleaned and raw multi-country YouTube datasets (US, GB, DE, FR, RU)
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## 🚀 Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/aoktayzkn/cen445-data-vis-project.git
cd cen445-data-vis-project
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
The application will open automatically in your default browser at `http://localhost:8501`.
