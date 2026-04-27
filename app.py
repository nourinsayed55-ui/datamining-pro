"""
Customer Segmentation Dashboard
Data Mining — AIE323 | Alamein University
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, io

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
)

# ─── Helpers ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

@st.cache_data
def load_data():
    """Load or generate customer dataset."""
    if os.path.exists("Mall_Customers.csv"):
        df = pd.read_csv("Mall_Customers.csv")
    else:
        np.random.seed(RANDOM_STATE)
        n = 300
        gender = np.random.choice(['Male', 'Female'], n)
        age    = np.random.randint(18, 71, n)
        income = np.random.randint(15, 138, n)
        score  = np.random.randint(1, 100, n)
        for i in range(n):
            if income[i] > 80 and age[i] < 40:
                score[i] = int(np.clip(score[i] + np.random.randint(20, 40), 1, 99))
            if income[i] < 40:
                score[i] = int(np.clip(score[i] - np.random.randint(0, 20), 1, 99))
        df = pd.DataFrame({
            'CustomerID': range(1, n + 1),
            'Gender':     gender,
            'Age':        age,
            'Annual Income (k$)':       income,
            'Spending Score (1-100)':   score,
        })
    return df


def preprocess(df):
    le = LabelEncoder()
    out = df.copy()
    out['Gender_enc'] = le.fit_transform(out['Gender'])
    feature_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(out[feature_cols])
    return X_scaled, scaler, feature_cols, out


def run_clustering(X_scaled, algo, n_clusters, eps=0.5, min_samples=5):
    if algo == "K-Means":
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    elif algo == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    else:  # DBSCAN
        model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return labels, model


def get_metrics(X, labels):
    mask = labels != -1
    Xm, lm = X[mask], labels[mask]
    if len(set(lm)) < 2:
        return None
    return {
        'Silhouette':        round(silhouette_score(Xm, lm), 4),
        'Davies-Bouldin':    round(davies_bouldin_score(Xm, lm), 4),
        'Calinski-Harabasz': round(calinski_harabasz_score(Xm, lm), 2),
    }


def assign_persona(inc, sco):
    if inc >= 70 and sco >= 60: return '💎 High-Value Champion'
    if inc >= 70 and sco <  50: return '💰 Cautious High-Earner'
    if inc <  50 and sco >= 60: return '🛒 Budget Enthusiast'
    if inc <  50 and sco <  50: return '🔻 At-Risk Low-Spender'
    return '🌟 Mid-Tier Regular'


# ─── Load Data ──────────────────────────────────────────────────────────────
df_raw = load_data()
X_scaled, scaler, feature_cols, df_proc = preprocess(df_raw)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca2.fit_transform(X_scaled)

# ─── Sidebar Controls ───────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Alamein_International_University_logo.png/200px-Alamein_International_University_logo.png",
                 width=120, use_container_width=False)
st.sidebar.title("⚙️ Controls")

algo       = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "Agglomerative", "DBSCAN"])
n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 5)
eps        = st.sidebar.slider("DBSCAN — Epsilon", 0.1, 2.0, 0.5, 0.05)
min_samp   = st.sidebar.slider("DBSCAN — Min Samples", 2, 20, 5)
x_axis     = st.sidebar.selectbox("X-axis Feature", feature_cols, index=1)
y_axis     = st.sidebar.selectbox("Y-axis Feature", feature_cols, index=2)

labels, model = run_clustering(X_scaled, algo, n_clusters, eps, min_samp)
df_proc['Cluster'] = labels
metrics = get_metrics(X_scaled, labels)

# ─── Header ─────────────────────────────────────────────────────────────────
st.title("🛍️ Customer Segmentation Dashboard")
st.caption("Data Mining Course — AIE323 | Alamein University | Project 1")

# ─── KPI Row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
k1.metric("👥 Customers",    len(df_raw))
k2.metric("🔢 Clusters Found", n_clusters_found)
k3.metric("📊 Silhouette",   metrics['Silhouette'] if metrics else "N/A")
k4.metric("📉 Davies-Bouldin", metrics['Davies-Bouldin'] if metrics else "N/A")
k5.metric("📈 Calinski-H.",  metrics['Calinski-Harabasz'] if metrics else "N/A")

st.divider()

# ─── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Cluster View", "📊 Profiles", "🔍 Customer Lookup", "📤 Upload & Predict", "📈 EDA"
])

# ════ TAB 1 — Cluster Scatter ════════════════════════════════════════════════
with tab1:
    st.subheader(f"Cluster Visualization — {algo}")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            df_proc[x_axis], df_proc[y_axis],
            c=labels, cmap='tab10', alpha=0.75, edgecolors='white', s=55
        )
        ax.set_xlabel(x_axis); ax.set_ylabel(y_axis)
        ax.set_title(f'{algo} — {x_axis} vs {y_axis}')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)

    with col_b:
        st.subheader("PCA 2D Projection")
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10',
                    alpha=0.75, edgecolors='white', s=55)
        ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
        ax2.set_title('PCA Projection')
        st.pyplot(fig2)

    # Cluster sizes
    st.subheader("Cluster Sizes")
    size_df = pd.Series(labels[labels != -1]).value_counts().sort_index().reset_index()
    size_df.columns = ['Cluster', 'Count']
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.bar(size_df['Cluster'].astype(str), size_df['Count'],
            color=plt.cm.tab10(np.linspace(0, 0.9, len(size_df))), edgecolor='white')
    ax3.set_xlabel('Cluster'); ax3.set_ylabel('Number of Customers')
    ax3.set_title('Customer Count per Cluster')
    st.pyplot(fig3)

# ════ TAB 2 — Cluster Profiles ═══════════════════════════════════════════════
with tab2:
    st.subheader("Customer Segment Profiles")
    num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    profile  = df_proc[df_proc['Cluster'] != -1].groupby('Cluster')[num_cols].mean().round(1)
    profile['Persona'] = profile.apply(
        lambda r: assign_persona(r['Annual Income (k$)'], r['Spending Score (1-100)']), axis=1
    )
    st.dataframe(profile, use_container_width=True)

    # Bar comparison
    st.subheader("Feature Comparison across Clusters")
    selected_feat = st.selectbox("Select Feature", num_cols)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    clusters_sorted = profile.index.tolist()
    vals = [profile.loc[c, selected_feat] for c in clusters_sorted]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(clusters_sorted)))
    ax4.bar([str(c) for c in clusters_sorted], vals, color=colors, edgecolor='white')
    ax4.set_xlabel('Cluster'); ax4.set_ylabel(f'Mean {selected_feat}')
    ax4.set_title(f'Mean {selected_feat} per Cluster')
    st.pyplot(fig4)

    # Radar chart
    st.subheader("Radar Chart")
    radar_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    radar_df   = df_proc[df_proc['Cluster'] != -1].groupby('Cluster')[radar_cols].mean()
    radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
    angles     = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles    += angles[:1]
    fig5, ax5 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    clr_list   = plt.cm.tab10(np.linspace(0, 0.9, len(radar_norm)))
    for (idx, row), color in zip(radar_norm.iterrows(), clr_list):
        vals_r = row.tolist() + [row.tolist()[0]]
        ax5.plot(angles, vals_r, color=color, linewidth=2, label=f'Cluster {idx}')
        ax5.fill(angles, vals_r, color=color, alpha=0.1)
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(radar_cols, fontsize=10)
    ax5.set_title('Cluster Radar Chart', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
    st.pyplot(fig5)

# ════ TAB 3 — Customer Lookup ════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Customer Lookup")
    customer_id = st.number_input("Enter Customer ID", min_value=int(df_raw['CustomerID'].min()),
                                   max_value=int(df_raw['CustomerID'].max()), value=1, step=1)
    if st.button("Find Segment"):
        row = df_proc[df_proc['CustomerID'] == customer_id]
        if row.empty:
            st.error("Customer ID not found.")
        else:
            r = row.iloc[0]
            cluster_id = int(r['Cluster'])
            persona    = assign_persona(r['Annual Income (k$)'], r['Spending Score (1-100)'])
            st.success(f"Customer {customer_id} → **Cluster {cluster_id}** | Persona: {persona}")
            st.table(r[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].to_frame().T)

# ════ TAB 4 — Upload & Predict ═══════════════════════════════════════════════
with tab4:
    st.subheader("📤 Upload New Customer Data & Get Predictions")
    st.info("Upload a CSV with columns: Gender, Age, Annual Income (k$), Spending Score (1-100)")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        st.write("Preview:", new_df.head())
        try:
            le2 = LabelEncoder()
            new_df['Gender_enc'] = le2.fit_transform(new_df['Gender'])
            X_new = scaler.transform(new_df[feature_cols])
            new_labels, _ = run_clustering(X_new, algo, n_clusters, eps, min_samp)
            new_df['Predicted_Cluster'] = new_labels
            new_df['Persona'] = new_df.apply(
                lambda r: assign_persona(r['Annual Income (k$)'], r['Spending Score (1-100)']), axis=1
            )
            st.success("✅ Predictions complete!")
            st.dataframe(new_df, use_container_width=True)
            csv_out = new_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ════ TAB 5 — EDA ════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Feature Distributions**")
        feat_sel = st.selectbox("Feature", ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        ax6.hist(df_raw[feat_sel], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax6.set_title(f'Distribution of {feat_sel}')
        ax6.set_xlabel(feat_sel); ax6.set_ylabel('Count')
        st.pyplot(fig6)

    with col2:
        st.write("**Correlation Heatmap**")
        num_raw = df_raw[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        sns.heatmap(num_raw.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    ax=ax7, linewidths=0.5, vmin=-1, vmax=1)
        ax7.set_title('Correlation Heatmap')
        st.pyplot(fig7)

    st.write("**Gender Distribution**")
    gender_counts = df_raw['Gender'].value_counts()
    fig8, ax8 = plt.subplots(figsize=(4, 4))
    ax8.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['steelblue', 'coral'], startangle=90)
    ax8.set_title('Gender Split')
    st.pyplot(fig8)
