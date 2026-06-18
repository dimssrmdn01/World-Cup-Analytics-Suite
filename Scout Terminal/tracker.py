import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Setup halaman
st.set_page_config(page_title="Scout Terminal", layout="wide", page_icon="🔭")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-deep: #050505;
    --card-bg: rgba(20, 20, 25, 0.7);
    --neon-cyan: #00f0ff;
    --neon-purple: #b026ff;
    --neon-green: #00ff66;
    --text-main: #e0e0e0;
    --text-dim: #888888;
    --border-glass: rgba(255, 255, 255, 0.1);
}

.stApp {
    background: radial-gradient(circle at top left, rgba(0, 240, 255, 0.05), transparent 40%),
                radial-gradient(circle at bottom right, rgba(176, 38, 255, 0.05), transparent 40%),
                var(--bg-deep);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

.scout-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--neon-cyan);
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
}

.scout-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    color: white;
    letter-spacing: 1px;
    line-height: 1.1;
    margin: 0;
    text-transform: uppercase;
    text-shadow: 0 0 20px rgba(0, 240, 255, 0.15);
}

.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    border-top: 3px solid var(--neon-cyan);
    transition: transform 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
}

.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    color: white;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1px;
}

hr {
    border-color: var(--border-glass);
    margin: 2rem 0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(10, 10, 12, 0.95);
    border-right: 1px solid var(--border-glass);
}
</style>
""", unsafe_allow_html=True)

# Cache data
@st.cache_data
def load_data():
    # Data riil
    data = [
        ['E. Haaland', 36, 8, 75.5, 5, 11.0],
        ['K. Mbappe', 29, 10, 80.2, 8, 12.5],
        ['R. Lewandowski', 23, 7, 77.0, 10, 10.5],
        ['H. Kane', 30, 9, 78.5, 12, 11.5],
        ['V. Osimhen', 26, 4, 73.0, 9, 11.0],
        ['L. Messi', 16, 16, 85.0, 10, 10.0],
        ['K. De Bruyne', 7, 16, 88.5, 15, 11.5],
        ['M. Odegaard', 15, 7, 86.0, 25, 12.5],
        ['B. Fernandes', 8, 14, 82.0, 35, 13.0],
        ['V. Junior', 15, 11, 81.0, 18, 11.5],
        ['N. Kante', 2, 3, 87.0, 65, 14.0],
        ['Rodri', 7, 9, 93.0, 55, 13.0],
        ['J. Bellingham', 14, 5, 85.5, 45, 13.5],
        ['D. Rice', 4, 2, 89.0, 60, 13.0],
        ['F. Valverde', 7, 5, 86.5, 40, 13.5],
        ['V. van Dijk', 3, 1, 91.0, 45, 10.5],
        ['R. Dias', 1, 0, 93.5, 50, 10.0],
        ['W. Saliba', 2, 1, 92.0, 48, 10.5],
        ['A. Rudiger', 2, 0, 88.0, 55, 11.0],
        ['K. Walker', 0, 3, 87.5, 42, 12.0],
        ['Bukayo Saka', 14, 11, 82.0, 25, 12.0],
        ['Son Heung-min', 17, 10, 81.5, 15, 11.5],
        ['L. Modric', 4, 6, 89.5, 20, 11.0],
        ['Casemiro', 7, 3, 83.0, 70, 12.0],
        ['Marquinhos', 2, 1, 94.0, 40, 10.0],
        ['T. Alexander-Arnold', 3, 12, 81.5, 38, 11.5],
        ['A. Robertson', 1, 8, 83.0, 45, 12.0],
        ['A. Hakimi', 5, 6, 85.0, 40, 12.5],
        ['Pedri', 3, 6, 90.0, 28, 11.5],
        ['Gavi', 2, 4, 88.5, 45, 12.0]
    ]
    # Buat dataframe
    df = pd.DataFrame(data, columns=['Player', 'Goals', 'Assists', 'Pass_Acc', 'Tackles', 'Stamina'])
    return df

# Load dataframe
df = load_data()

# Sidebar settings
st.sidebar.markdown("<h2 style='font-family: Bebas Neue; color: #00f0ff; font-size: 2rem;'>⚙️ PARAMETER SYSTEM</h2>", unsafe_allow_html=True)
k_clusters = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 4)

# Ekstrak fitur
features = ['Goals', 'Assists', 'Pass_Acc', 'Tackles', 'Stamina']
X = df[features]

# Standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Jalankan K-Means
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Format label
df['Cluster'] = "Tipe " + df['Cluster'].astype(str)

# Reduksi dimensi
pca = PCA(n_components=2)
pca_res = pca.fit_transform(X_scaled)
df['PCA1'] = pca_res[:, 0]
df['PCA2'] = pca_res[:, 1]

# Header UI
st.markdown('<div class="scout-eyebrow"> ADVANCED SCOUTING TERMINAL </div>', unsafe_allow_html=True)
st.markdown('<h1 class="scout-title">Player Performance &<br>Clustering Matrix</h1>', unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Top metrics
m1, m2, m3 = st.columns(3)

with m1:
    # Metrik pemain
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(df)}</div><div class="stat-label">Total Subjek Dianalisis</div></div>', unsafe_allow_html=True)
    
with m2:
    # Metrik cluster
    st.markdown(f'<div class="stat-card" style="border-top-color: var(--neon-purple);"><div class="stat-value">{k_clusters}</div><div class="stat-label">Kategori Gaya Bermain</div></div>', unsafe_allow_html=True)
    
with m3:
    # Metrik algoritma
    st.markdown('<div class="stat-card" style="border-top-color: var(--neon-green);"><div class="stat-value">K-MEANS</div><div class="stat-label">Mesin Clustering Aktif</div></div>', unsafe_allow_html=True)

st.markdown('<br><br>', unsafe_allow_html=True)

# Layout utama
col1, col2 = st.columns([6, 4], gap="large")

with col1:
    # Header visual
    st.markdown("<h3 style='font-family: Bebas Neue; letter-spacing: 1px; color: white;'> Peta Galaksi Pemain (PCA Projection)</h3>", unsafe_allow_html=True)
    
    # Plotly custom
    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='Cluster',
        hover_name='Player', hover_data=features,
        color_discrete_sequence=['#00f0ff', '#b026ff', '#00ff66', '#ff0055', '#ffaa00', '#ffffff'],
    )
    
    # Tema gelap
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#888888',
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Grid transparan
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False, title="")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False, title="")
    
    # Render plot
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Header tabel
    st.markdown("<h3 style='font-family: Bebas Neue; letter-spacing: 1px; color: white;'> Database Subjek</h3>", unsafe_allow_html=True)
    
    # Render tabel
    st.dataframe(
        df[['Player', 'Cluster'] + features],
        use_container_width=True,
        height=450
    )