import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from textblob import TextBlob
from datetime import datetime, timedelta

# Setup halaman
st.set_page_config(page_title="Sentiment Radar", layout="wide", page_icon="📡")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-dark: #0a0a0c;
    --neon-blue: #0088ff;
    --neon-red: #ff3366;
    --neon-green: #00ffcc;
    --text-dim: #777788;
}

.stApp {
    background-color: var(--bg-dark);
    color: white;
    font-family: 'Space Mono', monospace;
}

.radar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    color: var(--neon-blue);
    text-shadow: 0 0 15px rgba(0, 136, 255, 0.3);
    margin-bottom: 0;
}

.stat-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid var(--neon-blue);
    text-align: center;
}

.stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.5rem;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Init session
if 'tweet_data' not in st.session_state:
    st.session_state.tweet_data = pd.DataFrame(columns=['Waktu', 'User', 'Tweet', 'Polarity', 'Sentimen'])

# Mock database
KUMPULAN_TWEET = [
    "What a brilliant goal! Unbelievable!",
    "The referee is absolutely blind today.",
    "Boring match, passing is too slow.",
    "Messi is the undeniable GOAT! Magic!",
    "Terrible defense. They look lost.",
    "Great save by the keeper! Wow!",
    "I can't believe they missed that penalty.",
    "Solid tactical setup by the manager.",
    "This game is putting me to sleep.",
    "Incredible atmosphere in the stadium tonight!"
]

# Fungsi NLP
def analyze_sentiment(text):
    # Hitung skor
    score = TextBlob(text).sentiment.polarity
    # Tentukan kategori
    if score > 0.1: return score, 'Positif'
    elif score < -0.1: return score, 'Negatif'
    else: return score, 'Netral'

# Fungsi generator
def generate_mock_tweets(n=5):
    new_rows = []
    # Loop data
    for _ in range(n):
        now = datetime.now().strftime("%H:%M:%S")
        user = f"@user_{np.random.randint(1000, 9999)}"
        text = np.random.choice(KUMPULAN_TWEET)
        
        # Eksekusi NLP
        score, label = analyze_sentiment(text)
        new_rows.append({'Waktu': now, 'User': user, 'Tweet': text, 'Polarity': score, 'Sentimen': label})
    
    return pd.DataFrame(new_rows)

# Header UI
st.markdown('<h1 class="radar-title">📡 LIVE SENTIMENT RADAR</h1>', unsafe_allow_html=True)
st.markdown("<span style='color: #777788;'>NLP Engine: TextBlob | Source: X/Twitter Stream (Simulated)</span>", unsafe_allow_html=True)
st.divider()

# Tombol stream
col_btn, _, _ = st.columns([1, 2, 2])
with col_btn:
    if st.button(" Tarik Stream Baru", use_container_width=True):
        # Update data
        new_tweets = generate_mock_tweets(np.random.randint(3, 8))
        st.session_state.tweet_data = pd.concat([st.session_state.tweet_data, new_tweets], ignore_index=True)

df = st.session_state.tweet_data

# Cek isi
if not df.empty:
    # Hitung metrik
    total = len(df)
    pos = len(df[df['Sentimen'] == 'Positif'])
    neg = len(df[df['Sentimen'] == 'Negatif'])
    
    # Layout metrik
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="stat-box" style="border-color: #0088ff;"><div class="stat-num">{total}</div>Total Tweets</div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="stat-box" style="border-color: #00ffcc;"><div class="stat-num">{pos}</div>Positif</div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-box" style="border-color: #ff3366;"><div class="stat-num">{neg}</div>Negatif</div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Layout chart
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("###  Timeline Opini Publik")
        
        # Hitung frekuensi
        timeline = df.groupby(['Waktu', 'Sentimen']).size().reset_index(name='Jumlah')
        
        # Plot garis
        fig = px.line(timeline, x='Waktu', y='Jumlah', color='Sentimen', markers=True,
                      color_discrete_map={'Positif': '#00ffcc', 'Negatif': '#ff3366', 'Netral': '#aaaaaa'})
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#777788')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.markdown("### 💬 Live Feed")
        # Tabel feed
        st.dataframe(df[['User', 'Sentimen', 'Tweet']].tail(10).iloc[::-1], height=400, use_container_width=True)
else:
    # State kosong
    st.info("Menunggu sinyal... Klik 'Tarik Stream Baru' untuk mulai memantau sentimen publik.")