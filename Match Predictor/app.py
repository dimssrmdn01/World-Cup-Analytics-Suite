import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@st.cache_resource
def train_model():
    np.random.seed(42)

    team_power = {
        'Argentina': 90, 'France': 89, 'Brazil': 89, 'England': 87,
        'Spain': 86, 'Germany': 85, 'Portugal': 85, 'Belgium': 84,
        'Netherlands': 84, 'Croatia': 83, 'Uruguay': 82, 'Denmark': 81,
        'Switzerland': 80, 'Morocco': 80, 'Japan': 79, 'Senegal': 78,
        'Serbia': 78, 'Mexico': 78, 'USA': 77, 'Poland': 77,
        'South Korea': 77, 'Ecuador': 76, 'Cameroon': 75, 'Canada': 75,
        'Wales': 75, 'Iran': 74, 'Ghana': 74, 'Australia': 74,
        'Tunisia': 73, 'Costa Rica': 72, 'Saudi Arabia': 71, 'Qatar': 70
    }

    teams = list(team_power.keys())
    data = []

    for _ in range(5000):
        home, away = np.random.choice(teams, 2, replace=False)
        power_diff = (team_power[home] + 5) - team_power[away]
        home_score = np.random.poisson(max(0.5, 1.5 + (power_diff * 0.05)))
        away_score = np.random.poisson(max(0.5, 1.2 - (power_diff * 0.05)))
        result = 2 if home_score > away_score else (1 if home_score == away_score else 0)
        data.append([team_power[home], team_power[away], power_diff, result])

    df = pd.DataFrame(data, columns=['home_power', 'away_power', 'power_diff', 'result'])
    X = df[['home_power', 'away_power', 'power_diff']]
    y = df['result']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, team_power


model, scaler, team_power = train_model()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_initials(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper()


def confidence_label(p_max: float) -> str:
    if p_max >= 0.60:
        return "Keyakinan Tinggi"
    if p_max >= 0.45:
        return "Keyakinan Sedang"
    return "Hasil Ketat"


def generate_analysis(home, away, home_r, away_r, p_home, p_draw, p_away) -> str:
    diff = home_r - away_r

    if abs(diff) <= 3:
        rating_note = (
            f"Rating power kedua tim cuma beda {abs(diff)} poin, jadi hasilnya "
            f"kemungkinan besar ditentukan momentum di lapangan, bukan angka di atas kertas."
        )
    elif diff > 0:
        rating_note = (
            f"{home} datang dengan rating power {home_r}, unggul {diff} poin "
            f"dibanding {away} ({away_r}), ditambah keuntungan main di kandang."
        )
    else:
        rating_note = (
            f"Meski bermain di kandang, rating power {home} ({home_r}) masih "
            f"tertinggal {abs(diff)} poin dari {away} ({away_r})."
        )

    leader = home if p_home >= p_away else away
    other = away if leader == home else home
    leader_p = max(p_home, p_away)

    if p_draw >= 0.20 and abs(p_home - p_away) < 0.15:
        verdict = "Model menilai laga ini cukup terbuka, peluang seri juga tidak kecil."
    elif leader_p >= 0.55:
        verdict = f"{leader} diunggulkan model untuk membawa pulang tiga poin."
    else:
        verdict = f"{leader} sedikit di atas, tapi ruang bagi {other} untuk membuat kejutan masih terbuka."

    return f"{rating_note} {verdict}"


# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --pitch-deep: #0a2419;
    --pitch: #102e22;
    --panel: #14392a;
    --chalk: #f3efe3;
    --chalk-dim: #9fb8ac;
    --gold: #dba43c;
    --gold-soft: rgba(219, 164, 60, 0.16);
    --home: #c4453a;
    --home-soft: rgba(196, 69, 58, 0.14);
    --away: #3f7aa8;
    --away-soft: rgba(63, 122, 168, 0.14);
    --line: rgba(243, 239, 227, 0.12);
}

#MainMenu, footer { visibility: hidden; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(219,164,60,0.06), transparent 45%),
        radial-gradient(circle at 90% 10%, rgba(63,122,168,0.05), transparent 40%),
        var(--pitch-deep);
    color: var(--chalk);
    font-family: 'Inter', sans-serif;
}

.block-container { padding-top: 2.5rem; max-width: 760px; }

/* ---- Header ---- */
.eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    color: var(--gold);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.eyebrow .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 0 0 rgba(219,164,60,0.6);
    animation: pulse 2.4s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(219,164,60,0.5); }
    70%  { box-shadow: 0 0 0 6px rgba(219,164,60,0); }
    100% { box-shadow: 0 0 0 0 rgba(219,164,60,0); }
}
.page-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.1rem;
    line-height: 1.05;
    letter-spacing: 0.01em;
    margin: 0 0 0.8rem 0;
    color: var(--chalk);
}
.page-sub {
    color: var(--chalk-dim);
    font-size: 0.97rem;
    max-width: 540px;
    margin-bottom: 1.6rem;
}
.hr-line {
    border: none;
    border-top: 1px solid var(--line);
    margin: 1.6rem 0;
}

/* ---- Side labels ---- */
.side-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    margin-bottom: 0.5rem;
}
.home-label { color: var(--home); }
.away-label { color: var(--away); }

/* ---- Panels (st.container border=True) ---- */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--home-soft) !important;
    background: linear-gradient(180deg, var(--home-soft), transparent 60%);
    border-radius: 14px;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--away-soft) !important;
    background: linear-gradient(180deg, var(--away-soft), transparent 60%);
    border-radius: 14px;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
}

/* ---- Select boxes ---- */
div[data-baseweb="select"] > div {
    background-color: rgba(0,0,0,0.18) !important;
    border-color: var(--line) !important;
    border-radius: 9px !important;
}

/* ---- Sliders ---- */
div[data-testid="stSlider"] { padding-top: 0.4rem; }

/* ---- Predict button ---- */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold), #b9842c);
    color: #1a1206;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 0;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(219,164,60,0.25);
    color: #1a1206;
}

/* ---- VS card ---- */
.vs-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin: 1.6rem 0;
}
.vs-side { flex: 1; text-align: center; }
.vs-badge {
    width: 56px; height: 56px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    margin: 0 auto 0.5rem auto;
    border: 2px solid;
}
.vs-badge.home { border-color: var(--home); color: var(--home); background: var(--home-soft); }
.vs-badge.away { border-color: var(--away); color: var(--away); background: var(--away-soft); }
.vs-name { font-weight: 600; font-size: 0.95rem; }
.vs-rating {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--chalk-dim);
    margin-top: 0.15rem;
}
.vs-mid {
    display: flex; align-items: center; gap: 0.5rem;
    flex: 0 0 auto;
}
.vs-line { width: 22px; height: 1px; background: var(--line); }
.vs-circle {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.85rem;
    color: var(--gold);
    border: 1px solid var(--gold-soft);
    border-radius: 50%;
    width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
}

/* ---- Results ---- */
.result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    color: var(--chalk-dim);
    margin-bottom: 0.3rem;
}
.result-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.7rem;
    margin-bottom: 0.9rem;
}
.chip {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}
.chip.home { background: var(--home-soft); color: #e8847b; }
.chip.away { background: var(--away-soft); color: #82b3da; }
.chip.tie  { background: var(--gold-soft); color: var(--gold); }

.meter-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
    color: var(--chalk-dim);
}
.meter-labels strong { color: var(--chalk); }
.meter-track {
    display: flex;
    width: 100%;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.04);
}
.meter-seg { height: 100%; transition: width 0.6s ease; }
.meter-seg.home { background: var(--home); }
.meter-seg.draw { background: var(--gold); }
.meter-seg.away { background: var(--away); }

.analysis-card {
    margin-top: 1.3rem;
    padding: 1rem 1.1rem;
    border-left: 3px solid var(--gold);
    background: rgba(255,255,255,0.03);
    border-radius: 0 10px 10px 0;
    font-size: 0.93rem;
    color: var(--chalk-dim);
    line-height: 1.55;
}

.empty-state {
    margin-top: 1.6rem;
    padding: 1.4rem;
    border: 1px dashed var(--line);
    border-radius: 12px;
    text-align: center;
    color: var(--chalk-dim);
    font-size: 0.9rem;
}

@media (max-width: 640px) {
    .page-title { font-size: 2.3rem; }
    .vs-badge { width: 46px; height: 46px; font-size: 1.05rem; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="eyebrow"><span class="dot"></span>PIALA DUNIA &middot; PREDIKSI HASIL</div>
<h1 class="page-title">World Cup<br>Match Predictor</h1>
<p class="page-sub">Regresi logistik (L1) yang membandingkan rating kekuatan dua tim dan
keuntungan main di kandang, lalu menghitung peluang menang, seri, atau kalah.</p>
<hr class="hr-line">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Team selection
# ---------------------------------------------------------------------------

teams_list = list(team_power.keys())
col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<p class="side-label home-label">KANDANG</p>', unsafe_allow_html=True)
        home_team = st.selectbox("Tim kandang", teams_list, index=teams_list.index('Brazil'),
                                  label_visibility="collapsed")
        home_rating = st.slider("Rating power", 50, 100, team_power[home_team], key="home")

with col2:
    with st.container(border=True):
        st.markdown('<p class="side-label away-label">TANDANG</p>', unsafe_allow_html=True)
        away_team = st.selectbox("Tim tandang", teams_list, index=teams_list.index('Japan'),
                                  label_visibility="collapsed")
        away_rating = st.slider("Rating power", 50, 100, team_power[away_team], key="away")

st.markdown(f"""
<div class="vs-card">
    <div class="vs-side">
        <div class="vs-badge home">{get_initials(home_team)}</div>
        <div class="vs-name">{home_team}</div>
        <div class="vs-rating">PWR {home_rating}</div>
    </div>
    <div class="vs-mid">
        <span class="vs-line"></span>
        <span class="vs-circle">VS</span>
        <span class="vs-line"></span>
    </div>
    <div class="vs-side">
        <div class="vs-badge away">{get_initials(away_team)}</div>
        <div class="vs-name">{away_team}</div>
        <div class="vs-rating">PWR {away_rating}</div>
    </div>
</div>
""", unsafe_allow_html=True)

predict = st.button("MULAI PREDIKSI", use_container_width=True)

st.markdown('<hr class="hr-line">', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if predict:
    if home_team == away_team:
        st.warning("Pilih dua tim yang berbeda dulu.")
    else:
        power_diff = (home_rating + 5) - away_rating
        features = scaler.transform([[home_rating, away_rating, power_diff]])
        prob = model.predict_proba(features)[0]
        st.session_state.last_result = {
            "home_team": home_team, "away_team": away_team,
            "home_rating": home_rating, "away_rating": away_rating,
            "p_away": prob[0], "p_draw": prob[1], "p_home": prob[2],
        }

result = st.session_state.last_result

if result:
    p_home, p_draw, p_away = result["p_home"], result["p_draw"], result["p_away"]
    leader = result["home_team"] if p_home >= p_away else result["away_team"]
    chip_class = "tie" if p_draw >= max(p_home, p_away) else ("home" if leader == result["home_team"] else "away")
    chip_text = "Hasil Sangat Terbuka" if chip_class == "tie" else f"{leader} &middot; {confidence_label(max(p_home, p_away))}"

    st.markdown(f"""
    <div class="result-eyebrow">ANALISIS PERTANDINGAN</div>
    <div class="result-title">{result['home_team']} vs {result['away_team']}</div>
    <span class="chip {chip_class}">{chip_text}</span>
    <div class="meter-labels">
        <span>{result['home_team']} <strong>{p_home*100:.1f}%</strong></span>
        <span>Seri <strong>{p_draw*100:.1f}%</strong></span>
        <span>{result['away_team']} <strong>{p_away*100:.1f}%</strong></span>
    </div>
    <div class="meter-track">
        <div class="meter-seg home" style="width:{p_home*100:.2f}%"></div>
        <div class="meter-seg draw" style="width:{p_draw*100:.2f}%"></div>
        <div class="meter-seg away" style="width:{p_away*100:.2f}%"></div>
    </div>
    <div class="analysis-card">{generate_analysis(
        result['home_team'], result['away_team'],
        result['home_rating'], result['away_rating'],
        p_home, p_draw, p_away
    )}</div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">Atur kedua tim dan rating power di atas, lalu tekan
    <strong>MULAI PREDIKSI</strong> untuk melihat analisisnya.</div>
    """, unsafe_allow_html=True)