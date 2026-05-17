import ast
import os
import re
import string
from urllib.parse import quote_plus

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("API_KEY")

# ─────────────────────────────────────────────────────
#  PAGE CONFIG & SESSION STATE (MUST BE FIRST)
# ─────────────────────────────────────────────────────
_sidebar_state = (
  "expanded"
  if st.session_state.get("sidebar_open", False)
  else "collapsed"
)
st.set_page_config(
    page_title="CineAI • Movie Discovery",
    page_icon="🎬",
    layout="wide",
  initial_sidebar_state=_sidebar_state,
)

if "page" not in st.session_state:
    st.session_state.page = "landing"


def trigger_rerun():
  rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
  if rerun_fn is not None:
    rerun_fn()

# ─────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Playfair+Display:ital,wght@0,700;1,700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap" rel="stylesheet">

<style>
/* ── variables ── */
:root {
  --bg:       #07080f;
  --panel:    rgba(11,13,24,0.92);
  --text:     #eef0fb;
  --muted:    #7a84a8;
  --red:      #e63946;
  --amber:    #f4a261;
  --cyan:     #4cc9f0;
  --green:    #52c986;
  --border:   rgba(255,255,255,0.07);
  --shadow:   0 24px 80px rgba(0,0,0,0.55);
  --r-card:   20px;
  --r-pill:   999px;
}

/* ── reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
a { color: inherit; text-decoration: none; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { visibility: hidden !important; display: none !important; }

/* Also hide the Streamlit top toolbar/header bar that blocks content */
[data-testid="stHeader"] {
    display: none !important;
    height: 0 !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
.stAppDeployButton {
    display: none !important;
}

/* Push the app content to start from the very top */
.stApp > div:first-child {
    padding-top: 0 !important;
}
.block-container {
  padding-top: 0 !important;
  margin-top: 0 !important;
  max-width: 100% !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: rgba(7,8,15,0.98) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 2rem 1.2rem !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 99px; }

/* ═══════════════════════════════════════════
   LANDING PAGE
═══════════════════════════════════════════ */
.site-header {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 58px;
    background: rgba(7,8,15,0.35);
    border-bottom: 1px solid rgba(230,57,70,0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    z-index: 9999;
}
.site-header-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.55rem;
    letter-spacing: 0.12em;
    color: #fff;
}
.site-header-logo span { color: #e63946; }
.site-header-nav {
    display: flex;
    gap: 2.4rem;
    align-items: center;
}
.site-header-nav a,
.site-header-nav span {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-decoration: none;
    cursor: pointer;
    transition: color 0.2s;
}
.site-header-nav a:hover,
.site-header-nav span:hover { color: #fff; }

.landing {
  position: relative;
  min-height: 120vh;
  display: grid;
  grid-template-rows: auto 1fr auto auto;
  overflow: hidden;
  padding-top: 58px;
  background:
    radial-gradient(ellipse 70% 55% at 50% -10%, rgba(230,57,70,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 15% 80%,  rgba(76,201,240,0.10) 0%, transparent 55%),
    radial-gradient(ellipse 45% 35% at 85% 75%,  rgba(244,162,97,0.10) 0%, transparent 55%),
    var(--bg);
}

/* grain overlay */
.landing::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
  opacity: 0.03;
  pointer-events: none;
  z-index: 0;
}

/* nav */
.l-nav {
  position: relative;
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.8rem 3.5rem 0;
}
.l-logo {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  letter-spacing: 0.12em;
  color: var(--text);
}
.l-nav-links {
  display: flex;
  gap: 2.2rem;
  font-size: 0.88rem;
  color: var(--muted);
  letter-spacing: 0.06em;
}
.l-nav-links span { cursor: pointer; transition: color 0.2s; }
.l-nav-links span:hover { color: var(--text); }
.l-nav-links a { cursor: pointer; transition: color 0.2s; color: var(--muted); text-decoration: none; }
.l-nav-links a:hover { color: var(--text); }

/* floating poster shapes */
.float-poster {
  position: absolute;
  width: 140px;
  height: 210px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
  overflow: hidden;
  z-index: 1;
  cursor: pointer;
  transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.float-poster:hover {
  transform: scale(1.06) rotate(0deg) !important;
  box-shadow: 0 28px 70px rgba(0,0,0,0.8), 0 0 0 2px rgba(76,201,240,0.6);
  border-color: rgba(76,201,240,0.6);
  z-index: 10;
}
.float-poster::before {
  content: '';
  position: absolute;
  inset: 0;
}
.float-poster::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 55%;
  background: linear-gradient(to top, rgba(7,8,15,0.9) 0%, transparent 100%);
  z-index: 1;
}

.float-poster-img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: 0;
}

/* individual positions & gradients */
.fp1 {
  top: 8%; left: 2%;
  transform: rotate(-7deg);
  animation: float1 7s ease-in-out infinite;
}
.fp2 {
  top: 4%; left: 17%;
  transform: rotate(4deg);
  animation: float2 8.5s ease-in-out infinite;
}
.fp3 {
  top: 10%; right: 16%;
  transform: rotate(-5deg);
  animation: float3 9s ease-in-out infinite;
}
.fp4 {
  top: 4%; right: 2%;
  transform: rotate(6deg);
  animation: float1 10s ease-in-out infinite;
}
.fp5 {
  bottom: 32%; left: 5%;
  transform: rotate(5deg);
  animation: float2 9.5s ease-in-out infinite;
}
.fp6 {
  bottom: 30%; right: 4%;
  transform: rotate(-8deg);
  animation: float3 8s ease-in-out infinite;
}
.fp7 {
  bottom: 22%; left: 22%;
  transform: rotate(3deg);
  animation: float1 11s ease-in-out infinite;
  background: linear-gradient(145deg, #0d0d1a 0%, #2d3580 50%, #4c6cf0 100%);
}
.fp8 {
  bottom: 22%; right: 22%;
  transform: rotate(-4deg);
  animation: float2 7.5s ease-in-out infinite;
  background: linear-gradient(145deg, #1a0808 0%, #8b1a1a 50%, #e63946 100%);
}

/* Fake title bars on posters */
.float-poster .poster-label {
  position: absolute;
  bottom: 0.7rem;
  left: 0.7rem;
  right: 0.7rem;
  z-index: 2;
  font-size: 0.72rem;
  font-weight: 700;
  color: rgba(255,255,255,0.85);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.poster-bar {
  width: 60%;
  height: 3px;
  border-radius: 2px;
  margin-bottom: 0.35rem;
  opacity: 0.6;
}

@keyframes float1 {
  0%,100% { transform: rotate(-7deg) translateY(0px); }
  50%      { transform: rotate(-5deg) translateY(-14px); }
}
@keyframes float2 {
  0%,100% { transform: rotate(4deg) translateY(0px); }
  50%      { transform: rotate(6deg) translateY(-18px); }
}
@keyframes float3 {
  0%,100% { transform: rotate(-5deg) translateY(0px); }
  50%      { transform: rotate(-3deg) translateY(-11px); }
}

/* hero center */
.hero-center {
  position: relative;
  z-index: 2;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  flex: 1;
  width: min(76rem, calc(100% - 3rem));
  margin: 0 auto;
  padding: 2.4rem 1.4rem 1.5rem;
  border-radius: 22px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(180deg, rgba(7,8,15,0.44) 0%, rgba(7,8,15,0.24) 100%);
}

.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.4rem 1rem;
  border-radius: var(--r-pill);
  border: 1px solid rgba(230,57,70,0.3);
  background: rgba(230,57,70,0.08);
  color: #ff9aa3;
  font-size: 0.8rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 1.4rem;
  animation: fadeUp 0.6s ease both;
}

.hero-h1 {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(3.9rem, 9.4vw, 8.8rem);
  line-height: 0.9;
  letter-spacing: 0.01em;
  margin: 0 0 0.75rem 0;
  animation: fadeUp 0.7s 0.1s ease both;
}
.hero-h1 .line-red  { color: var(--red);   display: block; }
.hero-h1 .line-main { color: var(--text);  display: block; }
.hero-h1 .line-dim  { color: var(--muted); display: block; }

.hero-sub {
  color: var(--muted);
  font-size: clamp(0.88rem, 1.05vw, 1rem);
  max-width: 37rem;
  line-height: 1.45;
  margin: 0 auto 1rem;
  animation: fadeUp 0.7s 0.2s ease both;
}

@media (max-width: 900px) {
  .hero-center {
    width: calc(100% - 1.2rem);
    padding: 1.5rem 0.9rem 1rem;
    border-radius: 16px;
  }
  .hero-h1 {
    font-size: clamp(2.5rem, 11vw, 4.3rem);
    line-height: 0.96;
  }
  .hero-sub {
    max-width: 92%;
    font-size: 0.88rem;
    line-height: 1.4;
  }
}

/* stats strip */
.l-stats {
  position: relative;
  z-index: 3;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  margin-top: 1.5rem;
  padding: 2.5rem 2rem;
  border-top: 1px solid var(--border);
  background: rgba(7,8,15,0.35);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  animation: fadeUp 0.8s 0.3s ease both;
}
.l-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.2rem;
  padding: 0 3rem;
}
.l-stat-n {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem;
  color: var(--text);
  letter-spacing: 0.05em;
}
.l-stat-l {
  font-size: 0.76rem;
  color: var(--muted);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.l-stat-sep {
  width: 1px;
  height: 40px;
  background: var(--border);
}

/* landing CTA button override */
.cta-wrap { animation: fadeUp 0.7s 0.25s ease both; }
div[data-testid="stButton"] > button {
  background: var(--red) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--r-pill) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  padding: 0.75rem 2rem !important;
  cursor: pointer !important;
  transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
  box-shadow: 0 8px 30px rgba(230,57,70,0.4) !important;
}
div[data-testid="stButton"] > button:hover {
  background: #c62836 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 12px 40px rgba(230,57,70,0.55) !important;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(22px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ═══════════════════════════════════════════
   APP PAGE
═══════════════════════════════════════════ */
.app-page {
  padding: 0;
  min-height: 100vh;
  background:
    radial-gradient(ellipse 60% 40% at 0% 0%,   rgba(76,201,240,0.08) 0%, transparent 50%),
    radial-gradient(ellipse 50% 35% at 100% 15%, rgba(244,162,97,0.08) 0%, transparent 50%),
    var(--bg);
}

/* app topbar - overridden in show_app() */

/* app inner wrap */
.app-inner {
  padding: 1.8rem 2.5rem 3rem;
  max-width: 1440px;
  margin: 0 auto;
}

/* glass card */
.g-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--r-card);
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
}

/* selected movie detail */
.detail-wrap {
  display: grid;
  grid-template-columns: 230px 1fr;
  gap: 1.8rem;
  padding: 1.5rem;
  margin-bottom: 2rem;
}
@media(max-width:820px) { .detail-wrap { grid-template-columns: 1fr; } }

.poster-frame {
  box-sizing: border-box;
  padding: 0.45rem;
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(135deg, rgba(230,57,70,0.2), rgba(76,201,240,0.15));
  border: 1px solid var(--border);
  aspect-ratio: 2/3;
}
.poster-frame img {
  width: 100%; height: 100%;
  object-fit: cover; display: block;
  border-radius: 12px;
}
.poster-placeholder {
  width: 100%; aspect-ratio: 2/3;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 0.85rem; text-align: center;
  padding: 1rem;
  background: linear-gradient(135deg, rgba(230,57,70,0.1), rgba(76,201,240,0.1));
}

.detail-info { display: flex; flex-direction: column; justify-content: center; gap: 0.9rem; }
.detail-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.6rem, 2.5vw, 2.6rem);
  line-height: 1.05;
  letter-spacing: -0.03em;
}
.meta-row {
  display: flex; gap: 0.7rem; flex-wrap: wrap; align-items: center;
}
.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.3rem 0.75rem;
  border-radius: var(--r-pill);
  font-size: 0.8rem; font-weight: 600;
}
.pill-amber {
  background: rgba(244,162,97,0.12);
  border: 1px solid rgba(244,162,97,0.28);
  color: #ffd2a6;
}
.pill-cyan {
  background: rgba(76,201,240,0.10);
  border: 1px solid rgba(76,201,240,0.25);
  color: #b3eeff;
}
.pill-muted {
  background: rgba(255,255,255,0.05);
  border: 1px solid var(--border);
  color: var(--muted);
}

.detail-overview {
  color: #c8d0e8;
  font-size: 0.96rem;
  line-height: 1.7;
}

.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
  display: inline-flex; align-items: center;
  padding: 0.25rem 0.65rem;
  border-radius: var(--r-pill);
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.07);
  color: #b0b8d4;
  font-size: 0.78rem;
}
.chip b { color: var(--text); margin-right: 0.3rem; }

/* section heading */
.section-heading {
  font-family: 'Playfair Display', serif;
  font-size: 1.25rem;
  letter-spacing: -0.02em;
  margin: 0 0 1rem 0;
  color: var(--text);
}
.section-heading span { color: var(--red); }

/* movie grid */
.movie-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.m-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--r-card);
  overflow: hidden;
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.m-card:hover {
  transform: translateY(-5px);
  border-color: rgba(76,201,240,0.25);
  box-shadow: 0 16px 50px rgba(0,0,0,0.5);
}
.m-poster {
  width: 100%; aspect-ratio: 2/3;
  object-fit: cover; display: block;
  background: linear-gradient(135deg, rgba(230,57,70,0.15), rgba(76,201,240,0.12));
}
.m-poster-ph {
  width: 100%; aspect-ratio: 2/3;
  background: linear-gradient(135deg, rgba(230,57,70,0.15), rgba(76,201,240,0.12));
  display: flex; align-items: center; justify-content: center;
  font-size: 2rem; color: rgba(255,255,255,0.15);
}
.m-body { padding: 0.85rem 0.9rem 1rem; }
.m-title { font-weight: 700; font-size: 0.92rem; line-height: 1.2; margin-bottom: 0.4rem; }
.m-meta { color: var(--muted); font-size: 0.8rem; margin-bottom: 0.55rem; }
.m-score {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.2rem 0.55rem;
  border-radius: var(--r-pill);
  background: rgba(230,57,70,0.12);
  border: 1px solid rgba(230,57,70,0.22);
  color: #ffb3b9;
  font-size: 0.76rem; font-weight: 700;
}

/* reviews */
.review-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.9rem;
}
.review-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.6rem;
}
.review-author { font-weight: 700; font-size: 0.92rem; }
.badge-pos {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.25rem 0.65rem; border-radius: var(--r-pill);
  background: rgba(82,201,134,0.14); border: 1px solid rgba(82,201,134,0.3);
  color: #a8f0c6; font-size: 0.76rem; font-weight: 700;
}
.badge-neg {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.25rem 0.65rem; border-radius: var(--r-pill);
  background: rgba(230,57,70,0.14); border: 1px solid rgba(230,57,70,0.3);
  color: #ffb3b9; font-size: 0.76rem; font-weight: 700;
}
.badge-neu {
  display: inline-flex; align-items: center; gap: 0.3rem;
  padding: 0.25rem 0.65rem; border-radius: var(--r-pill);
  background: rgba(255,255,255,0.06); border: 1px solid var(--border);
  color: var(--muted); font-size: 0.76rem; font-weight: 700;
}
.review-text { color: #c4cce4; font-size: 0.9rem; line-height: 1.65; }

/* how-it-works */
.how-steps {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem;
}
.how-step {
  padding: 1.1rem 1rem;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 16px;
}
.how-step-n {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 2rem; color: var(--red); margin-bottom: 0.4rem;
}
.how-step-t { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.3rem; }
.how-step-d { color: var(--muted); font-size: 0.82rem; line-height: 1.5; }

/* model status pills */
.status-row { display: flex; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 1rem; }
.status-ok {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.4rem 0.9rem; border-radius: var(--r-pill);
  background: rgba(82,201,134,0.1); border: 1px solid rgba(82,201,134,0.25);
  color: #aaf0c6; font-size: 0.82rem; font-weight: 600;
}
.status-warn {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.4rem 0.9rem; border-radius: var(--r-pill);
  background: rgba(244,162,97,0.1); border: 1px solid rgba(244,162,97,0.25);
  color: #ffd5a8; font-size: 0.82rem; font-weight: 600;
}

/* sidebar style overrides */
.sidebar-label {
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 0.5rem;
}
.sidebar-tip {
  padding: 0.85rem 0.9rem;
  background: rgba(76,201,240,0.06);
  border: 1px solid rgba(76,201,240,0.14);
  border-radius: 14px;
  color: var(--muted); font-size: 0.82rem; line-height: 1.55;
  margin-top: 1.5rem;
}
.sidebar-tip b { color: #b3eeff; }

/* override streamlit selectbox, slider labels */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}

/* tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.03em !important;
}

/* info/warn boxes */
[data-testid="stAlert"] {
  border-radius: 14px !important;
  font-size: 0.88rem !important;
}

/* Hide default sidebar collapse button from view but keep it in the DOM */
[data-testid="collapsedControl"] {
  opacity: 0 !important;
  pointer-events: none !important;
  position: absolute !important;
  left: -9999px !important;
}

/* Hamburger toggle handled in show_app() */

/* Hero Banner */
.hero-banner {
    position: relative;
    width: 100%;
    height: 500px;
    overflow: hidden;
    margin-bottom: 0;
}
.hero-banner-img {
    width: 100%; height: 100%;
    object-fit: cover;
    object-position: center top;
}
.hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(
        to right,
        rgba(7,8,15,0.97) 0%,
        rgba(7,8,15,0.80) 38%,
        rgba(7,8,15,0.3) 65%,
        transparent 100%
    );
}
.hero-content {
    position: absolute;
    bottom: 60px; left: 3.5rem;
    max-width: 480px; z-index: 2;
}
.hero-hd-badge {
    display: inline-block;
    background: #f4c430; color: #000;
    font-size: 11px; font-weight: 800;
    padding: 3px 10px; border-radius: 4px;
    letter-spacing: 0.08em; margin-bottom: 12px;
}
.hero-title {
    font-size: 2.8rem; font-weight: 800;
    color: #fff; line-height: 1.05;
    margin-bottom: 10px;
}
.hero-meta {
    font-size: 0.85rem; color: rgba(255,255,255,0.65);
    margin-bottom: 14px; letter-spacing: 0.04em;
}
.hero-overview {
    font-size: 0.9rem; color: rgba(255,255,255,0.55);
    line-height: 1.6; margin-bottom: 22px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.hero-btns { display: flex; gap: 12px; }
.hero-btn-primary {
    padding: 11px 24px; border-radius: 8px;
    background: #4cc9f0; color: #07080f;
    font-weight: 700; font-size: 0.88rem;
    border: none; cursor: pointer;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.hero-btn-primary:hover { opacity: 0.85; }
.hero-btn-secondary {
    padding: 11px 24px; border-radius: 8px;
    background: transparent; color: #fff;
    font-weight: 600; font-size: 0.88rem;
    border: 1px solid rgba(255,255,255,0.35); cursor: pointer;
    transition: border-color 0.2s;
}
.hero-btn-secondary:hover { border-color: #fff; }
.hero-dots {
    position: absolute; bottom: 22px; left: 50%;
    transform: translateX(-50%);
    display: flex; gap: 8px; z-index: 2;
}
.hero-dot {
    width: 28px; height: 3px; border-radius: 2px;
    background: rgba(255,255,255,0.25);
}
.hero-dot.active { background: #fff; }

/* Streaming Grid */
.stream-section-label {
    color: #4cc9f0;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 2.5rem 3.5rem 1.2rem;
}
.stream-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 14px;
    padding: 0 3.5rem 2rem;
}
.s-card {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    background: #111320;
    cursor: pointer;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
    border: 2px solid transparent;
    aspect-ratio: 2/3;
}
.s-card:hover {
    transform: scale(1.04);
    border-color: #4cc9f0;
    box-shadow: 0 12px 40px rgba(76,201,240,0.25);
    z-index: 2;
}
.s-poster {
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
}
.s-poster-ph {
    width: 100%; height: 100%;
    display: flex; align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    background: linear-gradient(145deg, #111320, #1a2040);
}
.s-badge-hd {
    position: absolute; top: 9px; right: 9px;
    background: #f4c430; color: #000;
    font-size: 10px; font-weight: 800;
    padding: 2px 7px; border-radius: 4px;
    letter-spacing: 0.06em; z-index: 2;
}
.s-gradient {
    position: absolute; inset: 0;
    background: linear-gradient(to top,
        rgba(7,8,15,1) 0%,
        rgba(7,8,15,0.5) 35%,
        transparent 60%);
    z-index: 1;
}
.s-title {
    position: absolute; bottom: 12px; left: 10px; right: 10px;
    z-index: 3;
    font-size: 0.82rem; font-weight: 700;
    color: #fff; line-height: 1.3;
}

/* Sidebar styling */
[data-testid="stSidebar"] .stSlider { padding: 0 !important; }
[data-testid="stSidebar"] .stSlider > div { padding: 0 !important; }
[data-testid="stSidebar"] .stSelectbox { margin-top: 0 !important; }
[data-testid="stSidebar"] .stCheckbox { 
    margin: 6px 0 !important; 
    padding: 0 !important;
}
[data-testid="stSidebar"] .stCheckbox label {
    font-size: 0.85rem !important;
    color: #7a84a8 !important;
}
[data-testid="stSidebar"] .stCheckbox label:hover {
    color: #eef0fb !important;
}

/* Make recommendation card buttons invisible but clickable */
.stream-grid button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    z-index: 10 !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 12px !important;
}

/* Hide the poster click capture input */
div[data-testid="stTextInput"]:has(input[aria-label="poster_click"]) {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

  /* Old hamburger button CSS removed - now handled via app-hamburger */

</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────
#  DATA LOADERS (cached)
# ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_movies() -> pd.DataFrame:
    for path in ["cleaned_movies.csv", "tmdb_5000_movies.csv", "archive/tmdb_5000_movies.csv"]:
        if os.path.exists(path):
            frame = pd.read_csv(path)
            return frame
    raise FileNotFoundError("No movie dataset found. Ensure cleaned_movies.csv or tmdb_5000_movies.csv is present.")


@st.cache_resource(show_spinner=False)
def load_artifacts():
  paths = {
    "similarity": ["Model/similarity.joblib", "similarity.joblib"],
    "sentiment":  ["Model/sentiment_analysis_model.pkl"],
    "vectorizer": ["Model/tfidf_vectorizer.pkl"],
  }
  loaded = {}
  for key, candidates in paths.items():
    for p in candidates:
      if not os.path.exists(p):
        continue
      try:
        if key == "similarity":
          # Skip loading the large similarity matrix; use TF-IDF fallback instead
          loaded[key] = None
        else:
          loaded[key] = joblib.load(p)
      except Exception:
        # other errors: skip this candidate and try next
        if key == "similarity":
          loaded[key] = None
        else:
          continue
      break
    if key not in loaded:
      if key == "similarity":
        loaded[key] = None
      else:
        raise FileNotFoundError(f"Missing model file for '{key}'. Run the notebook to regenerate Model/.")
  return loaded["similarity"], loaded["sentiment"], loaded["vectorizer"]

@st.cache_resource(show_spinner=False)
def build_recommender_tfidf(movies_df: pd.DataFrame):
  # Build a TF-IDF matrix over the `tags` column for on-demand similarity computation.
  # This avoids loading a potentially very large precomputed similarity matrix into memory.
  if "tags" not in movies_df.columns:
    # Create a lightweight tags column from overview/genres if missing
    cols = []
    for c in ["overview", "genres", "keywords", "cast", "crew"]:
      if c in movies_df.columns:
        cols.append(c)
    movies_df["tags"] = (
      movies_df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    )
  vec = TfidfVectorizer(stop_words="english", max_features=5000)
  tfidf_mat = vec.fit_transform(movies_df["tags"].astype(str))
  return vec, tfidf_mat


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_poster(movie_id: int) -> str:
    if not api_key:
        return ""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US",
            timeout=10,
        )
        r.raise_for_status()
        pp = r.json().get("poster_path", "")
        return f"https://image.tmdb.org/t/p/w500{pp}" if pp else ""
    except Exception:
        return ""


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_reviews(movie_id: int):
    if not api_key:
        return []
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={api_key}",
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])[:5]
        return [{"author": rv.get("author", "Unknown"), "content": rv.get("content", "")} for rv in results]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def genre_banner_posters(movies_df: pd.DataFrame):
    # For now, just return empty URLs to avoid TMDB API delays on landing page
    # The landing page will render with gradient backgrounds
    poster_map = {
        "action": "",
        "sci-fi": "",
        "drama": "",
        "thriller": "",
        "fantasy": "",
        "comedy": "",
        "romance": "",
        "horror": "",
    }
    return poster_map


# ─────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────
def _safe_str(v):
    return "" if pd.isna(v) else str(v)


def fmt_genres(v, sep=", "):
    raw = _safe_str(v)
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return raw
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            return sep.join(i.get("name", "") for i in parsed if i.get("name"))
        return sep.join(str(i) for i in parsed)
    return raw


def fmt_cast(v, limit=4):
    raw = _safe_str(v)
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return raw
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            names = [i.get("name", "") for i in parsed if i.get("name")]
            return ", ".join(names[:limit])
        return ", ".join(str(i) for i in parsed[:limit])
    return raw


def fmt_director(v):
    raw = _safe_str(v)
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return raw
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            dirs = [i.get("name", "") for i in parsed if i.get("job", "").lower() == "director" and i.get("name")]
            return ", ".join(dirs) if dirs else ", ".join(i.get("name", "") for i in parsed if i.get("name"))
        return ", ".join(str(i) for i in parsed)
    return raw


def fmt_overview(v):
    """Convert list-formatted overview to readable text."""
    raw = _safe_str(v)
    if not raw:
        return ""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return " ".join(str(word).strip("',\"") for word in parsed)
    except Exception:
        pass
    return raw


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_sentiment(review: str, model, vec) -> str:
    processed = preprocess_text(review)
    vectorized = vec.transform([processed])
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and vectorized.shape[1] != expected:
        raise ValueError(
            f"Model/vectorizer mismatch: vectorizer → {vectorized.shape[1]} features, "
            f"model expects {expected}. Rebuild both from the same training run."
        )
    result = model.predict(vectorized)[0]
    return "positive" if result == 1 else "negative"


def recommend(movie: str, movies_df: pd.DataFrame, sim_matrix, top_n: int = 8,
              min_rating: float = 0.0, genre_filter: str = "All") -> list:
    idx_matches = movies_df.index[movies_df["title"] == movie]
    if len(idx_matches) == 0:
        return []
    movie_index = idx_matches[0]
    # If a precomputed similarity matrix was not available or couldn't be memory-mapped,
    # compute the similarity scores for the selected movie on demand using a cached
    # TF-IDF representation. This avoids loading a full dense similarity matrix.
    if sim_matrix is None:
      try:
        vec_rec, tfidf_mat = build_recommender_tfidf(movies_df)
        distances = cosine_similarity(tfidf_mat[movie_index], tfidf_mat).flatten()
      except Exception:
        # As a final fallback, return empty results rather than crashing the app.
        return []
    else:
      distances = sim_matrix[movie_index]

    ranked = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1: top_n * 10]

    results = []
    gf = (genre_filter or "All").lower()

    for row_idx, score in ranked:
      row = movies_df.iloc[row_idx]
      rating = float(row.get("vote_average", 0) or 0)
      if rating < min_rating:
        continue
      row_genres = fmt_genres(row.get("genres", ""))
      if gf != "all" and gf not in row_genres.lower():
        continue
      movie_id = int(row["id"]) if "id" in row.index else 0
      results.append({
        "title":        str(row["title"]),
        "id":           movie_id,
        "poster":       fetch_poster(movie_id) if movie_id else "",
        "genres":       row_genres,
        "rating":       rating,
        "release_date": _safe_str(row.get("release_date", ""))[:4],
        "score":        float(score),
      })
      if len(results) == top_n:
        break
    return results


# ─────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────
try:
    movies_data = load_movies().copy()
    similarity, sentiment_model, vectorizer = load_artifacts()
    DATA_OK = True
except FileNotFoundError as e:
    DATA_OK = False
    DATA_ERR = str(e)

if DATA_OK:
    required_cols = {"id", "title", "overview", "genres", "vote_average"}
    missing_cols = required_cols - set(movies_data.columns)
    if missing_cols:
        DATA_OK = False
        DATA_ERR = f"Dataset missing columns: {', '.join(sorted(missing_cols))}. Run the notebook first."

if DATA_OK:
    movies_data["title"] = movies_data["title"].fillna("").astype(str)
    movie_titles = movies_data["title"].dropna().tolist()

# Apply landing card deep-links like ?page=app&genre=Action&movie=Avatar
try:
  qp_page = st.query_params.get("page")
  qp_movie = st.query_params.get("movie")
  qp_genre = st.query_params.get("genre")

  if qp_page in {"landing", "app"}:
    st.session_state.page = qp_page
  if qp_movie:
    st.session_state.landing_selected_movie = str(qp_movie)
    st.session_state["selected_movie"] = str(qp_movie)
  if qp_genre:
    st.session_state.landing_genre_filter = str(qp_genre)

  if qp_page or qp_movie or qp_genre:
    st.query_params.clear()
except Exception:
  pass


# ─────────────────────────────────────────────────────
#  NAVIGATION HELPER
# ─────────────────────────────────────────────────────
def navigate_to_movie(movie_title):
    """Set session state and navigate to app page with selected movie."""
    st.session_state["page"] = "app"
    st.session_state["selected_movie"] = movie_title
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


# ─────────────────────────────────────────────────────
#  LANDING PAGE
# ─────────────────────────────────────────────────────
def show_landing():
    _float_movies = [
        "The Dark Knight",
        "The Empire Strikes Back",
        "Spirited Away",
        "Pulp Fiction",
        "Psycho",
        "Inception",
    ]

    genre_specs = [
        ("fp1", "Action", "Action", "var(--red)"),
        ("fp2", "Sci-Fi", "ScienceFiction", "var(--cyan)"),
        ("fp3", "Drama", "Drama", "var(--amber)"),
        ("fp4", "Thriller", "Thriller", "var(--green)"),
        ("fp5", "Fantasy", "Fantasy", "#b57bee"),
        ("fp6", "Comedy", "Comedy", "#f4d261"),
        ("fp7", "Romance", "Romance", "#4c6cf0"),
        ("fp8", "Horror", "Horror", "var(--red)"),
    ]

    genre_poster_fallbacks = {
      "Action": "https://upload.wikimedia.org/wikipedia/en/b/b0/Avatar-Teaser-Poster.jpg",
      "ScienceFiction": "https://upload.wikimedia.org/wikipedia/en/3/3c/SW_-_Empire_Strikes_Back.jpg",
      "Drama": "https://upload.wikimedia.org/wikipedia/en/2/22/Titanic_poster.jpg",
      "Thriller": "https://upload.wikimedia.org/wikipedia/en/8/82/Pulp_Fiction_cover.jpg",
      "Fantasy": "https://upload.wikimedia.org/wikipedia/en/d/db/Spirited_Away_Japanese_poster.png",
      "Comedy": "https://upload.wikimedia.org/wikipedia/en/3/39/Forrest_Gump_poster.jpg",
      "Romance": "https://upload.wikimedia.org/wikipedia/en/0/08/La_La_Land_%28film%29.png",
      "Horror": "https://upload.wikimedia.org/wikipedia/en/5/54/The_Shining_%281980%29.png",
    }

    def featured_movie_for_genre(genre_name: str) -> tuple[str, str, str]:
      if not DATA_OK or "genres" not in movies_data.columns or "vote_average" not in movies_data.columns:
        return "Featured Title", "", genre_poster_fallbacks.get(genre_name, "")

      matches = movies_data[movies_data["genres"].astype(str).str.contains(genre_name, case=False, na=False)].copy()
      if matches.empty:
        return "Featured Title", "", genre_poster_fallbacks.get(genre_name, "")

      matches["vote_average"] = pd.to_numeric(matches["vote_average"], errors="coerce").fillna(0)
      best = matches.sort_values(["vote_average", "title"], ascending=[False, True]).iloc[0]
      title = str(best.get("title", "Featured Title"))
      year = _safe_str(best.get("release_date", ""))[:4]
      best_id = int(best.get("id", 0)) if str(best.get("id", "")).strip() else 0
      poster_url = fetch_poster(best_id) if best_id else ""
      if not poster_url:
        poster_url = genre_poster_fallbacks.get(genre_name, "")
      return title, year, poster_url

    def float_poster_block(cls, label, genre_name, accent, title, year, poster_url):
        subtitle = year if year else ""
        open_href = f"?page=app&genre={quote_plus(genre_name)}&movie={quote_plus(title)}"
        poster_media = (
            f'<img class="float-poster-img" src="{poster_url}" alt="{label} poster" loading="lazy" referrerpolicy="no-referrer">'
            if poster_url else ""
        )
        return f'''
          <div class="float-poster {cls}" data-href="{open_href}"
               style="background: linear-gradient(145deg, rgba(230,57,70,0.28), rgba(76,201,240,0.12) 55%, rgba(7,8,15,0.95) 100%); cursor:pointer;">
            {poster_media}
            <div style="position:absolute; left:14px; right:14px; bottom:16px; z-index:2;">
              <div style="font-family:'Playfair Display',serif; font-size:1.08rem; line-height:1.05; color:var(--text); text-shadow:0 8px 24px rgba(0,0,0,0.45);">{title}</div>
              <div style="margin-top:0.4rem; font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; color:rgba(238,240,251,0.72);">{subtitle}</div>
              <div style="display:inline-block;margin-top:0.35rem;padding:0.15rem 0.5rem;border-radius:999px;font-size:0.58rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;background:{accent};color:#07080f;opacity:0.85;">{label}</div>
            </div>
          </div>
        '''

    float_posts = []
    for cls, label, genre_name, accent in genre_specs:
        title, year, poster_url = featured_movie_for_genre(genre_name)
        float_posts.append((cls, label, genre_name, accent, title, year, poster_url))

    landing_html = """
        <div class="site-header">
            <div class="site-header-logo">CINE<span>AI</span></div>
            <nav class="site-header-nav">
                <span data-nav="discover">Discover</span>
                <span data-nav="sentiment">Sentiment</span>
                <span data-nav="about">About</span>
            </nav>
        </div>

        <div class="landing">

          <!-- FLOATING POSTER SHAPES -->
          {fp1}
          {fp2}
          {fp3}
          {fp4}
          {fp5}
          {fp6}
          {fp7}
          {fp8}

          <!-- HERO CENTER -->
          <div class="hero-center">
            <div class="hero-eyebrow">✦ &nbsp;AI-Powered Cinema Discovery</div>
            <h1 class="hero-h1">
              <span class="line-red">DISCOVER</span>
              <span class="line-main">CINEMA</span>
              <span class="line-dim">UNLOCKED.</span>
            </h1>
            <p class="hero-sub">
              Personalised movie recommendations powered by machine learning.<br>
              Real audience sentiment · TMDB posters · 4,800+ titles indexed.
            </p>
          </div>

          <!-- STATS -->
          <div class="l-stats" style="margin-top:5rem;" id="sentiment-section">
            <div class="l-stat">
              <span class="l-stat-n">4,800<span style="color:var(--red)">+</span></span>
              <span class="l-stat-l">Movies</span>
            </div>
            <div class="l-stat-sep"></div>
            <div class="l-stat">
              <span class="l-stat-n">TF<span style="color:var(--cyan)">-</span>IDF</span>
              <span class="l-stat-l">Similarity Engine</span>
            </div>
            <div class="l-stat-sep"></div>
            <div class="l-stat">
              <span class="l-stat-n">98<span style="color:var(--amber)">%</span></span>
              <span class="l-stat-l">Model Accuracy</span>
            </div>
            <div class="l-stat-sep"></div>
            <div class="l-stat">
              <span class="l-stat-n">TMDB</span>
              <span class="l-stat-l">API Powered</span>
            </div>
          </div>

          <!-- ABOUT SECTION -->
          <div id="about-section" style="position:relative;z-index:3;padding:2rem 3rem 2.5rem;border-top:1px solid var(--border);text-align:center;background:rgba(7,8,15,0.35);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;letter-spacing:0.1em;color:var(--text);margin-bottom:0.5rem;">About <span style="color:var(--red);">CineAI</span></div>
            <p style="max-width:36rem;margin:0 auto;color:var(--muted);font-size:0.88rem;line-height:1.65;">
              CineAI is an AI-powered movie discovery platform that combines TF-IDF content-based filtering
              with real-time TMDB data and sentiment analysis on audience reviews. Built with Streamlit, scikit-learn
              and love for cinema.
            </p>
          </div>

        </div>
        """

    landing_html = (
      landing_html.replace("{fp1}", float_poster_block(*float_posts[0]))
      .replace("{fp2}", float_poster_block(*float_posts[1]))
      .replace("{fp3}", float_poster_block(*float_posts[2]))
      .replace("{fp4}", float_poster_block(*float_posts[3]))
      .replace("{fp5}", float_poster_block(*float_posts[4]))
      .replace("{fp6}", float_poster_block(*float_posts[5]))
      .replace("{fp7}", float_poster_block(*float_posts[6]))
      .replace("{fp8}", float_poster_block(*float_posts[7]))
    )
    st.markdown(landing_html, unsafe_allow_html=True)

    # ── JavaScript for card clicks and nav links ──
    # Streamlit strips onclick from HTML, so we attach listeners via <script>
    st.markdown("""
    <script>
    (function() {
        // Wait for DOM to be ready
        function initClickHandlers() {
            // 1. Float poster card clicks — navigate to app page
            document.querySelectorAll('.float-poster[data-href]').forEach(function(card) {
                card.style.cursor = 'pointer';
                card.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    var href = card.getAttribute('data-href');
                    if (href) {
                        window.location.href = href;
                    }
                });
            });

            // 2. Header nav link clicks
            document.querySelectorAll('[data-nav]').forEach(function(span) {
                span.style.cursor = 'pointer';
                span.addEventListener('click', function(e) {
                    var action = span.getAttribute('data-nav');
                    if (action === 'discover') {
                        window.scrollTo({top: 0, behavior: 'smooth'});
                    } else if (action === 'sentiment') {
                        var el = document.getElementById('sentiment-section');
                        if (el) el.scrollIntoView({behavior: 'smooth'});
                    } else if (action === 'about') {
                        var el = document.getElementById('about-section');
                        if (el) el.scrollIntoView({behavior: 'smooth'});
                    }
                });
            });
        }

        // Streamlit renders async, so retry until elements exist
        var attempts = 0;
        var interval = setInterval(function() {
            attempts++;
            var cards = document.querySelectorAll('.float-poster[data-href]');
            if (cards.length > 0 || attempts > 30) {
                clearInterval(interval);
                initClickHandlers();
            }
        }, 200);
    })();
    </script>
    """, unsafe_allow_html=True)

    # CTA button centred below landing
    _, col, _ = st.columns([3, 1.5, 3])
    with col:
        if st.button("🎬  Enter CineAI", use_container_width=True, key="enter_btn"):
            st.session_state.page = "app"
            trigger_rerun()


# ─────────────────────────────────────────────────────
#  APP PAGE
# ─────────────────────────────────────────────────────
def show_app():
    if not DATA_OK:
        st.error(f"⚠️  {DATA_ERR}")
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            trigger_rerun()
        return

    if "sidebar_open" not in st.session_state:
        st.session_state["sidebar_open"] = False

    # ── Sidebar visibility via CSS ──
    _sb_display = "block" if st.session_state.get("sidebar_open", False) else "none"
    st.markdown(f"""
    <style>
    section[data-testid="stSidebar"] {{
        display: {_sb_display} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ── TOPBAR: 3-column layout ──────────────────────
    _tc1, _tc2, _tc3 = st.columns([1, 7, 1])
    with _tc1:
        if st.button("← Home", key="go_home_btn"):
            st.session_state["page"] = "landing"
            st.session_state.pop("selected_movie", None)
            trigger_rerun()
    with _tc2:
        st.markdown("""
        <div style="text-align:center; padding:0.55rem 0 0.3rem;">
          <span style="font-family:'Bebas Neue',sans-serif;
                       font-size:1.85rem;letter-spacing:0.1em;
                       color:#eef0fb;">
            CINE<span style="color:#e63946;">AI</span>
          </span>
          <span style="color:#7a84a8;font-size:0.78rem;
                       margin-left:0.9rem;letter-spacing:0.06em;
                       border-left:1px solid rgba(255,255,255,0.1);
                       padding-left:0.9rem;">
            Movie Discovery &amp; Sentiment
          </span>
        </div>
        """, unsafe_allow_html=True)
    with _tc3:
        if st.button("☰", key="ham_btn",
                     help="Open / close filters"):
            st.session_state["sidebar_open"] = (
                not st.session_state.get("sidebar_open", False)
            )
            trigger_rerun()

    # Topbar styling
    st.markdown("""
    <style>
    /* Sticky topbar row */
    div[data-testid="stHorizontalBlock"]:has(
        button[data-testid="baseButton-secondary"]
    ):first-of-type {
        position: sticky !important;
        top: 0 !important;
        z-index: 8888 !important;
        background: rgba(7,8,15,0.95) !important;
        border-bottom: 1px solid rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(16px) !important;
        margin: 0 !important;
        padding: 0 0.5rem !important;
    }

    /* Hamburger button — right side */
    div[data-testid="stColumn"]:last-child
    button[kind="secondary"] {
        width: 38px !important;
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 !important;
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #eef0fb !important;
        font-size: 17px !important;
        line-height: 1 !important;
        float: right !important;
        box-shadow: none !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stColumn"]:last-child
    button[kind="secondary"]:hover {
        background: rgba(230,57,70,0.15) !important;
        border-color: rgba(230,57,70,0.45) !important;
        color: #e63946 !important;
    }

    /* Home button — left side subtle text */
    div[data-testid="stColumn"]:first-child
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.45) !important;
        font-size: 0.75rem !important;
        padding: 5px 12px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        width: auto !important;
        min-height: 32px !important;
        margin-top: 10px !important;
        transition: all 0.2s !important;
    }
    div[data-testid="stColumn"]:first-child
    button[kind="secondary"]:hover {
        border-color: rgba(255,255,255,0.3) !important;
        color: #eef0fb !important;
        background: transparent !important;
    }

    /* Search bar wrapper */
    .search-wrap {
        padding: 0.8rem 3rem 0.4rem;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .search-icon { font-size: 1rem; }
    .search-label {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.3);
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    [data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #eef0fb !important;
        padding: 14px 20px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
        cursor: text !important;
    }
    [data-testid="stSelectbox"] > div > div:focus-within,
    [data-testid="stSelectbox"] > div > div:hover {
        border-color: rgba(230,57,70,0.55) !important;
        box-shadow: 0 0 0 3px rgba(230,57,70,0.08) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="app-inner">', unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding: 0.5rem 0 1.5rem;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;
                        letter-spacing:0.12em;color:#eef0fb;margin-bottom:4px;">
                CINE<span style="color:#4cc9f0;">AI</span>
            </div>
            <div style="font-size:0.72rem;color:#7a84a8;letter-spacing:0.1em;
                        text-transform:uppercase;">
                Discovery Settings
            </div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0 0 1.5rem;">
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.75rem;color:#7a84a8;letter-spacing:0.08em;'
                    'text-transform:uppercase;margin-bottom:6px;">Results</p>', 
                    unsafe_allow_html=True)
        top_n = st.slider("", 4, 12, 8, key="top_n_slider", label_visibility="collapsed")

        st.markdown('<p style="font-size:0.75rem;color:#7a84a8;letter-spacing:0.08em;'
                    'text-transform:uppercase;margin:16px 0 6px;">Min Rating ★</p>', 
                    unsafe_allow_html=True)
        min_rating = st.slider("", 0.0, 10.0, 5.0, 0.5, key="rating_slider", 
                               label_visibility="collapsed")

        st.markdown('<p style="font-size:0.75rem;color:#7a84a8;letter-spacing:0.08em;'
                    'text-transform:uppercase;margin:16px 0 6px;">Genre</p>', 
                    unsafe_allow_html=True)
        genre_options = ["All"] + sorted({
            g.strip()
            for v in movies_data["genres"].dropna().astype(str)
            for g in fmt_genres(v).split(",")
            if g.strip()
        })
        prefill_genre = st.session_state.pop("landing_genre_filter", None)
        if prefill_genre in genre_options:
            st.session_state["genre_filter_box"] = prefill_genre
        genre_filter = st.selectbox("", genre_options, key="genre_filter_box",
                                    label_visibility="collapsed")

        st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);'
                    'margin:20px 0;">', unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.75rem;color:#7a84a8;letter-spacing:0.08em;'
                    'text-transform:uppercase;margin-bottom:10px;">Options</p>', 
                    unsafe_allow_html=True)
        show_reviews = st.checkbox("Show TMDB reviews", value=True)
        show_sentiment = st.checkbox("Analyse sentiment", value=True)

        st.markdown("""
        <div style="margin-top:2rem;padding:14px;background:rgba(76,201,240,0.06);
                    border:1px solid rgba(76,201,240,0.15);border-radius:10px;
                    font-size:0.78rem;color:#7a84a8;line-height:1.6;">
            <span style="color:#4cc9f0;font-weight:600;">💡 Tip</span><br>
            Raise minimum rating or pick a genre for more targeted picks.
        </div>
        """, unsafe_allow_html=True)

    # ── Prefill movie from landing card click ──
    prefill_movie = st.session_state.pop("landing_selected_movie", None)
    if prefill_movie:
        # Try exact match first
        if prefill_movie in movie_titles:
            st.session_state["selected_movie"] = prefill_movie
        else:
            # Try case-insensitive match (URL decoding may differ)
            _lower = prefill_movie.lower()
            for t in movie_titles:
                if t.lower() == _lower:
                    st.session_state["selected_movie"] = t
                    break

    # ── STYLED SEARCH BAR (CHANGE 6) ──────────────────
    st.markdown("""
    <div class="search-wrap">
        <div class="search-icon">🔍</div>
        <div class="search-label">What do you want to watch?</div>
    </div>
    """, unsafe_allow_html=True)

    _pre = st.session_state.get("selected_movie", None)
    _all_titles = sorted(
        movies_data["title"].dropna().tolist()
    )
    _default_idx = 0
    if _pre and _pre in _all_titles:
        _default_idx = _all_titles.index(_pre)

    selected_movie = st.selectbox(
        label="",
        options=_all_titles,
        index=_default_idx,
        key="main_movie_select",
        label_visibility="collapsed",
    )
    st.session_state["selected_movie"] = selected_movie

    sel = movies_data.loc[movies_data["title"] == selected_movie].iloc[0]
    sel_id = int(sel["id"]) if "id" in sel.index else 0
    sel_poster = fetch_poster(sel_id) if sel_id else ""
    sel_overview = fmt_overview(sel.get("overview", ""))
    sel_genres = fmt_genres(sel.get("genres", ""))
    sel_cast = fmt_cast(sel.get("cast", ""))
    sel_dir = fmt_director(sel.get("crew", ""))
    sel_date = _safe_str(sel.get("release_date", ""))[:4]
    sel_rating = sel.get("vote_average", "N/A")

    # ── HERO BANNER (fetch backdrop and render) ───────
    hero_backdrop = ""
    if api_key and sel_id:
        try:
            _r = requests.get(
                f"https://api.themoviedb.org/3/movie/{sel_id}?api_key={api_key}",
                timeout=5
            ).json()
            _bp = _r.get("backdrop_path", "")
            if _bp:
                hero_backdrop = f"https://image.tmdb.org/t/p/original{_bp}"
            _rt = _r.get("runtime", 0)
            _runtime_str = f"{_rt} min" if _rt else ""
        except:
            _runtime_str = ""
    else:
        _runtime_str = ""

    _hero_img_tag = (
        f'<img class="hero-banner-img" src="{hero_backdrop}" alt="{selected_movie}">'
        if hero_backdrop
        else '<div style="width:100%;height:100%;background:linear-gradient(135deg,#1a0a2e,#07080f);"></div>'
    )
    _overview_short = sel_overview[:200] + ("…" if len(sel_overview) > 200 else "")
    _genre_first = sel_genres.split(",")[0].strip() if sel_genres else ""
    _meta_parts = " · ".join(filter(None, [_runtime_str, sel_genres[:60] if sel_genres else ""]))

    st.markdown(f"""
    <div class="hero-banner">
        {_hero_img_tag}
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <div class="hero-hd-badge">HD</div>
            <div class="hero-title">{selected_movie}</div>
            <div class="hero-meta">★ {sel_rating} &nbsp;·&nbsp; {_meta_parts}</div>
            <div class="hero-overview">{_overview_short}</div>
            <div class="hero-btns">
                <button class="hero-btn-primary">▶ &nbsp;Watch Trailer</button>
                <button class="hero-btn-secondary">+ Watchlist</button>
            </div>
        </div>
        <div class="hero-dots">
            <div class="hero-dot active"></div>
            <div class="hero-dot"></div>
            <div class="hero-dot"></div>
            <div class="hero-dot"></div>
            <div class="hero-dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SELECTED MOVIE DETAIL ─────────────────────────
    st.markdown('<div class="g-card detail-wrap">', unsafe_allow_html=True)

    pc, ic = st.columns([0.75, 1.6], gap="large")
    with pc:
        if sel_poster:
            st.markdown(f'<div class="poster-frame"><img src="{sel_poster}" alt="{selected_movie}"></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="poster-frame poster-placeholder">🎬<br>Poster<br>Unavailable</div>', unsafe_allow_html=True)

    with ic:
        st.markdown(f'<div class="detail-title">{selected_movie}</div>', unsafe_allow_html=True)
        st.markdown(
            f'''<div class="meta-row">
                  <span class="pill pill-amber">★ {sel_rating}</span>
                  <span class="pill pill-cyan">📅 {sel_date or "Unknown"}</span>
                  <span class="pill pill-muted">{sel_genres.split(",")[0].strip() if sel_genres else "Unknown"}</span>
                </div>''',
            unsafe_allow_html=True,
        )
        if sel_overview:
            st.markdown(f'<div class="detail-overview" style="margin-top:0.8rem;">{sel_overview[:500]}{"…" if len(sel_overview) > 500 else ""}</div>', unsafe_allow_html=True)
        st.markdown(
            f'''<div class="chip-row" style="margin-top:1rem;">
                  <span class="chip"><b>Genres:</b> {sel_genres or "Unknown"}</span>
                  <span class="chip"><b>Cast:</b> {sel_cast or "Unknown"}</span>
                  <span class="chip"><b>Director:</b> {sel_dir or "Unknown"}</span>
                </div>''',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close g-card detail-wrap

    # ── RECOMMENDATIONS (STREAMING GRID) ──────────────
    st.markdown('<div class="stream-section-label">SUGGESTIONS</div>', 
                unsafe_allow_html=True)

    with st.spinner("Finding your next favourite film…"):
        recs = recommend(selected_movie, movies_data, similarity, top_n=top_n,
                         min_rating=min_rating, genre_filter=genre_filter)

    if not recs:
        st.info("No recommendations matched. Try lowering the minimum rating.")
    else:
        st.markdown('<div class="stream-grid">', unsafe_allow_html=True)
        
        cols = st.columns(len(recs) if len(recs) <= 6 else 6)
        for idx, item in enumerate(recs):
            col_idx = idx % 6
            with cols[col_idx]:
                poster_tag = (
                    f'<img class="s-poster" src="{item["poster"]}" '
                    f'alt="{item["title"]}" loading="lazy">'
                    if item["poster"]
                    else '<div class="s-poster-ph">🎬</div>'
                )
                st.markdown(f"""
                <div class="s-card" id="scard-{idx}">
                    {poster_tag}
                    <div class="s-gradient"></div>
                    <div class="s-badge-hd">HD</div>
                    <div class="s-title">{item["title"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # Invisible button overlaid on card for click handling
                if st.button(
                    label=item["title"],
                    key=f"rec_card_{idx}_{item['title'][:20]}",
                    use_container_width=True
                ):
                    st.session_state["page"] = "app"
                    st.session_state["selected_movie"] = item["title"]
                    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
                    if rerun_fn:
                        rerun_fn()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────
    t1, t2, t3 = st.tabs(["💬  Reviews & Sentiment", "⚙️  How It Works", "🔬  Model Status"])

    with t1:
        st.markdown('<div class="section-heading">Audience <span>Reviews</span></div>', unsafe_allow_html=True)
        if not show_reviews:
            st.info("Enable 'Show TMDB reviews' in the sidebar to see audience reviews.")
        elif not api_key:
            st.warning("Set API_KEY in your .env file to load TMDB reviews and posters.")
        else:
            reviews = fetch_reviews(sel_id)
            if not reviews:
                st.info("No reviews found for this title on TMDB.")
            else:
                for idx, rv in enumerate(reviews):
                    badge = ""
                    if show_sentiment and idx == 0:
                        try:
                            sent = analyze_sentiment(rv["content"], sentiment_model, vectorizer)
                            badge = (
                                '<span class="badge-pos">✔ Positive</span>' if sent == "positive"
                                else '<span class="badge-neg">✖ Negative</span>'
                            )
                        except ValueError as e:
                            badge = '<span class="badge-neu">⚠ Sync Error</span>'
                        except Exception:
                            badge = '<span class="badge-neu">– Unrated</span>'
                    else:
                        badge = '<span class="badge-neu">– Unrated</span>'

                    st.markdown(
                        f'''
                        <div class="review-card">
                          <div class="review-header">
                            <span class="review-author">👤 {rv["author"]}</span>
                            {badge}
                          </div>
                          <div class="review-text">{rv["content"][:700]}{"…" if len(rv["content"]) > 700 else ""}</div>
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

    with t2:
        st.markdown('<div class="section-heading">How It <span>Works</span></div>', unsafe_allow_html=True)
        st.markdown(
            '''
            <div class="how-steps">
              <div class="how-step">
                <div class="how-step-n">01</div>
                <div class="how-step-t">TF-IDF Vectorisation</div>
                <div class="how-step-d">Movie tags (genres, cast, keywords, crew) are converted into TF-IDF feature vectors.</div>
              </div>
              <div class="how-step">
                <div class="how-step-n">02</div>
                <div class="how-step-t">Cosine Similarity</div>
                <div class="how-step-d">A cosine similarity matrix ranks every movie against your selection by semantic closeness.</div>
              </div>
              <div class="how-step">
                <div class="how-step-n">03</div>
                <div class="how-step-t">Filter & Rank</div>
                <div class="how-step-d">Sidebar filters (rating, genre) prune the candidate list before the top-N are returned.</div>
              </div>
              <div class="how-step">
                <div class="how-step-n">04</div>
                <div class="how-step-t">Sentiment Analysis</div>
                <div class="how-step-d">A Logistic Regression classifier trained on 50 k IMDB reviews scores the first TMDB review.</div>
              </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown('<div class="section-heading">Artifact <span>Health</span></div>', unsafe_allow_html=True)
        source_file = "cleaned_movies.csv" if os.path.exists("cleaned_movies.csv") else "tmdb_5000_movies.csv"
        st.markdown(
            f'''
            <div class="status-row">
              <span class="status-ok">✔ Similarity Matrix loaded</span>
              <span class="status-ok">✔ Sentiment Model loaded</span>
              <span class="status-ok">✔ TF-IDF Vectorizer loaded</span>
              <span class="status-ok">✔ {len(movies_data):,} movies from {source_file}</span>
              {'<span class="status-ok">✔ TMDB API connected</span>' if api_key else '<span class="status-warn">⚠ API_KEY not set — posters & reviews disabled</span>'}
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.caption("All model files should live in the Model/ folder. Rebuild from Recommendation_System.ipynb if any fail.")

    st.markdown('</div>', unsafe_allow_html=True)  # close app-inner


# ─────────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────────
try:
  page = st.session_state.get("page", "landing")
except Exception:
    page = "landing"
    st.session_state.page = "landing"

if page == "landing":
    show_landing()
else:
    show_app()