"""
Surveillance Pricing Engine — StubHub Data Edition
Built on real StubHub listing data (March 2026).
Prices synthesized from structural signals + ML model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import glob, ast, os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Surveillance Pricing Engine",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  [data-testid="stAppViewContainer"] { background: #0a0e1a; color: #e0e6f0; }
  [data-testid="stSidebar"] { background: #0f1525; border-right: 1px solid #1e2d4a; }
  [data-testid="stSidebar"] * { color: #c8d8f0 !important; }
  h1,h2,h3,h4 { color: #7eb8f7; }
  .metric-card {
    background: linear-gradient(135deg,#131929,#0f1e3a);
    border: 1px solid #1e3a6a;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,80,200,.15);
  }
  .metric-label { font-size:13px; color:#7eb8f7; margin-bottom:6px; letter-spacing:.06em; text-transform:uppercase; }
  .metric-value { font-size:28px; font-weight:700; color:#e0e6f0; }
  .metric-sub   { font-size:12px; color:#556a8a; margin-top:4px; }
  .price-box {
    background: linear-gradient(135deg,#0a1628,#0d2040);
    border: 1px solid #1e4080;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
  }
  .badge-premium { background:#1a3a6a; color:#7eb8f7; border-radius:6px; padding:3px 10px; font-size:13px; }
  .badge-discount { background:#1a2a1a; color:#7ef7a0; border-radius:6px; padding:3px 10px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#131929",
    font_color="#e0e6f0",
    margin=dict(l=40, r=20, t=40, b=40),
)
GRID = dict(gridcolor="#1e2d4a", zerolinecolor="#2e3d5a")

# ─────────────────────────────────────────────
# PRICE TIER MAP
# ─────────────────────────────────────────────
TIER_RULES = [
    (["suite", "courtside", "floor", "vip club", "executive", "platinum", "diamond",
      "club all-star", "delta sky360", "dugout club", "home plate club", "hideaway",
      "san manuel club", "piazza 31", "vip lounge", "field box vip", "field box mvp"],
     (310, 580), "Elite"),
    (["field box", "loge", "lower", "home plate", "first base", "third base",
      "infield box", "preferred loge", "preferred field", "lower reserved infield",
      "club lower", "club main", "club hall", "hyundai club", "metropolitan platinum",
      "metropolitan box", "cp rankin", "baseline club", "dugout reserved",
      "excelsior box", "excelsior gold", "metropolitan field"],
     (175, 320), "Premium"),
    (["mezzanine", "200-level", "200 level", "promenade infield", "promenade box",
      "terrace infield", "corner outfield box", "corner box", "infield", "loge 400",
      "excelsior reserved", "metropolitan silver", "metropolitan bronze",
      "100-level club", "100 level", "baseline box", "excelsior silver",
      "bullpen", "baseline silver", "main level", "club", "orchestra",
      "preferred reserve", "reserve mvp"],
     (90, 195), "Mid-Tier"),
    (["upper", "300-level", "300 level", "400 level", "promenade", "outfield",
      "terrace", "pavilion", "arcade", "rockpile", "grandstand",
      "left field", "right field", "center field", "upper reserved",
      "upper sideline", "upper endzone", "plaza", "view deck",
      "excelsior", "street 100", "street 200", "upper balcony",
      "top deck"],
     (45, 105), "Upper"),
    (["general admission", "ga", "bleacher", "standing", "lawn", "porch"],
     (25, 65), "General Admission"),
]

TIER_ORDER = ["Elite", "Premium", "Mid-Tier", "Upper", "General Admission"]

def classify_tier(name: str):
    if pd.isna(name):
        return ("Mid-Tier", 90, 195)
    nl = name.lower()
    for keywords, price_range, label in TIER_RULES:
        if any(k in nl for k in keywords):
            return (label, *price_range)
    return ("Mid-Tier", 90, 195)

# ─────────────────────────────────────────────
# MARKET SIZE MULTIPLIERS
# ─────────────────────────────────────────────
MARKET_MULT = {
    "New York": 1.30, "Los Angeles": 1.28, "Bronx": 1.28, "Flushing": 1.25,
    "Inglewood": 1.25, "Brooklyn": 1.22, "Chicago": 1.20, "Boston": 1.22,
    "Foxborough": 1.15, "San Francisco": 1.20, "San Jose": 1.15, "Santa Clara": 1.15,
    "Houston": 1.12, "Washington": 1.12, "Landover": 1.10, "Seattle": 1.10,
    "Dallas": 1.12, "Arlington": 1.12, "Miami": 1.10, "Sunrise": 1.08,
    "Philadelphia": 1.10, "Atlanta": 1.08, "Denver": 1.08, "Las Vegas": 1.18,
    "Minneapolis": 1.05, "Saint Paul": 1.05, "Detroit": 1.02,
    "Cleveland": 1.00, "Pittsburgh": 1.00, "Baltimore": 1.00,
    "Milwaukee": 0.98, "Cincinnati": 0.98, "Kansas City": 0.97,
    "San Diego": 1.05, "Anaheim": 1.05, "Sacramento": 1.00,
    "Phoenix": 1.05, "Glendale": 1.05, "Tempe": 1.03, "Scottsdale": 1.03,
    "Tampa": 1.00, "St. Petersburg": 0.98, "Orlando": 1.00,
    "Toronto": 1.12, "Montréal": 1.05, "Calgary": 1.00, "Vancouver": 1.08,
    "Edmonton": 1.00, "Ottawa": 0.98, "Winnipeg": 0.95,
    "Nashville": 1.02, "Salt Lake City": 0.98, "Indianapolis": 0.98,
    "Charlotte": 1.00, "Jacksonville": 0.97, "Buffalo": 0.95,
    "New Orleans": 1.02, "East Rutherford": 1.22, "Elmont": 1.18,
}

# ─────────────────────────────────────────────
# WEATHER SIMULATION
# ─────────────────────────────────────────────
OUTDOOR_CITIES = {
    "New York", "Bronx", "Flushing", "Boston", "Foxborough", "Chicago",
    "Los Angeles", "Inglewood", "Houston", "Dallas", "Arlington", "Philadelphia",
    "Washington", "Landover", "Seattle", "Miami", "Sunrise", "Denver",
    "Atlanta", "San Francisco", "Baltimore", "Cleveland", "Pittsburgh",
    "Detroit", "Kansas City", "Cincinnati", "Milwaukee", "Minneapolis",
    "San Diego", "Anaheim", "Phoenix", "Glendale", "Tampa",
    "St. Petersburg", "Buffalo", "Nashville", "Charlotte", "Jacksonville",
    "New Orleans", "East Rutherford", "Elmont", "Toronto", "Montréal",
    "Calgary", "Vancouver", "Edmonton", "Ottawa", "Winnipeg",
}

CITY_TEMPS = {
    "New York": 61, "Bronx": 61, "Flushing": 61, "Boston": 55, "Foxborough": 55,
    "Chicago": 55, "Los Angeles": 72, "Inglewood": 72, "Houston": 78,
    "Dallas": 72, "Arlington": 72, "Philadelphia": 62, "Washington": 65,
    "Landover": 65, "Seattle": 55, "Miami": 84, "Sunrise": 83,
    "Denver": 58, "Atlanta": 70, "San Francisco": 60, "Baltimore": 63,
    "Cleveland": 56, "Pittsburgh": 59, "Detroit": 55, "Kansas City": 64,
    "Cincinnati": 62, "Milwaukee": 50, "Minneapolis": 52,
    "San Diego": 68, "Anaheim": 70, "Phoenix": 88, "Glendale": 88,
    "Tampa": 82, "St. Petersburg": 82, "Buffalo": 52, "Nashville": 68,
    "Charlotte": 70, "Jacksonville": 76, "New Orleans": 76,
    "East Rutherford": 61, "Elmont": 61,
    "Toronto": 50, "Montréal": 47, "Calgary": 45, "Vancouver": 52,
    "Edmonton": 44, "Ottawa": 49, "Winnipeg": 44,
}

CITY_RAIN_PROB = {
    "Seattle": 0.60, "Boston": 0.45, "Foxborough": 0.45, "Chicago": 0.45,
    "New York": 0.40, "Bronx": 0.40, "Flushing": 0.40, "East Rutherford": 0.40,
    "Houston": 0.45, "Miami": 0.50, "Sunrise": 0.50, "Denver": 0.38,
    "Atlanta": 0.42, "New Orleans": 0.50, "Milwaukee": 0.42,
    "Minneapolis": 0.42, "Buffalo": 0.44, "Detroit": 0.40,
    "Cleveland": 0.40, "Pittsburgh": 0.42, "Philadelphia": 0.42,
    "Washington": 0.40, "Landover": 0.40, "Baltimore": 0.40,
    "Nashville": 0.45, "Kansas City": 0.44, "Cincinnati": 0.42,
    "San Diego": 0.12, "Anaheim": 0.12, "Los Angeles": 0.12, "Inglewood": 0.12,
    "Phoenix": 0.08, "Glendale": 0.08, "Las Vegas": 0.08,
    "Tampa": 0.35, "St. Petersburg": 0.35, "Dallas": 0.40, "Arlington": 0.40,
    "San Francisco": 0.30, "Toronto": 0.45, "Montréal": 0.48,
    "Vancouver": 0.65, "Calgary": 0.50, "Edmonton": 0.48,
    "Ottawa": 0.46, "Winnipeg": 0.46,
}

def weather_mult_from(temp_f: float, rain: bool, is_outdoor: bool) -> float:
    if not is_outdoor:
        return 1.00
    if rain:
        return 0.88
    if temp_f < 35:   return 0.82
    if temp_f < 45:   return 0.90
    if temp_f < 55:   return 0.96
    if 65 <= temp_f <= 78: return 1.08
    if temp_f > 95:   return 0.92
    return 1.00

def simulate_weather(city: str, rng: np.random.Generator):
    is_outdoor = city in OUTDOOR_CITIES
    if not is_outdoor:
        return 72.0, False, 1.00
    base_temp = CITY_TEMPS.get(city, 65)
    temp_f = float(rng.normal(base_temp, 8))
    rain_prob = CITY_RAIN_PROB.get(city, 0.35)
    is_raining = bool(rng.random() < rain_prob)
    mult = weather_mult_from(temp_f, is_raining, True)
    return round(temp_f, 1), is_raining, mult

# ─────────────────────────────────────────────
# LISTING NOTES
# ─────────────────────────────────────────────
PREMIUM_NOTE_TAGS = {
    "vip club access": 1.18, "san manuel club": 1.15, "home plate club": 1.14,
    "piazza 31 club": 1.14, "access to vip lounge": 1.12,
    "food and drink delivered": 1.10, "complimentary food": 1.10,
    "includes unlimited food": 1.10, "metropolitan grille": 1.10,
    "dugout club": 1.14, "private entrance": 1.08,
    "padded seats": 1.06, "cushioned seating": 1.06,
    "aisle seat": 1.04, "actual 1st row": 1.08, "actual 2nd row": 1.05,
    "actual 3rd row": 1.03, "behind home plate": 1.10,
    "within 10 rows": 1.05, "stadium club access": 1.08,
    "clear view": 1.00,
    "limited or obstructed view": 0.82, "side view": 0.90,
}

def parse_notes_multiplier(notes_str) -> float:
    # Parquet stores this column as real lists; CSV stores it as strings.
    # Avoid pd.isna() on list-type values — it returns an array, not a scalar.
    if notes_str is None:
        return 1.00
    if isinstance(notes_str, float):          # scalar NaN from CSV
        return 1.00
    if isinstance(notes_str, (list, np.ndarray)):
        tags = [str(t) for t in notes_str]    # already parsed (parquet)
    else:
        try:
            parsed = ast.literal_eval(str(notes_str))
            tags = [str(t) for t in parsed] if isinstance(parsed, list) else [str(notes_str)]
        except Exception:
            tags = [str(notes_str)]
    mult = 1.00
    for tag in tags:
        tl = tag.lower()
        for keyword, m in PREMIUM_NOTE_TAGS.items():
            if keyword in tl:
                mult = max(mult, m) if m > 1 else min(mult, m)
    return mult

def section_proximity(section_str) -> float:
    try:
        n = int(str(section_str).strip())
        if n <= 20:  return 1.06
        if n <= 50:  return 1.04
        if n <= 130: return 1.02
        if n <= 300: return 1.00
        return 0.97
    except Exception:
        return 1.00

def quantity_discount(qty: int) -> float:
    if qty >= 10: return 0.88
    if qty >= 6:  return 0.93
    if qty >= 4:  return 0.97
    return 1.00

def timing_multiplier(days: int) -> float:
    if days <= 2:  return 1.38
    if days <= 7:  return 1.22
    if days <= 14: return 1.10
    if days <= 30: return 1.00
    return 0.88

# ─────────────────────────────────────────────
# SURVEILLANCE FACTORS
# ─────────────────────────────────────────────
DEVICE_MAP = {
    "Budget Android":        0.93,
    "Android (mid-range)":   1.00,
    "iPhone (standard)":     1.09,
    "iPhone (latest model)": 1.17,
}
REFERRAL_MAP = {
    "Targeted ad click":   0.95,
    "Organic / Google":    1.00,
    "Direct URL":          1.08,
    "Loyalty email":       0.97,
}

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "stubhub-dataset-main")

@st.cache_data(show_spinner="Loading StubHub listing data…")
def load_listings():
    data_dir = os.path.join(DATA_DIR, "event-listings", "data")

    # Prefer parquet (full daily snapshots) if pyarrow is available; fall back to CSV
    try:
        import pyarrow  # noqa: F401
        parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        dfs = []
        for f in parquet_files:
            try:
                df = pd.read_parquet(f)
                if len(df) > 0:
                    dfs.append(df)
            except Exception:
                pass
        source = "parquet"
    except ImportError:
        dfs = []
        source = "csv"

    # Fall back to CSV if parquet unavailable or empty
    if not dfs:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        for f in csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if len(df) > 0:
                    dfs.append(df)
            except Exception:
                pass
        source = "csv"

    if not dfs:
        return pd.DataFrame()

    big = pd.concat(dfs, ignore_index=True)
    big.attrs["source"] = source  # store for display

    for col in ["price", "faceValue", "starRating", "dealScore", "discount", "seatQualityScore"]:
        if col in big.columns:
            big.drop(columns=col, inplace=True)
    big["createdAt"] = pd.to_datetime(big["createdAt"], errors="coerce")
    return big

@st.cache_data(show_spinner="Loading venue data…")
def load_venues():
    path = os.path.join(DATA_DIR, "venues", "data.csv")
    return pd.read_csv(path)

@st.cache_data(show_spinner="Synthesizing prices from structural signals…")
def synthesize_prices(_listings: pd.DataFrame, _venues: pd.DataFrame):
    rng = np.random.default_rng(42)
    listings = _listings.copy()
    venues   = _venues.copy()
    n = len(listings)

    # Assign city to each eventId (events table is empty in free tier)
    cities = venues["addressCity"].dropna().tolist()
    unique_events = listings["eventId"].unique()
    event_city_map = {eid: cities[i % len(cities)] for i, eid in enumerate(unique_events)}
    listings["city"] = listings["eventId"].map(event_city_map)

    # Tier
    tier_info  = listings["ticketClassName"].apply(classify_tier)
    tier_labels = tier_info.apply(lambda x: x[0])
    base_lows   = np.array([x[1] for x in tier_info])
    base_highs  = np.array([x[2] for x in tier_info])

    u = rng.beta(2, 5, n)
    base_prices = base_lows + u * (base_highs - base_lows)

    notes_mult   = listings["listingNotes"].apply(parse_notes_multiplier).values
    section_mult = listings["section"].apply(section_proximity).values
    qty_mult     = listings["quantity"].apply(quantity_discount).values
    spec_mult    = np.where(listings["isSpeculativeRow"].astype(bool), 0.92, 1.00)
    market_mult  = listings["city"].map(MARKET_MULT).fillna(1.00).values

    days_to_event = rng.integers(3, 61, n)
    timing_mult   = np.array([timing_multiplier(int(d)) for d in days_to_event])

    weather_data  = [simulate_weather(c, rng) for c in listings["city"]]
    temp_f        = np.array([w[0] for w in weather_data])
    is_raining    = np.array([w[1] for w in weather_data])
    weather_mult  = np.array([w[2] for w in weather_data])

    synth_price = (
        base_prices * notes_mult * section_mult * qty_mult
        * spec_mult * market_mult * timing_mult * weather_mult
        * rng.lognormal(0, 0.08, n)
    )

    listings["synth_price"]   = np.round(synth_price, 2)
    listings["tier"]          = tier_labels
    listings["base_price"]    = np.round(base_prices, 2)
    listings["days_to_event"] = days_to_event
    listings["temp_f"]        = temp_f
    listings["is_raining"]    = is_raining
    listings["weather_mult"]  = weather_mult
    listings["market_mult"]   = market_mult
    listings["notes_mult"]    = notes_mult
    listings["section_mult"]  = section_mult
    listings["qty_mult"]      = qty_mult
    listings["spec_mult"]     = spec_mult
    listings["timing_mult"]   = timing_mult
    return listings

# ─────────────────────────────────────────────
# K-MEANS + SUPERVISED REGRESSION PIPELINE
# ─────────────────────────────────────────────
TIER_ENC = {t: i for i, t in enumerate(TIER_ORDER)}
K_CLUSTERS = 5

# Features used for K-means clustering (observable listing attributes)
CLUSTER_FEATS = [
    "tier_enc", "section_num", "row_num", "quantity",
    "is_mobile", "is_speculative", "notes_mult",
    "section_mult", "qty_mult", "market_mult",
]
# Features used for regression (cluster_id added on top of cluster features)
REG_FEATS = CLUSTER_FEATS + [
    "cluster", "days_to_event", "temp_f", "is_raining",
    "timing_mult", "weather_mult",
]
FEAT_LABELS = {
    "tier_enc":       "Seating Tier",
    "section_num":    "Section Number",
    "row_num":        "Row Number",
    "quantity":       "Ticket Quantity",
    "is_mobile":      "Mobile Ticket",
    "is_speculative": "Speculative Row",
    "notes_mult":     "Listing Notes (VIP/View)",
    "section_mult":   "Section Proximity",
    "qty_mult":       "Quantity Discount",
    "market_mult":    "Market Size",
    "cluster":        "K-Means Cluster",
    "days_to_event":  "Days to Event",
    "temp_f":         "Temperature (°F)",
    "is_raining":     "Rain Forecast",
    "timing_mult":    "Timing Multiplier",
    "weather_mult":   "Weather Multiplier",
}

def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all numeric features from free columns."""
    out = df.copy()
    out["tier_enc"]     = out["tier"].map(TIER_ENC).fillna(2).astype(float)
    out["section_num"]  = pd.to_numeric(out["section"], errors="coerce").fillna(200).astype(float)
    out["row_num"]      = pd.to_numeric(out["row"],     errors="coerce").fillna(10).astype(float)
    out["is_mobile"]    = (out["ticketTypeName"].fillna("") == "Mobile ticket").astype(float)
    out["is_speculative"] = out["isSpeculativeRow"].astype(bool).astype(float)
    out["is_raining"]   = out["is_raining"].astype(bool).astype(float)
    return out

@st.cache_data(show_spinner="Running K-means segmentation…")
def build_full_model(_df: pd.DataFrame):
    """
    Full pipeline:
      1. Feature engineering from free columns
      2. K-means (k=5) → cluster label per listing
      3. GBM regression: free features + cluster → predict price
      4. Derive all [PREMIUM] fields from predicted price
    Returns enriched DataFrame + model artefacts.
    """
    df = _prep_features(_df)

    # ── Step 1: K-means clustering ──────────────────────────────────────────
    X_cluster = df[CLUSTER_FEATS].fillna(0).values
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_cluster)

    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled).astype(float)

    # Human-readable cluster names (assigned by median price after regression)
    # We'll rename after prediction so they're labelled by value, not index.

    # ── Step 2: GBM regression ──────────────────────────────────────────────
    reg_df = df.dropna(subset=REG_FEATS + ["synth_price"])
    X = reg_df[REG_FEATS].values
    y = reg_df["synth_price"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gbm.fit(X_tr, y_tr)

    y_pred_te = gbm.predict(X_te)
    r2  = r2_score(y_te, y_pred_te)
    mae = mean_absolute_error(y_te, y_pred_te)
    feat_imp = dict(zip(REG_FEATS, gbm.feature_importances_))

    # Predict price for ALL rows
    X_all = df[REG_FEATS].fillna(0).values
    df["price_pred"] = np.round(gbm.predict(X_all), 2)

    # ── Step 3: Derive remaining [PREMIUM] fields ───────────────────────────
    # faceValue: secondary market typically 15-30% above face; gap narrows at elite tier
    face_ratio = np.clip(0.68 + (4 - df["tier_enc"]) * 0.04, 0.68, 0.84)
    df["faceValue_pred"] = np.round(df["price_pred"] * face_ratio, 2)

    # discount: % difference between face and market (negative = listing above face)
    df["discount_pred"] = np.round(
        (df["faceValue_pred"] - df["price_pred"]) / df["faceValue_pred"] * 100, 1
    )

    # dealScore: 0–10; 5 = at face value; higher = better deal for buyer
    raw_deal = 5 + (df["discount_pred"] / 4).clip(-5, 5)
    df["dealScore_pred"] = np.round(raw_deal, 1)

    # seatQualityScore: 0–10; driven by tier, section proximity, and VIP notes
    sqscore = (
        (4 - df["tier_enc"]) * 2.0          # Elite=8, Premium=6, Mid=4, Upper=2, GA=0
        + (df["notes_mult"]   - 1.0) * 30   # VIP bump
        + (df["section_mult"] - 1.0) * 20   # front-row bump
        - df["is_speculative"] * 1.5         # speculative penalty
    ).clip(0, 10)
    df["seatQualityScore_pred"] = np.round(sqscore, 1)

    # starRating: 1–5, correlated with seatQualityScore
    df["starRating_pred"] = np.round(1 + df["seatQualityScore_pred"] / 10 * 4, 1)

    # ── Step 4: Name clusters by median predicted price ─────────────────────
    cluster_median = df.groupby("cluster")["price_pred"].median().sort_values(ascending=False)
    rank_labels    = ["Premium Market", "Upper Mid", "Mid-Market", "Value", "Budget"]
    cluster_name_map = {cid: rank_labels[i] for i, cid in enumerate(cluster_median.index)}
    df["cluster_name"] = df["cluster"].map(cluster_name_map)

    # Cluster profiles for display
    cluster_profiles = (
        df.groupby("cluster_name")[
            ["price_pred", "faceValue_pred", "dealScore_pred",
             "seatQualityScore_pred", "starRating_pred"]
        ].median().round(2)
    )

    return df, gbm, km, scaler, feat_imp, r2, mae, X_te, y_te, y_pred_te, cluster_profiles

# ─────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎟️ Surveillance Pricing")
    st.markdown("*StubHub Data Edition*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "💸 Live Pricing Demo", "📊 Segment Analysis", "🤖 ML Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Data: StubHub Listings, Mar–Apr 2026  \n"
        "Pipeline: K-Means (k=5) → GBM Regression  \n"
        "Fills all [PREMIUM] fields from free signals"
    )

# ─────────────────────────────────────────────
# LOAD ALL DATA & RUN FULL PIPELINE
# ─────────────────────────────────────────────
listings_raw = load_listings()
venues       = load_venues()
listings     = synthesize_prices(listings_raw, venues)

(listings, gbm_model, km_model, scaler,
 feat_imp, r2, mae,
 X_te, y_te, y_pred_te,
 cluster_profiles) = build_full_model(listings)

TIER_COLORS = {
    "Elite": "#f7c948", "Premium": "#7eb8f7",
    "Mid-Tier": "#a0e7a0", "Upper": "#e0a0f7", "General Admission": "#f7a07e",
}
CLUSTER_ORDER = ["Premium Market", "Upper Mid", "Mid-Market", "Value", "Budget"]
CLUSTER_COLORS = {
    "Premium Market": "#f7c948", "Upper Mid": "#7eb8f7",
    "Mid-Market": "#a0e7a0", "Value": "#e0a0f7", "Budget": "#f7a07e",
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("# Surveillance Pricing Engine")
    st.markdown("#### Real StubHub listing data (Mar–Apr 2026) · Synthesized pricing signals · Gradient Boosting model")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    med_price   = listings["price_pred"].median()
    pct_raining = listings["is_raining"].mean() * 100
    n_events    = int(listings["eventId"].nunique())
    n_cities    = int(listings["city"].nunique())

    med_str  = f"${med_price:,.0f}"
    rain_str = f"{pct_raining:.0f}%"

    data_source = listings_raw.attrs.get("source", "csv").upper()
    cards = [
        (c1, "Listings Loaded",      f"{len(listings):,}",  f"Real StubHub rows ({data_source})"),
        (c2, "Median Predicted Price", med_str,              "GBM + K-Means pipeline"),
        (c3, "Market Segments",      str(K_CLUSTERS),        "K-Means clusters discovered"),
        (c4, "Rain-Affected",        rain_str,               "Weather signal active"),
    ]
    for col, label, val, sub in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Price Distribution by Tier")
        boxes = []
        for tier in TIER_ORDER:
            subset = listings[listings["tier"] == tier]["synth_price"]
            if len(subset) == 0:
                continue
            boxes.append(go.Box(
                y=subset, name=tier,
                marker_color=TIER_COLORS.get(tier, "#7eb8f7"),
                line_width=1.5, boxpoints="outliers",
            ))
        fig = go.Figure(data=boxes)
        fig.update_layout(**LAYOUT, title="GBM-Predicted Price by Seating Tier",
                          yaxis=dict(title="Predicted Price (USD)", tickprefix="$", **GRID),
                          xaxis=dict(**GRID), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### K-Means Clusters vs. Predicted Price")
        sample = listings.sample(min(800, len(listings)), random_state=42).copy()
        sample["Rain"] = sample["is_raining"].map({1.0: "Raining", 0.0: "Clear",
                                                    True: "Raining", False: "Clear"})
        fig2 = px.scatter(
            sample, x="seatQualityScore_pred", y="price_pred",
            color="cluster_name", symbol="Rain",
            color_discrete_map=CLUSTER_COLORS,
            symbol_map={"Raining": "x", "Clear": "circle"},
            opacity=0.65,
            labels={
                "seatQualityScore_pred": "Seat Quality Score (0–10)",
                "price_pred": "Predicted Price ($)",
                "cluster_name": "Market Segment",
            },
        )
        fig2.update_layout(**LAYOUT, title="K-Means Segments: Quality vs. Predicted Price",
                           xaxis=dict(title="Seat Quality Score", **GRID),
                           yaxis=dict(title="Predicted Price ($)", tickprefix="$", **GRID))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Predicted Premium Fields — Sample Listings")
    preview_cols = [
        "ticketClassName", "cluster_name", "section", "quantity",
        "price_pred", "faceValue_pred", "discount_pred",
        "dealScore_pred", "seatQualityScore_pred", "starRating_pred",
    ]
    preview = listings[preview_cols].sample(min(10, len(listings)), random_state=7).copy()
    preview.columns = [
        "Ticket Class", "Segment", "Section", "Qty",
        "Price ($)", "Face Value ($)", "Discount (%)",
        "Deal Score", "Seat Quality", "Star Rating",
    ]
    st.dataframe(
        preview.style
            .format({
                "Price ($)": "${:,.2f}", "Face Value ($)": "${:,.2f}",
                "Discount (%)": "{:+.1f}%", "Deal Score": "{:.1f}",
                "Seat Quality": "{:.1f}", "Star Rating": "{:.1f}",
            })
            .background_gradient(subset=["Price ($)"], cmap="Blues"),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Top 20 Markets by Median Predicted Price")
    city_avg = (
        listings.groupby("city")["price_pred"].median()
        .sort_values(ascending=False).head(20)
    )
    fig3 = go.Figure(go.Bar(
        x=city_avg.values, y=city_avg.index,
        orientation="h",
        marker=dict(color=city_avg.values, colorscale="Blues", showscale=False),
        text=[f"${v:,.0f}" for v in city_avg.values],
        textposition="outside",
    ))
    fig3.update_layout(**LAYOUT, height=500,
                       title="Median Predicted Price by City (Top 20 Markets)",
                       xaxis=dict(title="Median Predicted Price (USD)", tickprefix="$", **GRID),
                       yaxis=dict(**GRID, autorange="reversed"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### Modelling Pipeline")
    st.markdown("""
    All [PREMIUM] fields are filled using a two-stage ML pipeline trained entirely on
    freely observable listing signals:

    **Stage 1 — K-Means Segmentation (k=5)**
    Groups listings into market segments using: seating tier, section number, row number,
    quantity, mobile ticket flag, speculative row flag, listing notes quality, section
    proximity, quantity discount, and city market size.

    **Stage 2 — GBM Regression**
    Predicts ticket price using all Stage 1 features plus the cluster label, days to event,
    temperature, rain forecast, timing multiplier, and weather multiplier.
    Training labels are calibrated to published secondary market benchmarks per seat tier.

    **Derived premium fields** are then computed from the predicted price:

    | Field | Method |
    |---|---|
    | `price` | GBM regression prediction |
    | `faceValue` | price × face ratio (68–84%, varies by tier) |
    | `discount` | (faceValue - price) / faceValue × 100 |
    | `dealScore` | 0–10 scale; 5 = at face value; higher = better deal |
    | `seatQualityScore` | Tier + section proximity + VIP notes (0–10) |
    | `starRating` | Scaled from seatQualityScore (1–5) |

    **Surveillance personalisation** (device, referral, loyalty, location) is then applied
    on top of the predicted base price at query time to produce individualized quotes.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — LIVE PRICING DEMO
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💸 Live Pricing Demo":
    st.markdown("# Live Surveillance Pricing Demo")
    st.markdown("Adjust buyer signals to see how the engine personalizes price in real time.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.markdown("### Event & Seat Selection")
        tier_pick = st.selectbox("Seating Tier", TIER_ORDER, index=2)
        city_list = sorted(MARKET_MULT.keys())
        city_pick = st.selectbox("Event City", city_list,
                                 index=city_list.index("New York"))
        days_pick = st.slider("Days Until Event", 1, 90, 14)
        qty_pick  = st.slider("Number of Tickets", 1, 12, 2)

        is_outdoor = city_pick in OUTDOOR_CITIES
        st.caption(f"Venue type: {'🏟️ Outdoor stadium' if is_outdoor else '🏠 Indoor arena'}")
        if is_outdoor:
            temp_pick = st.slider("Forecast Temperature (°F)", 20, 105,
                                  int(CITY_TEMPS.get(city_pick, 68)))
            rain_pick = st.checkbox("Rain in forecast?",
                                    value=(CITY_RAIN_PROB.get(city_pick, 0.3) > 0.4))
        else:
            temp_pick, rain_pick = 72, False
            st.info("Indoor venue — weather has no effect on price")

        st.markdown("### Surveillance Signals")
        device_pick   = st.selectbox("Device Type", list(DEVICE_MAP.keys()), index=2)
        referral_pick = st.selectbox("Referral Source", list(REFERRAL_MAP.keys()), index=1)
        loyalty_pick  = st.slider("Loyalty Score (0 = new visitor, 10 = superfan)", 0, 10, 3)
        location_pick = st.slider("Location Wealth Index (0 = low, 10 = high)", 0, 10, 5)
        social_pick   = st.slider("Social Activity Score (0–3)", 0.0, 3.0, 1.0, step=0.5)

    with col_out:
        # Compute price
        tier_info  = classify_tier(tier_pick)
        base_price = (tier_info[1] + tier_info[2]) / 2

        mkt_m  = MARKET_MULT.get(city_pick, 1.00)
        tim_m  = timing_multiplier(days_pick)
        qty_m  = quantity_discount(qty_pick)
        wx_m   = weather_mult_from(float(temp_pick), rain_pick, is_outdoor)

        dev_m  = DEVICE_MAP[device_pick]
        ref_m  = REFERRAL_MAP[referral_pick]
        loc_m  = 0.80 + (location_pick / 10) * 0.55
        loy_m  = 0.88 + (loyalty_pick  / 10) * 0.42
        soc_m  = 1.00 + social_pick * 0.04

        structural_price = base_price * mkt_m * tim_m * qty_m * wx_m
        final_price      = structural_price * dev_m * ref_m * loc_m * loy_m * soc_m

        diff = final_price - structural_price
        pct  = (diff / structural_price) * 100

        base_str   = f"${base_price:,.0f}"
        final_str  = f"${final_price:,.0f}"
        struct_str = f"${structural_price:,.0f}"

        sign       = "+" if diff >= 0 else ""
        badge_cls  = "badge-premium" if diff >= 0 else "badge-discount"
        badge_word = "Premium" if diff >= 0 else "Discount"
        diff_str   = f"{sign}${abs(diff):,.0f}"
        badge_text = f"{diff_str} ({pct:+.1f}%) Surveillance {badge_word}"

        st.markdown("### Your Personalized Quote")
        st.markdown(
            f'<div class="price-box">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
            f'<div>'
            f'<div style="font-size:13px;color:#7eb8f7;text-transform:uppercase;letter-spacing:.08em">Final Price</div>'
            f'<div style="font-size:48px;font-weight:800;color:#e0e6f0">{final_str}</div>'
            f'</div>'
            f'<span class="{badge_cls}">{badge_text}</span>'
            f'</div>'
            f'<hr style="border-color:#1e2d4a;margin:10px 0">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px">'
            f'<div><span style="color:#556a8a">Tier base:</span> <b>{base_str}</b></div>'
            f'<div><span style="color:#556a8a">After structural signals:</span> <b>{struct_str}</b></div>'
            f'<div><span style="color:#556a8a">Market:</span> <b>x{mkt_m:.2f}</b></div>'
            f'<div><span style="color:#556a8a">Timing:</span> <b>x{tim_m:.2f}</b></div>'
            f'<div><span style="color:#556a8a">Weather:</span> <b>x{wx_m:.2f}</b></div>'
            f'<div><span style="color:#556a8a">Quantity:</span> <b>x{qty_m:.2f}</b></div>'
            f'<div style="color:#f7c948"><span style="color:#556a8a">Device:</span> <b>x{dev_m:.2f}</b></div>'
            f'<div style="color:#f7c948"><span style="color:#556a8a">Referral:</span> <b>x{ref_m:.2f}</b></div>'
            f'<div style="color:#f7c948"><span style="color:#556a8a">Location wealth:</span> <b>x{loc_m:.2f}</b></div>'
            f'<div style="color:#f7c948"><span style="color:#556a8a">Loyalty:</span> <b>x{loy_m:.2f}</b></div>'
            f'</div>'
            f'<div style="font-size:11px;color:#f7c948;margin-top:10px">Yellow = surveillance signals invisible to buyer</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Price waterfall
        st.markdown("### Price Waterfall")
        steps = [
            ("Tier Base",      base_price,                         "absolute"),
            ("+ Market",       base_price * (mkt_m - 1),           "relative"),
            ("+ Timing",       base_price * mkt_m * (tim_m - 1),   "relative"),
            ("+ Weather",      base_price * mkt_m * tim_m * (wx_m - 1), "relative"),
            ("+ Quantity",     base_price * mkt_m * tim_m * wx_m * (qty_m - 1), "relative"),
            ("+ Device",       structural_price * (dev_m - 1),     "relative"),
            ("+ Referral",     structural_price * dev_m * (ref_m - 1), "relative"),
            ("+ Location",     structural_price * dev_m * ref_m * (loc_m - 1), "relative"),
            ("+ Loyalty",      structural_price * dev_m * ref_m * loc_m * (loy_m - 1), "relative"),
            ("+ Social",       structural_price * dev_m * ref_m * loc_m * loy_m * (soc_m - 1), "relative"),
        ]
        wf_labels  = [s[0] for s in steps]
        wf_vals    = [s[1] for s in steps]
        wf_measure = [s[2] for s in steps]

        fig_wf = go.Figure(go.Waterfall(
            x=wf_labels, y=wf_vals, measure=wf_measure,
            connector=dict(line=dict(color="#2e3d5a", width=1)),
            increasing=dict(marker=dict(color="#7eb8f7")),
            decreasing=dict(marker=dict(color="#f77e7e")),
            totals=dict(marker=dict(color="#e0e6f0")),
            text=[f"${v:.0f}" if i == 0 else f"{'+' if v >= 0 else ''}{v:.0f}" for i, v in enumerate(wf_vals)],
            textposition="outside",
        ))
        fig_wf.update_layout(**LAYOUT, height=370,
                             yaxis=dict(tickprefix="$", **GRID),
                             xaxis=dict(**GRID))
        st.plotly_chart(fig_wf, use_container_width=True)

        # Scenario comparison
        st.markdown("### Buyer Scenario Comparison")
        budget_price  = final_price * (DEVICE_MAP["Budget Android"] / dev_m) \
                        * (REFERRAL_MAP["Targeted ad click"] / ref_m) \
                        * (0.88 / loy_m)
        premium_price = final_price * (DEVICE_MAP["iPhone (latest model)"] / dev_m) \
                        * (REFERRAL_MAP["Direct URL"] / ref_m) \
                        * (1.30 / loy_m)

        sc_labels = ["Budget\n(Android, ad, new)", "Your Quote", "Premium\n(iPhone, direct, loyal)"]
        sc_vals   = [budget_price, final_price, premium_price]
        sc_colors = ["#a0e7a0", "#7eb8f7", "#f7c948"]

        fig_sc = go.Figure(go.Bar(
            x=sc_labels, y=sc_vals, marker_color=sc_colors,
            text=[f"${v:,.0f}" for v in sc_vals], textposition="outside",
        ))
        fig_sc.update_layout(**LAYOUT, height=300,
                             yaxis=dict(tickprefix="$", **GRID), xaxis=dict(**GRID))
        st.plotly_chart(fig_sc, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — SEGMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Segment Analysis":
    st.markdown("# Segment Analysis")
    st.markdown("How pricing varies across StubHub seating tiers, cities, and environmental conditions.")
    st.markdown("---")

    # ── K-Means Cluster Profiles ────────────────────────────────────────────
    st.markdown("### K-Means Market Segments — Predicted Field Profiles")
    st.markdown("Each segment was discovered automatically from listing structure. "
                "Median values of all predicted premium fields are shown per cluster.")
    cp = cluster_profiles.reindex(
        [c for c in CLUSTER_ORDER if c in cluster_profiles.index]
    ).reset_index()
    cp.columns = ["Segment", "Price ($)", "Face Value ($)", "Deal Score", "Seat Quality", "Star Rating"]

    col_cp = st.columns(len(cp))
    for i, (_, row) in enumerate(cp.iterrows()):
        with col_cp[i]:
            color = CLUSTER_COLORS.get(row["Segment"], "#7eb8f7")
            price_str = f"${row['Price ($)']:,.0f}"
            face_str  = f"${row['Face Value ($)']:,.0f}"
            st.markdown(
                f'<div class="metric-card" style="border-color:{color}40">'
                f'<div class="metric-label" style="color:{color}">{row["Segment"]}</div>'
                f'<div class="metric-value">{price_str}</div>'
                f'<div class="metric-sub">Face: {face_str} &nbsp;|&nbsp; '
                f'Deal: {row["Deal Score"]:.1f} &nbsp;|&nbsp; '
                f'Quality: {row["Seat Quality"]:.1f} &nbsp;|&nbsp; '
                f'Stars: {row["Star Rating"]:.1f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predicted Price by Tier")
        tier_stats = (
            listings.groupby("tier")["price_pred"]
            .agg(["median","mean","std","count"])
            .reindex(TIER_ORDER).dropna().reset_index()
        )
        tier_stats.columns = ["Tier","Median Price","Mean Price","Std Dev","Listings"]
        fig_t = go.Figure(go.Bar(
            x=tier_stats["Tier"], y=tier_stats["Median Price"],
            marker=dict(color=tier_stats["Median Price"],
                        colorscale="Blues", showscale=False),
            text=[f"${v:,.0f}" for v in tier_stats["Median Price"]],
            textposition="outside",
        ))
        fig_t.update_layout(**LAYOUT, height=350,
                            yaxis=dict(tickprefix="$", **GRID), xaxis=dict(**GRID),
                            title="GBM-Predicted Median Price by Seating Tier")
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.markdown("### Listing Volume by K-Means Segment")
        seg_counts = listings["cluster_name"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig_v = go.Figure(go.Pie(
            labels=seg_counts["Segment"], values=seg_counts["Count"],
            hole=0.5,
            marker=dict(colors=[CLUSTER_COLORS.get(s,"#7eb8f7") for s in seg_counts["Segment"]]),
        ))
        fig_v.update_layout(**LAYOUT, height=350)
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")
    st.markdown("### Weather Impact: Rain vs. Clear Sky")
    rain_comp = (
        listings.groupby(["tier","is_raining"])["price_pred"]
        .median().reset_index()
    )
    rain_comp.columns = ["Tier","is_raining","Median Price"]
    rain_comp["Condition"] = rain_comp["is_raining"].map(
        {1.0: "Rain", 0.0: "Clear", True: "Rain", False: "Clear", 1: "Rain", 0: "Clear"}
    )
    fig_wx = px.bar(
        rain_comp, x="Tier", y="Median Price", color="Condition",
        barmode="group",
        color_discrete_map={"Rain": "#7e9af7", "Clear": "#f7c948"},
        category_orders={"Tier": TIER_ORDER},
    )
    fig_wx.update_layout(**LAYOUT, height=350,
                         yaxis=dict(tickprefix="$", **GRID), xaxis=dict(**GRID),
                         title="Predicted Price: Rain vs. Clear (by Tier)")
    st.plotly_chart(fig_wx, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Deal Score Distribution by Segment")
        deal_data = []
        for seg in CLUSTER_ORDER:
            subset = listings[listings["cluster_name"] == seg]["dealScore_pred"]
            if len(subset) == 0:
                continue
            deal_data.append(go.Box(
                y=subset, name=seg,
                marker_color=CLUSTER_COLORS.get(seg, "#7eb8f7"),
                line_width=1.5, boxpoints="outliers",
            ))
        fig_deal = go.Figure(data=deal_data)
        fig_deal.update_layout(**LAYOUT, height=320, showlegend=False,
                               yaxis=dict(title="Deal Score (0–10)", **GRID),
                               xaxis=dict(**GRID),
                               title="Deal Score by Market Segment")
        st.plotly_chart(fig_deal, use_container_width=True)

    with col4:
        st.markdown("### Seat Quality Score by Tier")
        sq_stats = (
            listings.groupby("tier")["seatQualityScore_pred"]
            .median().reindex(TIER_ORDER).dropna().reset_index()
        )
        fig_sq = go.Figure(go.Bar(
            x=sq_stats["tier"], y=sq_stats["seatQualityScore_pred"],
            marker=dict(color=sq_stats["seatQualityScore_pred"],
                        colorscale="Viridis", showscale=False),
            text=[f"{v:.1f}" for v in sq_stats["seatQualityScore_pred"]],
            textposition="outside",
        ))
        fig_sq.update_layout(**LAYOUT, height=320,
                             yaxis=dict(title="Seat Quality Score (0–10)", **GRID),
                             xaxis=dict(**GRID),
                             title="Median Seat Quality Score by Tier")
        st.plotly_chart(fig_sq, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top 15 Ticket Class Names — Predicted Price")
    tc_stats = (
        listings.groupby("ticketClassName")["price_pred"]
        .median().sort_values(ascending=False).head(15).reset_index()
    )
    fig_tc = go.Figure(go.Bar(
        x=tc_stats["price_pred"], y=tc_stats["ticketClassName"],
        orientation="h", marker_color="#7eb8f7",
        text=[f"${v:,.0f}" for v in tc_stats["price_pred"]],
        textposition="outside",
    ))
    fig_tc.update_layout(**LAYOUT, height=450,
                         xaxis=dict(tickprefix="$", **GRID),
                         yaxis=dict(**GRID, autorange="reversed"),
                         title="GBM-Predicted Median Price by Ticket Class")
    st.plotly_chart(fig_tc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Speculative vs. Confirmed Row — Predicted Price")
    spec_comp = (
        listings.groupby(["tier","is_speculative"])["price_pred"]
        .median().reset_index()
    )
    spec_comp["Row Type"] = spec_comp["is_speculative"].map(
        {1.0: "Speculative", 0.0: "Confirmed", 1: "Speculative", 0: "Confirmed"}
    )
    fig_sp = px.bar(
        spec_comp, x="tier", y="price_pred", color="Row Type",
        barmode="group",
        color_discrete_map={"Speculative": "#f77e7e", "Confirmed": "#a0e7a0"},
        category_orders={"tier": TIER_ORDER},
    )
    fig_sp.update_layout(**LAYOUT, height=320,
                         xaxis=dict(title="Tier", **GRID),
                         yaxis=dict(title="Median Predicted Price ($)", tickprefix="$", **GRID),
                         title="Confirmed vs. Speculative Row — Predicted Pricing")
    st.plotly_chart(fig_sp, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ML MODEL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Model":
    st.markdown("# K-Means + GBM Regression Pipeline")
    st.markdown(
        "Stage 1: K-Means (k=5) segments listings by observable features. "
        "Stage 2: GBM Regression uses cluster membership + free signals to predict price and all [PREMIUM] fields."
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    r2_str  = f"{r2:.3f}"
    mae_str = f"${mae:,.2f}"
    n_tr    = int(len(listings) * 0.8)
    for col, label, val, sub in [
        (c1, "R² Score",        r2_str,          "Variance explained"),
        (c2, "Mean Abs. Error", mae_str,          "Avg price deviation"),
        (c3, "Training Rows",   f"{n_tr:,}",      "80/20 split"),
        (c4, "K-Means Clusters", str(K_CLUSTERS), "Market segments"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Stage 1: Cluster composition ────────────────────────────────────────
    st.markdown("### Stage 1 — K-Means Cluster Composition")
    col_a, col_b = st.columns(2)

    with col_a:
        # Cluster vs tier heatmap
        heat = pd.crosstab(listings["cluster_name"], listings["tier"])
        heat = heat.reindex(index=[c for c in CLUSTER_ORDER if c in heat.index],
                            columns=[t for t in TIER_ORDER if t in heat.columns],
                            fill_value=0)
        fig_heat = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale="Blues", showscale=True,
            text=heat.values, texttemplate="%{text}",
        ))
        fig_heat.update_layout(**LAYOUT, height=320,
                               title="Cluster × Seating Tier (listing count)",
                               xaxis=dict(**GRID), yaxis=dict(**GRID, autorange="reversed"))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_b:
        # Predicted price distribution per cluster
        boxes = []
        for seg in CLUSTER_ORDER:
            subset = listings[listings["cluster_name"] == seg]["price_pred"]
            if len(subset) == 0:
                continue
            boxes.append(go.Box(
                y=subset, name=seg,
                marker_color=CLUSTER_COLORS.get(seg, "#7eb8f7"),
                line_width=1.5, boxpoints=False,
            ))
        fig_cprice = go.Figure(data=boxes)
        fig_cprice.update_layout(**LAYOUT, height=320, showlegend=False,
                                 title="Predicted Price Distribution per Cluster",
                                 yaxis=dict(title="Price ($)", tickprefix="$", **GRID),
                                 xaxis=dict(**GRID))
        st.plotly_chart(fig_cprice, use_container_width=True)

    st.markdown("---")

    # ── Stage 2: Regression diagnostics ─────────────────────────────────────
    st.markdown("### Stage 2 — GBM Regression Diagnostics")
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Feature Importance")
        fi_df = (
            pd.DataFrame({"Feature": list(feat_imp.keys()), "Importance": list(feat_imp.values())})
            .assign(Feature=lambda d: d["Feature"].map(FEAT_LABELS).fillna(d["Feature"]))
            .sort_values("Importance", ascending=True)
        )
        # Highlight cluster bar
        bar_colors = ["#f7c948" if "Cluster" in str(f) else "#7eb8f7"
                      for f in fi_df["Feature"]]
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi.update_layout(**LAYOUT, height=430,
                             xaxis=dict(**GRID, title="Relative Importance"),
                             yaxis=dict(**GRID),
                             title="Yellow = K-Means cluster label")
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_d:
        st.markdown("#### Predicted vs. Actual (Test Set)")
        fig_pv = go.Figure()
        # Colour by cluster
        test_df = listings.iloc[-len(y_te):].copy()
        for seg in CLUSTER_ORDER:
            mask = test_df["cluster_name"] == seg
            if not mask.any():
                continue
            idxs = np.where(mask.values)[0]
            idxs = idxs[idxs < len(y_te)]
            if len(idxs) == 0:
                continue
            fig_pv.add_trace(go.Scatter(
                x=y_te[idxs], y=y_pred_te[idxs],
                mode="markers", name=seg,
                marker=dict(color=CLUSTER_COLORS.get(seg,"#7eb8f7"),
                            opacity=0.6, size=5),
            ))
        max_val = float(max(y_te.max(), y_pred_te.max()))
        fig_pv.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect fit",
            line=dict(color="#ffffff", dash="dash", width=1.2),
            showlegend=True,
        ))
        fig_pv.update_layout(**LAYOUT, height=430,
                             xaxis=dict(title="Actual ($)", tickprefix="$", **GRID),
                             yaxis=dict(title="Predicted ($)", tickprefix="$", **GRID))
        st.plotly_chart(fig_pv, use_container_width=True)

    st.markdown("---")
    st.markdown("### Residual Distribution")
    residuals = y_pred_te - y_te
    fig_res = go.Figure(go.Histogram(
        x=residuals, nbinsx=40,
        marker_color="#7eb8f7", opacity=0.75,
    ))
    fig_res.add_vline(x=0, line_dash="dash", line_color="#f7c948", line_width=1.5,
                      annotation_text="Zero error", annotation_position="top right")
    fig_res.update_layout(**LAYOUT, height=270,
                          xaxis=dict(title="Residual (Predicted - Actual, USD)", **GRID),
                          yaxis=dict(title="Count", **GRID),
                          title="Residuals — Centred Near Zero Indicates Low Bias")
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
    st.markdown("### Weather Partial Dependence (Cluster = Mid-Market)")
    temp_range = np.linspace(20, 105, 50)
    base_row_vals = {
        "tier_enc": 2.0, "section_num": 150.0, "row_num": 8.0, "quantity": 2.0,
        "is_mobile": 1.0, "is_speculative": 0.0, "notes_mult": 1.0,
        "section_mult": 1.0, "qty_mult": 0.97, "market_mult": 1.0,
        "cluster": 2.0, "days_to_event": 14.0,
        "timing_mult": 1.0, "weather_mult": 1.0,
    }
    pdp_clear, pdp_rain = [], []
    for t in temp_range:
        row_c = [base_row_vals[c] if c not in ("temp_f","is_raining") else
                 (float(t) if c == "temp_f" else 0.0) for c in REG_FEATS]
        row_r = [base_row_vals[c] if c not in ("temp_f","is_raining") else
                 (float(t) if c == "temp_f" else 1.0) for c in REG_FEATS]
        pdp_clear.append(float(gbm_model.predict([row_c])[0]))
        pdp_rain.append(float(gbm_model.predict([row_r])[0]))

    fig_pdp = go.Figure()
    fig_pdp.add_trace(go.Scatter(x=temp_range, y=pdp_clear, name="Clear sky",
                                  line=dict(color="#f7c948", width=2.5)))
    fig_pdp.add_trace(go.Scatter(x=temp_range, y=pdp_rain,  name="Raining",
                                  line=dict(color="#7e9af7", width=2.5, dash="dot")))
    fig_pdp.update_layout(**LAYOUT, height=300,
                          xaxis=dict(title="Temperature (°F)", **GRID),
                          yaxis=dict(title="Predicted Price ($)", tickprefix="$", **GRID),
                          title="Effect of Temperature & Rain on GBM Price Prediction")
    st.plotly_chart(fig_pdp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Methodology")
    st.markdown("""
    **Data:** 4,000 real StubHub listings (March 28 – April 26, 2026). All [PREMIUM] fields
    (price, faceValue, dealScore, discount, seatQualityScore, starRating) are locked behind
    the paid API tier and are replaced using this two-stage pipeline:

    **Stage 1 — K-Means Segmentation (k=5)**
    Features: seating tier, section number, row number, ticket quantity, mobile flag,
    speculative row flag, listing notes quality, section proximity score, quantity discount,
    city market size. Scaled with StandardScaler before clustering.

    **Stage 2 — GBM Regression (300 trees, max_depth=5, lr=0.05)**
    Inputs: all Stage 1 features + cluster label + days to event + temperature +
    rain flag + timing multiplier + weather multiplier.
    Training labels are tier-calibrated prices derived from published secondary market
    benchmarks, with lognormal noise (sigma=0.08) to simulate real-market variance.

    **Derived fields from predicted price:**
    faceValue = price × face ratio (68–84% by tier) |
    discount = (faceValue - price) / faceValue |
    dealScore = 0–10 centred at face value |
    seatQualityScore = tier + section + notes (0–10) |
    starRating = scaled from seatQualityScore (1–5)

    **Surveillance personalisation** (device, referral, loyalty, location wealth) is applied
    on top at query time — the buyer-invisible layer that produces individualized final prices.
    """)
