# app.py
# Twin Destiny Engine – public version for Streamlit Cloud
# 2025-02-16
# Refactored with timezone input, time step, date range

import streamlit as st
import datetime
import zoneinfo
import math
import random
import hashlib
from openai import OpenAI
import os

# ────────────────────────────────────────────────
#  Constants & Configuration
# ────────────────────────────────────────────────

st.set_page_config(page_title="Twin Destiny Engine", layout="wide")

MODEL_NAME = "grok-4-1-fast-reasoning"   # change to whatever model you have access to

THEMES = {
    "Mythic":    "ancient, legendary, timeless",
    "Sci-Fi":    "cosmic, futuristic, parallel universes",
    "Romantic":  "intimate, emotional, tender",
    "Dark":      "shadowed, mysterious, intense"
}

BRANCHES = {
    "tragic":       "🌑 Tragic Path",
    "heroic":       "🔥 Heroic Path",
    "transcendent": "🌌 Transcendent Path"
}

# ────────────────────────────────────────────────
#  Vedic Astrology Data & Functions
# ────────────────────────────────────────────────

RASHIS = [
    "Mesha","Vrishabha","Mithuna","Karka",
    "Simha","Kanya","Tula","Vrishchika",
    "Dhanu","Makara","Kumbha","Meena"
]

NAKSHATRAS = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra",
    "Punarvasu","Pushya","Ashlesha","Magha","Purvaphalguni",
    "Uttaraphalguni","Hasta","Chitra","Swati","Vishakha",
    "Anuradha","Jyeshta","Mula","Purvashada","Uttarashada",
    "Shravana","Dhanishta","Shatabhisha","Purvabhadra",
    "Uttarabhadra","Revati"
]

NAKSHATRA_RULERS = {
    "Ashwini": "Ketu", "Bharani": "Venus", "Krittika": "Sun",
    "Rohini": "Moon", "Mrigashira": "Mars", "Ardra": "Rahu",
    "Punarvasu": "Jupiter", "Pushya": "Saturn", "Ashlesha": "Mercury",
    "Magha": "Ketu", "Purvaphalguni": "Venus", "Uttaraphalguni": "Sun",
    "Hasta": "Moon", "Chitra": "Mars", "Swati": "Rahu",
    "Vishakha": "Jupiter", "Anuradha": "Saturn", "Jyeshta": "Mercury",
    "Mula": "Ketu", "Purvashada": "Venus", "Uttarashada": "Sun",
    "Shravana": "Moon", "Dhanishta": "Mars", "Shatabhisha": "Rahu",
    "Purvabhadra": "Jupiter", "Uttarabhadra": "Saturn", "Revati": "Mercury"
}

PLANET_BRANCH_WEIGHT = {
    "Sun": ("heroic", 1.0),     "Mars": ("heroic", 0.9),
    "Moon": ("tragic", 0.9),    "Venus": ("tragic", 0.8),
    "Saturn": ("tragic", 1.0),  "Rahu": ("tragic", 0.7),
    "Mercury": ("transcendent", 0.8),
    "Jupiter": ("transcendent", 1.0),
    "Ketu": ("transcendent", 1.1)
}

LAGNA_ELEMENT = {
    "Mesha":"Fire", "Simha":"Fire", "Dhanu":"Fire",
    "Vrishabha":"Earth", "Kanya":"Earth", "Makara":"Earth",
    "Mithuna":"Air", "Tula":"Air", "Kumbha":"Air",
    "Karka":"Water", "Vrishchika":"Water", "Meena":"Water"
}

ELEMENT_BRANCH_MAP = {"Fire":"heroic", "Air":"heroic", "Earth":"tragic", "Water":"transcendent"}

LAGNA_LORD = {
    "Mesha":"Mars", "Vrishabha":"Venus", "Mithuna":"Mercury",
    "Karka":"Moon", "Simha":"Sun", "Kanya":"Mercury",
    "Tula":"Venus", "Vrishchika":"Mars", "Dhanu":"Jupiter",
    "Makara":"Saturn", "Kumbha":"Saturn", "Meena":"Jupiter"
}

PLANET_BRANCH = {
    "Sun":"heroic", "Mars":"heroic", "Saturn":"tragic",
    "Moon":"transcendent", "Venus":"transcendent",
    "Mercury":"transcendent", "Jupiter":"transcendent"
}

SHUKLA_BIRDS = {
    "Eagle":   ["Ashwini","Bharani","Krittika","Rohini","Mrigashira"],
    "Owl":     ["Ardra","Punarvasu","Pushya","Ashlesha","Magha","Purvaphalguni"],
    "Raven":   ["Uttaraphalguni","Hasta","Chitra","Swati","Vishakha"],
    "Cock":    ["Anuradha","Jyeshta","Mula","Purvashada","Uttarashada"],
    "Peacock": ["Shravana","Dhanishta","Shatabhisha","Purvabhadra","Uttarabhadra","Revati"]
}

KRISHNA_BIRDS = {
    "Peacock": ["Ashwini","Bharani","Krittika","Rohini","Mrigashira"],
    "Cock":    ["Ardra","Punarvasu","Pushya","Ashlesha","Magha","Purvaphalguni"],
    "Raven":   ["Uttaraphalguni","Hasta","Chitra","Swati","Vishakha"],
    "Owl":     ["Anuradha","Jyeshta","Mula","Purvashada","Uttarashada"],
    "Eagle":   ["Shravana","Dhanishta","Shatabhisha","Purvabhadra","Uttarabhadra","Revati"]
}

PANCHAPAKSHI_BRANCH_MAP = {
    "Eagle":   ("tragic", 1.0),
    "Owl":     ("transcendent", 0.7),
    "Raven":   ("heroic", 1.0),
    "Cock":    ("heroic", 0.8),
    "Peacock": ("transcendent", 1.0)
}

J2000 = 2451545.0

class AstronomicalConstants:
    epoch = 2444238.5  # 1980 January 0.0
    ecliptic_longitude_epoch = 278.833540
    ecliptic_longitude_perigee = 282.596403
    eccentricity = 0.016718
    moon_mean_longitude_epoch = 64.975464
    moon_mean_perigee_epoch = 349.383063
    moon_eccentricity = 0.054900

c = AstronomicalConstants()

def fixangle(a):
    return a - 360.0 * math.floor(a / 360.0)

def torad(d):
    return d * math.pi / 180.0

def todeg(r):
    return r * 180.0 / math.pi

def kepler(m, ecc):
    epsilon = 0.000001
    m = torad(m)
    e = m
    while True:
        delta = e - ecc * math.sin(e) - m
        e = e - delta / (1.0 - ecc * math.cos(e))
        if abs(delta) <= epsilon:
            break
    return todeg(e)

def julian_date(dt_utc: datetime.datetime) -> float:
    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day + (
        dt_utc.hour +
        dt_utc.minute / 60 +
        dt_utc.second / 3600
    ) / 24
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25 * (y + 4716)) \
         + int(30.6001 * (m + 1)) \
         + d + B - 1524.5
    return jd

def lahiri_ayanamsa(jd: float) -> float:
    T = (jd - J2000) / 36525
    ayan_arcsec = (
        5029.0966 * T +
        1.11113 * T**2 -
        0.000006 * T**3
    )
    return 23.8531 + ayan_arcsec / 3600

def sun_tropical_longitude(jd: float) -> float:
    day = jd - c.epoch
    N = fixangle((360 / 365.2422) * day)
    M = fixangle(N + c.ecliptic_longitude_epoch - c.ecliptic_longitude_perigee)
    Ec = kepler(M, c.eccentricity)
    tan_half = math.sqrt((1 + c.eccentricity) / (1 - c.eccentricity)) * math.tan(torad(Ec / 2))
    Ec = 2 * todeg(math.atan(tan_half))
    lambda_sun = fixangle(Ec + c.ecliptic_longitude_perigee)
    return lambda_sun

def sun_sidereal_longitude(jd: float) -> float:
    trop = sun_tropical_longitude(jd)
    ayan = lahiri_ayanamsa(jd)
    return (trop - ayan) % 360

def moon_tropical_longitude(jd: float) -> float:
    day = jd - c.epoch
    sun_lon = sun_tropical_longitude(jd)
    moon_longitude = fixangle(13.1763966 * day + c.moon_mean_longitude_epoch)
    MM = fixangle(moon_longitude - 0.1114041 * day - c.moon_mean_perigee_epoch)
    evection = 1.2739 * math.sin(torad(2 * (moon_longitude - sun_lon) - MM))
    M_sun = fixangle((360 / 365.2422) * day + c.ecliptic_longitude_epoch - c.ecliptic_longitude_perigee)
    annual_eq = 0.1858 * math.sin(torad(M_sun))
    A3 = 0.37 * math.sin(torad(M_sun))
    MmP = MM + evection - annual_eq - A3
    mEc = 6.2886 * math.sin(torad(MmP))
    A4 = 0.214 * math.sin(torad(2 * MmP))
    lP = moon_longitude + evection + mEc - annual_eq + A4
    variation = 0.6583 * math.sin(torad(2 * (lP - sun_lon)))
    lPP = lP + variation
    return lPP

def moon_sidereal_longitude(jd: float) -> float:
    trop = moon_tropical_longitude(jd)
    ayan = lahiri_ayanamsa(jd)
    return (trop - ayan) % 360

def ascendant_sidereal(jd: float, lat: float, lon: float) -> float:
    d = jd - J2000
    eps = 23.439291 - 0.000013 * d
    eps_rad = math.radians(eps)
    gmst = (
        280.46061837
        + 360.98564736629 * d
    ) % 360
    lst = (gmst + lon) % 360
    lst_rad = math.radians(lst)
    lat_rad = math.radians(lat)
    numerator = math.sin(lst_rad)
    denominator = (
        math.cos(lst_rad) * math.cos(eps_rad)
        - math.tan(lat_rad) * math.sin(eps_rad)
    )
    asc_tropical = math.degrees(math.atan2(numerator, denominator)) % 360
    ayan = lahiri_ayanamsa(jd)
    asc_sidereal = (asc_tropical - ayan) % 360
    return asc_sidereal

def rashi_from_longitude(lon):
    return RASHIS[int(lon // 30) % 12]

def nakshatra_from_longitude(lon):
    NAK_SIZE = 360 / 27
    PADA_SIZE = NAK_SIZE / 4
    nak_index = int(lon / NAK_SIZE) % 27
    pada = int((lon % NAK_SIZE) / PADA_SIZE) + 1
    return NAKSHATRAS[nak_index], pada

def paksha_from_longitudes(moon_lon, sun_lon):
    return "Shukla" if ((moon_lon - sun_lon) % 360) < 180 else "Krishna"

def panchapakshi_bird(nakshatra: str, paksha: str) -> str:
    table = SHUKLA_BIRDS if paksha == "Shukla" else KRISHNA_BIRDS
    for bird, nak_list in table.items():
        if nakshatra in nak_list:
            return bird
    return "Raven"

def vedic_profile(name, utc_dt, lat, lon):
    jd = julian_date(utc_dt)
    sun_sid = sun_sidereal_longitude(jd)
    moon_sid = moon_sidereal_longitude(jd)
    asc_sid = ascendant_sidereal(jd, lat, lon)
    nakshatra, pada = nakshatra_from_longitude(moon_sid)
    paksha = paksha_from_longitudes(moon_sid, sun_sid)
    bird = panchapakshi_bird(nakshatra, paksha)
    return {
        "name": name,
        "sun_sidereal": sun_sid,
        "moon_sidereal": moon_sid,
        "ascendant_sidereal": asc_sid,
        "sun_rashi": rashi_from_longitude(sun_sid),
        "moon_rashi": rashi_from_longitude(moon_sid),
        "ascendant_rashi": rashi_from_longitude(asc_sid),
        "nakshatra": nakshatra,
        "pada": pada,
        "paksha": paksha,
        "bird": bird
    }

# ────────────────────────────────────────────────
#  Branch Logic
# ────────────────────────────────────────────────

def branch_from_panchapakshi(bird):
    return PANCHAPAKSHI_BRANCH_MAP.get(bird, ("heroic", 0.5))

def dasha_branch_from_nakshatra(nakshatra):
    planet = NAKSHATRA_RULERS.get(nakshatra)
    return PLANET_BRANCH_WEIGHT.get(
        planet,
        ("heroic", 0.5)
    )

def branch_from_lagna(ascendant_rashi):
    element = LAGNA_ELEMENT.get(ascendant_rashi)
    branch = ELEMENT_BRANCH_MAP.get(element, "heroic")
    return branch, 0.4

def branch_from_lagna_lord(ascendant_rashi):
    lord = LAGNA_LORD.get(ascendant_rashi)
    branch = PLANET_BRANCH.get(lord, "heroic")
    return branch, 0.6, lord

def resolve_branch_tone(A, B):
    scores = {"tragic": 0.0, "heroic": 0.0, "transcendent": 0.0}
    sources = {"A": {}, "B": {}}
    for label, P in [("A", A), ("B", B)]:
        p_branch, p_w = branch_from_panchapakshi(P["bird"])
        scores[p_branch] += p_w
        sources[label]["panchapakshi"] = (p_branch, p_w)
        d_branch, d_w = dasha_branch_from_nakshatra(P["nakshatra"])
        scores[d_branch] += d_w
        sources[label]["dasha"] = (d_branch, d_w)
        l_branch, l_w = branch_from_lagna(P["ascendant_rashi"])
        scores[l_branch] += l_w
        sources[label]["lagna"] = (l_branch, l_w)
        ll_branch, ll_w, lord = branch_from_lagna_lord(P["ascendant_rashi"])
        scores[ll_branch] += ll_w
        sources[label]["lagna_lord"] = (ll_branch, ll_w, lord)
    total = sum(scores.values()) or 1.0
    norm_scores = {k: v / total for k, v in scores.items()}
    dominant = max(norm_scores, key=norm_scores.get)
    return dominant, norm_scores, sources

# ────────────────────────────────────────────────
#  Symbolic / Saga helpers
# ────────────────────────────────────────────────

BRANCH_SYMBOLS = {
    "tragic": [
        ("Broken Mirror", "A reflection fractured by karma and unspoken truth"),
        ("Falling Ash", "Loss that purifies through suffering"),
        ("Closed Gate", "A path denied until a lesson is learned"),
        ("Dimming Star", "Hope weakened but not extinguished")
    ],
    "heroic": [
        ("Crossing Blade", "A choice that demands courage and action"),
        ("Rising Flame", "Willpower awakened under pressure"),
        ("Open Road", "A journey taken despite uncertainty"),
        ("Beacon Fire", "Leadership that guides others forward")
    ],
    "transcendent": [
        ("Lotus Bloom", "Awakening beyond attachment"),
        ("Veil Lifted", "Perception freed from illusion"),
        ("Converging Rivers", "Unity after long separation"),
        ("Still Light", "Presence that exists beyond struggle")
    ]
}

TIMELINE_SYMBOLS = {
    "Travelers at a crossroads": (
        "Crossroads",
        "Multiple destinies coexist before a choice collapses them into one"
    ),
    "Rivals in a sky city": (
        "High Tower",
        "Conflict elevated above the ordinary world"
    ),
    "Guardian and seeker in exile": (
        "Broken Seal",
        "Protection persists even when belonging is lost"
    ),
    "Messengers before collapse": (
        "Last Signal",
        "Truth delivered at the edge of collapse"
    )
}

DUO_SYMBOLS = {
    "The Seer & The Shield": (
        "Watching Wall",
        "Insight protected by resolve"
    ),
    "The Flame & The Anchor": (
        "Bound Fire",
        "Passion held steady by duty"
    ),
    "The Wanderer & The Lighthouse": (
        "Distant Shore",
        "Guidance that remains even when paths diverge"
    )
}

def chapter_symbols(branch_tone, chapter_index, timelines, duo):
    symbols = []
    branch_pool = BRANCH_SYMBOLS.get(branch_tone, [])
    if branch_pool:
        sym, meaning = branch_pool[(chapter_index - 1) % len(branch_pool)]
        symbols.append((sym, meaning))
    if timelines:
        t = timelines[(chapter_index - 1) % len(timelines)]
        if t in TIMELINE_SYMBOLS:
            symbols.append(TIMELINE_SYMBOLS[t])
    if duo in DUO_SYMBOLS:
        symbols.append(DUO_SYMBOLS[duo])
    return symbols

def mythic_duo(seed):
    r = random.Random(seed + 3)
    return r.choice([
        "The Seer & The Shield",
        "The Flame & The Anchor",
        "The Wanderer & The Lighthouse"
    ])

def parallel_timelines(seed):
    r = random.Random(seed + 1)
    return r.sample([
        "Travelers at a crossroads",
        "Rivals in a sky city",
        "Guardian and seeker in exile",
        "Messengers before collapse"
    ], 2)

# ────────────────────────────────────────────────
#  Deterministic text helpers
# ────────────────────────────────────────────────

def normalize_seed(seed):
    return int(seed) & ((1 << 63) - 1)

def make_seed(a_profile, b_profile):
    def s(x):
        return str(x)
    seed_str = "|".join([
        s(round(a_profile["sun_sidereal"], 6)),
        s(round(a_profile["moon_sidereal"], 6)),
        s(a_profile["nakshatra"]),
        s(a_profile["paksha"]),
        s(a_profile["bird"]),
        s(round(b_profile["sun_sidereal"], 6)),
        s(round(b_profile["moon_sidereal"], 6)),
        s(b_profile["nakshatra"]),
        s(b_profile["paksha"]),
        s(b_profile["bird"]),
    ])
    digest = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

def cosmic_origin(a, b):
    return f"Born under different skies—one in {a['place']} by {a['phase']} light, the other in {b['place']}—your paths bent toward each other."

# ────────────────────────────────────────────────
#  AI Helpers
# ────────────────────────────────────────────────

@st.cache_data(show_spinner="Calling language model...", ttl=3600*6)
def ai_generate(prompt, temperature=0.35, max_tokens=280):
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"user", "content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(API error: {str(e)})"

def build_poetic_prompt(engine_name, base_text, context, theme):
    tone = THEMES.get(theme, "")
    return f"""
You are a storyteller writing in a {theme.lower()} tone.
Style: {tone}
Mystery:
- Born in {context['a_place']} under {context['a_phase']}
Enigma:
- Born in {context['b_place']} under {context['b_phase']}
Rarity: {context['rarity']}
Archetype: {context['duo']}
Timelines: {", ".join(context['timelines'])}
Rules:
- No predictions
- No advice
- Symbolic narrative
Expand the following {engine_name}:
{base_text}
"""

def generate_fantasy_story(context, theme):
    prompt = f"""
Write a short {theme.lower()} fantasy story (2–3 paragraphs).
Characters:
- Mystery from {context['a_place']}
- Enigma from {context['b_place']}
Bond: {context['duo']}
Rarity: {context['rarity']}
Themes: {", ".join(context['timelines'])}
Write as legend. No advice. No future prediction.
"""
    return ai_generate(prompt, temperature=0.4, max_tokens=450)

# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────

st.title("🌌 Twin Destiny Engine")
st.caption("Vedic-inspired symbolic destiny weave for two birth charts")

with st.sidebar:
    st.subheader("🧑 Mystery")
    a_name = st.text_input("Name", "Mahaan")
    a_date = st.date_input("Date", datetime.date(1993, 7, 12), min_value=datetime.date(1900, 1, 1), max_value=datetime.date(2030, 12, 31))
    a_time = st.time_input("Time", datetime.time(12, 26), step=datetime.timedelta(minutes=1))
    a_lat = st.number_input("Latitude", value=13.32)
    a_lon = st.number_input("Longitude", value=75.77)
    a_tz = st.text_input("Timezone", "Asia/Kolkata")
    st.subheader("👩 Enigma")
    b_name = st.text_input("Name ", "Unknown")
    b_date = st.date_input("Date ", datetime.date(1993, 7, 12), min_value=datetime.date(1900, 1, 1), max_value=datetime.date(2030, 12, 31))
    b_time = st.time_input("Time ", datetime.time(12, 26), step=datetime.timedelta(minutes=1))
    b_lat = st.number_input("Latitude ", value=13.32)
    b_lon = st.number_input("Longitude ", value=75.77)
    b_tz = st.text_input("Timezone ", "Asia/Kolkata")
theme = st.selectbox("Theme", ["Mythic", "Sci-Fi", "Romantic", "Dark"])

if st.button("✨ Generate Destiny"):
    try:
        a_tzinfo = zoneinfo.ZoneInfo(a_tz)
        b_tzinfo = zoneinfo.ZoneInfo(b_tz)
    except zoneinfo.ZoneInfoNotFoundError:
        st.error("Invalid timezone. Use format like 'Asia/Kolkata'")
        st.stop()
    a_utc = datetime.datetime.combine(a_date, a_time)\
        .replace(tzinfo=a_tzinfo)\
        .astimezone(zoneinfo.ZoneInfo("UTC"))
    b_utc = datetime.datetime.combine(b_date, b_time)\
        .replace(tzinfo=b_tzinfo)\
        .astimezone(zoneinfo.ZoneInfo("UTC"))
    A = vedic_profile(a_name, a_utc, a_lat, a_lon)
    B = vedic_profile(b_name, b_utc, b_lat, b_lon)
    seed = normalize_seed(make_seed(A, B))
    dominant, branch_scores, branch_sources = resolve_branch_tone(A, B)
    context = {
        "a_place": A["nakshatra"],
        "b_place": B["nakshatra"],
        "a_phase": A["paksha"],
        "b_phase": B["paksha"],
        "duo": mythic_duo(seed),
        "rarity": "Rare",  # simplified
        "timelines": parallel_timelines(seed),
    }
    origin_prompt = build_poetic_prompt(
        "Cosmic Origin",
        cosmic_origin(
            {"place": A["nakshatra"], "phase": A["paksha"]},
            {"place": B["nakshatra"], "phase": B["paksha"]}
        ),
        context,
        theme
    )
    origin = ai_generate(origin_prompt, 0.25, 220)
    st.subheader("🌠 Cosmic Origin")
    st.write(origin)
    st.subheader(f"Dominant Branch → **{dominant.title()}**")
    col1, col2 = st.columns([3,2])
    with col1:
        st.write("Short saga fragment:")
        saga_text = generate_fantasy_story(context, theme)
        st.markdown(f"> {saga_text}")
    with col2:
        st.write("**How the branch was chosen**")
        for lbl, key, prof in [("Mystery","A",A), ("Enigma","B",B)]:
            s = branch_sources[key]
            st.caption(f"**{lbl}**")
            st.write(f"• Bird → {s['panchapakshi'][0]} ({s['panchapakshi'][1]:.1f})")
            st.write(f"• Nakshatra → {s['dasha'][0]} ({s['dasha'][1]:.1f})")
            st.write(f"• Lagna → {s['lagna'][0]} ({s['lagna'][1]:.1f})")
            ll = s["lagna_lord"]
            st.write(f"• Lord ({ll[2]}) → {ll[0]} ({ll[1]:.1f})")
        st.write("**Final normalized weights**")
        for k,v in branch_scores.items():
            st.progress(v)
            st.caption(f"{k.title()}: {v:.2f}")
    st.caption(f"Seed (for reproducibility): **{seed}**")
st.markdown("---")
st.caption("Public demo version — no login, no saving, no PDF export. Pure calculation + AI poetic layer.")
