# app.py
# Twin Destiny Engine – public version for Streamlit Cloud
# 2025-02-16

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

PLANET_BRANCH_LORD = {
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

def julian_date(dt_utc):
    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600)/24
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = int(365.25*(y+4716)) + int(30.6001*(m+1)) + d + B - 1524.5
    return jd

def lahiri_ayanamsa(jd):
    T = (jd - J2000) / 36525
    ayan_arcsec = 5029.0966*T + 1.11113*T**2 - 0.000006*T**3
    return 23.8531 + ayan_arcsec/3600

def sun_sidereal_longitude(jd):
    d = jd - J2000
    L = (280.460 + 0.9856474*d) % 360
    g = math.radians((357.528 + 0.9856003*d) % 360)
    sun_trop = (L + 1.915*math.sin(g) + 0.020*math.sin(2*g)) % 360
    return (sun_trop - lahiri_ayanamsa(jd)) % 360

def moon_sidereal_longitude(jd):
    d = jd - J2000
    L0 = 218.3164477 + 13.17639648*d
    M  = 134.9633964 + 13.06499295*d
    F  = 93.2720950  + 13.22935024*d
    moon_trop = (L0 +
                 6.289*math.sin(math.radians(M)) +
                 1.274*math.sin(math.radians(2*(L0-M))) +
                 0.658*math.sin(math.radians(2*L0)) +
                 0.214*math.sin(math.radians(2*M)) -
                 0.186*math.sin(math.radians(M)) -
                 0.059*math.sin(math.radians(2*(L0-F)))
                ) % 360
    return (moon_trop - lahiri_ayanamsa(jd)) % 360

def ascendant_sidereal(jd, lat, lon):
    d = jd - J2000
    eps = 23.439291 - 0.000013*d
    eps_rad = math.radians(eps)
    gmst = (280.46061837 + 360.98564736629*d) % 360
    lst = (gmst + lon) % 360
    lst_rad = math.radians(lst)
    lat_rad = math.radians(lat)
    num = math.sin(lst_rad)
    den = math.cos(lst_rad)*math.cos(eps_rad) - math.tan(lat_rad)*math.sin(eps_rad)
    asc_trop = math.degrees(math.atan2(num, den)) % 360
    return (asc_trop - lahiri_ayanamsa(jd)) % 360

def rashi_from_longitude(lon):
    return RASHIS[int(lon // 30) % 12]

def nakshatra_from_longitude(lon):
    NAK_SIZE = 360 / 27
    nak_index = int(lon / NAK_SIZE) % 27
    return NAKSHATRAS[nak_index]

def paksha_from_longitudes(moon_lon, sun_lon):
    return "Shukla" if ((moon_lon - sun_lon) % 360) < 180 else "Krishna"

def panchapakshi_bird(nakshatra, paksha):
    table = SHUKLA_BIRDS if paksha == "Shukla" else KRISHNA_BIRDS
    for bird, naks in table.items():
        if nakshatra in naks:
            return bird
    return "Raven"

def vedic_profile(name, utc_dt, lat, lon):
    jd = julian_date(utc_dt)
    sun   = sun_sidereal_longitude(jd)
    moon  = moon_sidereal_longitude(jd)
    asc   = ascendant_sidereal(jd, lat, lon)
    nak   = nakshatra_from_longitude(moon)
    pak   = paksha_from_longitudes(moon, sun)
    bird  = panchapakshi_bird(nak, pak)
    return {
        "name": name,
        "sun": sun, "moon": moon, "ascendant": asc,
        "nakshatra": nak, "paksha": pak, "bird": bird,
        "ascendant_rashi": rashi_from_longitude(asc)
    }

# ────────────────────────────────────────────────
#  Branch Logic
# ────────────────────────────────────────────────

def branch_from_panchapakshi(bird):
    return PANCHAPAKSHI_BRANCH_MAP.get(bird, ("heroic", 0.5))

def dasha_branch_from_nakshatra(nakshatra):
    planet = NAKSHATRA_RULERS.get(nakshatra, "Sun")
    return PLANET_BRANCH_WEIGHT.get(planet, ("heroic", 0.5))

def branch_from_lagna(rashi):
    el = LAGNA_ELEMENT.get(rashi, "Fire")
    br = ELEMENT_BRANCH_MAP.get(el, "heroic")
    return br, 0.4

def branch_from_lagna_lord(rashi):
    lord = LAGNA_LORD.get(rashi, "Sun")
    br   = PLANET_BRANCH_LORD.get(lord, "heroic")
    return br, 0.6, lord

def resolve_branch_tone(A, B):
    scores = {"tragic":0.0, "heroic":0.0, "transcendent":0.0}
    sources = {"A":{},"B":{}}

    for label, P in [("A",A), ("B",B)]:
        # 1. Panchapakshi
        pb, pw = branch_from_panchapakshi(P["bird"])
        scores[pb] += pw
        sources[label]["panchapakshi"] = (pb, pw)

        # 2. Moon nakshatra lord (dasha style)
        db, dw = dasha_branch_from_nakshatra(P["nakshatra"])
        scores[db] += dw
        sources[label]["dasha"] = (db, dw)

        # 3. Lagna element
        lb, lw = branch_from_lagna(P["ascendant_rashi"])
        scores[lb] += lw
        sources[label]["lagna"] = (lb, lw)

        # 4. Lagna lord
        llb, llw, lord = branch_from_lagna_lord(P["ascendant_rashi"])
        scores[llb] += llw
        sources[label]["lagna_lord"] = (llb, llw, lord)

    total = sum(scores.values()) or 1.0
    norm  = {k:v/total for k,v in scores.items()}
    dom   = max(norm, key=norm.get)
    return dom, norm, sources

# ────────────────────────────────────────────────
#  Symbolic / Saga helpers
# ────────────────────────────────────────────────

BRANCH_SYMBOLS = {
    "tragic": [
        ("Broken Mirror", "A reflection fractured by karma and unspoken truth"),
        ("Falling Ash",   "Loss that purifies through suffering"),
    ],
    "heroic": [
        ("Crossing Blade", "A choice that demands courage and action"),
        ("Rising Flame",   "Willpower awakened under pressure"),
    ],
    "transcendent": [
        ("Lotus Bloom",    "Awakening beyond attachment"),
        ("Veil Lifted",    "Perception freed from illusion"),
    ]
}

TIMELINE_SYMBOLS = {
    "Travelers at a crossroads":     ("Crossroads",    "Multiple destinies coexist before choice"),
    "Rivals in a sky city":          ("High Tower",    "Conflict elevated above the ordinary"),
}

DUO_SYMBOLS = {
    "The Seer & The Shield":    ("Watching Wall",  "Insight protected by resolve"),
    "The Flame & The Anchor":   ("Bound Fire",     "Passion held steady by duty"),
}

def chapter_symbols(branch_tone, chapter_idx, timelines, duo):
    syms = []
    pool = BRANCH_SYMBOLS.get(branch_tone, [])
    if pool:
        syms.append(pool[(chapter_idx-1) % len(pool)])
    if timelines and chapter_idx-1 < len(timelines):
        t = timelines[chapter_idx-1]
        if t in TIMELINE_SYMBOLS:
            syms.append(TIMELINE_SYMBOLS[t])
    if duo in DUO_SYMBOLS:
        syms.append(DUO_SYMBOLS[duo])
    return syms

def mythic_duo(seed):
    r = random.Random(seed + 100)
    return r.choice(list(DUO_SYMBOLS.keys()))

def parallel_timelines(seed):
    r = random.Random(seed + 200)
    pool = list(TIMELINE_SYMBOLS.keys())
    return r.sample(pool, min(2, len(pool)))

# ────────────────────────────────────────────────
#  Deterministic text helpers
# ────────────────────────────────────────────────

def normalize_seed(n):
    return int(n) & ((1 << 63) - 1)

def make_seed(a, b):
    parts = [
        str(round(a.get("sun", 0.0), 6)),
        str(round(a.get("moon", 0.0), 6)),
        a.get("nakshatra", ""),
        a.get("paksha", ""),
        a.get("bird", ""),
        str(round(b.get("sun", 0.0), 6)),
        str(round(b.get("moon", 0.0), 6)),
        b.get("nakshatra", ""),
        b.get("paksha", ""),
        b.get("bird", ""),
    ]
    s = "|".join(parts)
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)

def cosmic_origin(a, b):
    return f"Born under different skies — one in **{a}** by {b} light, the other in distant rhythm — yet threads still pulled taut."

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

def poetic_expand(base_text, context, theme, seed):
    tone = THEMES.get(theme, "")
    prompt = f"""You are a poetic mythic chronicler.
Style: {tone}, symbolic, no predictions, no advice.

Mystery born under: {context['a_nak']}
Enigma born under:  {context['b_nak']}
Duo archetype:      {context['duo']}
Rarity class:       {context['rarity']}
Timelines:          {', '.join(context['timelines'])}

Expand / poetically rephrase the following seed sentence into 3–5 beautiful lines:

{base_text}
"""
    return ai_generate(prompt, temperature=0.3, max_tokens=220)

def generate_short_saga(branch_tone, context, theme, seed):
    prompt = f"""Write a very short {theme.lower()} mythic fragment (80–140 words).
Focus: one key moment / feeling
Characters: Mystery & Enigma — {context['duo']}
Atmosphere: {branch_tone} tone
Timelines hint: {', '.join(context['timelines'][:2])}
Symbolic only. No predictions. No moral. Legend style.
"""
    return ai_generate(prompt, temperature=0.45, max_tokens=320)

# ────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────

st.title("🌌 Twin Destiny Engine")
st.caption("Vedic-inspired symbolic destiny weave for two birth charts")

with st.sidebar:

    st.subheader("Mystery (A)")
    a_name  = st.text_input("Name", "Mahaan", key="a_name")
    a_date  = st.date_input("Date",  datetime.date(1993,7,12), key="a_date")
    a_time  = st.time_input("Time",  datetime.time(12,26), key="a_time")
    a_lat   = st.number_input("Lat",   value=13.32, key="a_lat")
    a_lon   = st.number_input("Lon",   value=75.77, key="a_lon")

    st.subheader("Enigma (B)")
    b_name  = st.text_input("Name ", "Unknown", key="b_name")
    b_date  = st.date_input("Date ", datetime.date(1993,7,12), key="b_date")
    b_time  = st.time_input("Time ", datetime.time(12,26), key="b_time")
    b_lat   = st.number_input("Lat ",  value=13.32, key="b_lat")
    b_lon   = st.number_input("Lon ",  value=75.77, key="b_lon")

    theme_choice = st.selectbox("Narrative Tone", list(THEMES.keys()), index=0)

if st.button("✦ Weave Destiny", type="primary"):

    # Convert to UTC
    a_dt = datetime.datetime.combine(a_date, a_time).replace(tzinfo=zoneinfo.ZoneInfo("Asia/Kolkata"))
    b_dt = datetime.datetime.combine(b_date, b_time).replace(tzinfo=zoneinfo.ZoneInfo("Asia/Kolkata"))
    a_utc = a_dt.astimezone(zoneinfo.ZoneInfo("UTC"))
    b_utc = b_dt.astimezone(zoneinfo.ZoneInfo("UTC"))

    A = vedic_profile(a_name, a_utc, a_lat, a_lon)
    B = vedic_profile(b_name, b_utc, b_lat, b_lon)

    seed_raw = make_seed(A, B)
    seed = normalize_seed(seed_raw)

    dominant, scores, sources = resolve_branch_tone(A, B)

    timelines = parallel_timelines(seed)
    duo       = mythic_duo(seed)
    rarity    = "Rare"   # simplified

    ctx = {
        "a_nak": A["nakshatra"],
        "b_nak": B["nakshatra"],
        "duo":   duo,
        "rarity": rarity,
        "timelines": timelines,
    }

    origin_base = cosmic_origin(A["nakshatra"], A["paksha"])
    origin_poetic = poetic_expand(origin_base, ctx, theme_choice, seed)

    st.subheader("🌠 Cosmic Origin")
    st.markdown(origin_poetic)

    st.subheader(f"Dominant Branch → **{dominant.title()}**")

    col1, col2 = st.columns([3,2])

    with col1:
        st.write("Short saga fragment:")
        saga_text = generate_short_saga(dominant, ctx, theme_choice, seed)
        st.markdown(f"> {saga_text}")

    with col2:
        st.write("**How the branch was chosen**")
        for lbl, key, prof in [("Mystery","A",A), ("Enigma","B",B)]:
            s = sources[key]
            st.caption(f"**{lbl}**")
            st.write(f"• Bird → {s['panchapakshi'][0]} ({s['panchapakshi'][1]:.1f})")
            st.write(f"• Nakshatra → {s['dasha'][0]} ({s['dasha'][1]:.1f})")
            st.write(f"• Lagna → {s['lagna'][0]} ({s['lagna'][1]:.1f})")
            ll = s["lagna_lord"]
            st.write(f"• Lord ({ll[2]}) → {ll[0]} ({ll[1]:.1f})")

        st.write("**Final normalized weights**")
        for k,v in scores.items():
            st.progress(v)
            st.caption(f"{k.title()}: {v:.2f}")

    st.caption(f"Seed (for reproducibility): **{seed}**")

st.markdown("---")
st.caption("Public demo version — no login, no saving, no PDF export. Pure calculation + AI poetic layer.")
