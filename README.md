# Twin Destiny Engine

🌌 **Vedic-inspired symbolic destiny explorer for two people**  
A poetic, deterministic fate-weaving web app that calculates a shared "destiny branch" (Tragic / Heroic / Transcendent) using Vedic astrology principles and generates short mythic stories with Grok AI.

Live demo: [https://your-streamlit-cloud-url.streamlit.app](https://your-streamlit-cloud-url.streamlit.app)  
*(replace with your actual deployed link after you deploy)*

## What it does

- Computes sidereal positions (Lahiri ayanamsa) for Sun, Moon, Ascendant
- Determines Panchapakshi bird, nakshatra lord influence, lagna element & lord
- Resolves a dominant **destiny tone** (tragic / heroic / transcendent) from the two charts
- Generates a deterministic seed → mythic duo archetype, parallel timelines, rarity class
- Uses Grok AI to poetically expand the "Cosmic Origin" and write short saga fragments

## Features

- Pure client-side + API calculation (no database, no login)
- Fully deterministic – same birth data → same seed & branch every time
- Four narrative tones: Mythic, Sci-Fi, Romantic, Dark
- Symbolic chapter annotations logic included (but currently light in UI)
- Clean, minimal Streamlit interface

## Screenshots

*(Add 1–3 screenshots here after deployment – e.g. input sidebar + result page)*

## Tech Stack

- **Frontend**: Streamlit
- **Backend logic**: Pure Python (Vedic calculations from scratch)
- **AI**: xAI Grok API (`grok-beta`)
- **Astronomy**: Custom sidereal longitude + ascendant formulas (no external ephemeris library)

## Installation (local development)

```bash
# 1. Clone
git clone https://github.com/yourusername/twin-destiny-engine.git
cd twin-destiny-engine

# 2. Create virtual env (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# or
.\venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key (create .env or set in terminal)
export XAI_API_KEY="your_xai_api_key_here"

# 5. Run
streamlit run app.py
