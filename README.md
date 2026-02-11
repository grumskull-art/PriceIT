# PriceIT / Shopping-Politiet

High-precision price intelligence API + iOS Share Extension extraction script.

## Files
- `main.py` – FastAPI backend with SerpApi shopping analysis.
- `extension_preprocessing.js` – `NSExtensionJavaScriptPreprocessingJS` logic with JSON-LD-first extraction.
- `.env.example` – env var structure for API key + SerpApi key.

## Backend quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

## API
`POST /v1/analyze`

Required header:
- `X-API-Key: <API_KEY>`

Sample payload:
```json
{
  "product_name": "Nike Air Zoom Pegasus 40",
  "current_price": "999,00 DKK",
  "currency": "DKK",
  "brand": "Nike",
  "gtin13": "0196608554022",
  "country": "dk"
}
```

## Notes
- GTIN is prioritized over all other identifiers.
- Fuzzy matching includes strict filtering to reduce accessories/wrong models.
- Used/refurbished products are excluded by default.
- Current caching is in-memory; for production, replace cache helpers with Redis.
