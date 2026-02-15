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

## iPhone Web App (tonight mode)
- Open browser URL: `/app` (same host as API).
- Root `/` redirects to `/app`.
- Web app uses internal endpoint `POST /v1/analyze-web` (no API key input in UI).
- Web app supports `POST /v1/search-web` via button `Søg Produkt` (viser flere tilbud).
- Web app supports `POST /v1/extract-web`:
  - paste product URL
  - tap `Hent data fra link`
  - fields auto-fill before analysis
- Market is Denmark-only in backend (DK + `amazon.de`, `eBay` blocked).

### Phone access when backend runs in WSL2
Run these in Windows PowerShell as Administrator:
```powershell
# find WSL IP
wsl hostname -I

# forward Windows port 8000 to WSL port 8000
C:\Windows\System32\netsh.exe interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=<WSL_IP> connectport=8000

# allow firewall inbound
C:\Windows\System32\netsh.exe advfirewall firewall add rule name="PriceIT 8000" dir=in action=allow protocol=TCP localport=8000
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

## iOS Share Extension Wiring
- Full setup guide: `IOS_SHARE_EXTENSION_SETUP.md`
- Includes:
  - required `Info.plist` keys
  - `ShareViewController.swift` example that calls `/v1/analyze`
  - local network/ATS notes for same-night testing
