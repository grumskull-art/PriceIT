# PriceIT / Shopping-Politiet

High-precision price intelligence API + iOS Share Extension extraction script.

## Files
- `main.py` – FastAPI backend with SerpApi shopping analysis.
- `extension_preprocessing.js` – `NSExtensionJavaScriptPreprocessingJS` logic with JSON-LD-first extraction.
- `.env.example` – env var structure for API key + SerpApi key.
- `GPT_QUERY_PROMPT.md` – prompt template for generating query variants in GPT.

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
- Web app har hurtigt `Butiksfilter` (Zalando/Next/Name It/Børnesider).
- Web app har `GPT query-pack` felt (valgfrit): indsæt én søge-variant pr linje, eller lad tom for auto query-boost.
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
- Fuzzy matching now uses stricter relevance scoring (type/feature/color/age-size/brand) to reduce wrong products.
- Used/refurbished products are excluded by default.
- Current caching is in-memory; for production, replace cache helpers with Redis.
- Auto query-boost via OpenAI er valgfri:
  - `OPENAI_API_KEY=<din nøgle>`
  - `OPENAI_MODEL=gpt-4o-mini` (kan ændres)

## GPT Prompt (Query Pack)
Brug denne prompt i GPT og copy/paste outputtet ind i feltet `GPT query-pack` i appen.

```text
Du er query-generator for dansk e-commerce prissøgning.
Lav 8 korte søgequeries for samme produkt.

Krav:
- Fokus på danske butikker og shopping-søgning.
- Behold brand/model hvis kendt.
- Lav variationer af produkttype (fx hoodie/sweatshirt/sweater).
- Lav variationer af farveord (fx blå/blue/royal/navy), men kun relevante.
- Fjern støj (fx køn/alder ord) i nogle varianter.
- Ingen forklaringer.
- Output kun rå queries, én pr linje.

Input:
Produktnavn: <indsæt>
Brand: <indsæt eller tom>
SKU: <indsæt eller tom>
GTIN: <indsæt eller tom>
```

## iOS Share Extension Wiring
- Full setup guide: `IOS_SHARE_EXTENSION_SETUP.md`
- Includes:
  - required `Info.plist` keys
  - `ShareViewController.swift` example that calls `/v1/analyze`
  - local network/ATS notes for same-night testing
