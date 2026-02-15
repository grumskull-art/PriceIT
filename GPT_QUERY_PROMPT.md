# GPT Query Prompt (copy/paste)

Use this prompt in GPT to generate extra search queries for PriceIT.

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

## How to use in app

1. Paste output lines into `GPT query-pack` in `/app` (optional override).
2. Press `Søg Produkt`.
3. The backend merges these with its own built-in query expansion.
4. If `OPENAI_API_KEY` is configured, the backend can also auto-generate boost queries when needed.
