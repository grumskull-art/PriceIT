import json
import logging
import os
import re
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

try:
    from serpapi import GoogleSearch
except ImportError:  # fallback for older package style
    from google_search_results import GoogleSearch  # type: ignore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shopping-politiet")

API_KEY = os.getenv("API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
DEFAULT_COUNTRY = "dk"
DEFAULT_CURRENCY = "DKK"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))

app = FastAPI(title="Shopping-Politiet", version="1.0.0")

# Simple in-memory cache. In production, replace with Redis for shared, multi-instance cache.
CACHE: Dict[str, Dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    product_name: Optional[str] = Field(default=None, min_length=1)
    current_price: Optional[float | str] = None
    currency: Optional[str] = None
    brand: Optional[str] = None
    gtin13: Optional[str] = None
    gtin8: Optional[str] = None
    sku: Optional[str] = None
    country: str = Field(default=DEFAULT_COUNTRY, min_length=2, max_length=2)
    include_used: bool = False
    include_refurbished: bool = False

    @field_validator("current_price", mode="before")
    @classmethod
    def normalize_price(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9,.-]", "", value).strip()
            if cleaned.count(",") == 1 and cleaned.count(".") > 1:
                cleaned = cleaned.replace(".", "").replace(",", ".")
            elif cleaned.count(",") == 1 and cleaned.count(".") == 0:
                cleaned = cleaned.replace(",", ".")
            elif cleaned.count(",") > 1 and cleaned.count(".") == 0:
                cleaned = cleaned.replace(",", "")
            try:
                return float(cleaned)
            except ValueError as exc:
                raise ValueError(f"Invalid price format: {value}") from exc
        raise ValueError("current_price must be string or number")

    @field_validator("gtin13", "gtin8", mode="before")
    @classmethod
    def normalize_gtin(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        digits = re.sub(r"\D", "", str(value))
        return digits if digits else None

    @field_validator("sku", "product_name", "brand", "currency", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip()
        return s or None


class Alternative(BaseModel):
    shop: str
    price: float
    currency: str
    url: str


class AnalyzeResponse(BaseModel):
    verdict: str
    query_type: str
    market_min_price: Optional[float]
    market_avg_price: Optional[float]
    current_page_price: Optional[float]
    currency: str
    savings_vs_min: Optional[float]
    top_3_alternatives: List[Alternative]
    excluded_results: int


def cache_key(payload: AnalyzeRequest) -> str:
    identity = payload.gtin13 or payload.gtin8 or payload.sku or payload.product_name or "unknown"
    return f"{payload.country}:{identity}:{payload.include_used}:{payload.include_refurbished}"


def get_cached(key: str) -> Optional[List[Dict[str, Any]]]:
    record = CACHE.get(key)
    if not record:
        return None
    if time.time() - record["ts"] > CACHE_TTL_SECONDS:
        CACHE.pop(key, None)
        return None
    return record["results"]


def set_cached(key: str, results: List[Dict[str, Any]]) -> None:
    CACHE[key] = {"ts": time.time(), "results": results}


def build_search_query(payload: AnalyzeRequest) -> tuple[str, str]:
    gtin = payload.gtin13 or payload.gtin8
    if gtin:
        return gtin, "gtin_exact"
    if payload.sku and payload.brand:
        return f'"{payload.brand}" "{payload.sku}"', "sku_brand"
    if payload.sku:
        return f'"{payload.sku}"', "sku"
    if payload.product_name:
        if payload.brand:
            return f'"{payload.brand}" {payload.product_name}', "name_brand_fuzzy"
        return payload.product_name, "name_fuzzy"
    raise HTTPException(status_code=400, detail="One of gtin, sku, or product_name is required")


def call_serpapi(query: str, country: str, currency: str) -> List[Dict[str, Any]]:
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY is not configured")

    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "gl": country.lower(),
        "hl": "en",
        "num": 30,
    }
    if currency:
        params["currency"] = currency

    try:
        search = GoogleSearch(params)
        data = search.get_dict()
    except Exception as exc:  # network/timeouts and client errors
        logger.exception("SerpApi call failed")
        raise HTTPException(status_code=502, detail=f"Search provider failure: {exc}") from exc

    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Malformed search response")

    return data.get("shopping_results", []) or []


def parse_price(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        cleaned = re.sub(r"[^0-9,.-]", "", raw)
        if cleaned.count(",") == 1 and cleaned.count(".") == 0:
            cleaned = cleaned.replace(",", ".")
        elif cleaned.count(",") > 1 and cleaned.count(".") == 0:
            cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def is_wrong_model(payload: AnalyzeRequest, title: str) -> bool:
    if not payload.product_name:
        return False
    name_tokens = {t for t in re.findall(r"\w+", payload.product_name.lower()) if len(t) > 2}
    title_tokens = set(re.findall(r"\w+", title.lower()))
    if not name_tokens:
        return False
    overlap = len(name_tokens & title_tokens) / max(len(name_tokens), 1)
    # Strict threshold for fuzzy mode to avoid accessories.
    return overlap < 0.45


def contains_accessory_signal(product_name: Optional[str], title: str) -> bool:
    if not product_name:
        return False
    title_l = title.lower()
    product_l = product_name.lower()
    accessory_terms = {"case", "cover", "belt", "charger", "strap", "cable", "holder", "refill"}
    if any(t in title_l for t in accessory_terms) and not any(t in product_l for t in accessory_terms):
        return True
    return False


def parse_and_filter_results(payload: AnalyzeRequest, raw_results: List[Dict[str, Any]]) -> tuple[List[Alternative], int]:
    filtered: List[Alternative] = []
    excluded = 0

    for r in raw_results:
        title = str(r.get("title", "")).strip()
        source = str(r.get("source", "Unknown shop")).strip() or "Unknown shop"
        link = str(r.get("link", "")).strip()
        condition = str(r.get("condition", "")).lower()
        shipping = str(r.get("shipping", "")).lower()

        price = parse_price(r.get("price") or r.get("extracted_price"))
        if price is None or not link:
            excluded += 1
            continue

        if payload.country.lower() == "dk":
            ship_intl = any(k in shipping for k in ["international", "eu", "worldwide", "global"])
            if "dk" not in source.lower() and ship_intl is False and "denmark" not in shipping:
                excluded += 1
                continue

        if ("used" in condition or "used" in title.lower()) and not payload.include_used:
            excluded += 1
            continue

        if ("refurbished" in condition or "refurbished" in title.lower()) and not payload.include_refurbished:
            excluded += 1
            continue

        if contains_accessory_signal(payload.product_name, title):
            excluded += 1
            continue

        if payload.gtin13 or payload.gtin8:
            gtin = payload.gtin13 or payload.gtin8
            if gtin and gtin not in json.dumps(r).replace(" ", ""):
                # keep result if Google omitted gtin in blob but only when title strongly matches
                if is_wrong_model(payload, title):
                    excluded += 1
                    continue
        elif payload.product_name and is_wrong_model(payload, title):
            excluded += 1
            continue

        filtered.append(
            Alternative(
                shop=source,
                price=price,
                currency=payload.currency or DEFAULT_CURRENCY,
                url=link,
            )
        )

    filtered.sort(key=lambda x: x.price)
    return filtered, excluded


def decide_verdict(current_page_price: Optional[float], market_min: Optional[float], market_avg: Optional[float]) -> str:
    if market_min is None:
        return "NO_MARKET_DATA"
    if current_page_price is None:
        return "NEEDS_CURRENT_PRICE"

    if current_page_price <= market_min:
        return "GREAT_DEAL"

    if current_page_price > market_min * 1.05:
        return "OVERPRICED"

    if market_avg is not None and current_page_price <= market_avg:
        return "OK"

    return "OK"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None)) -> AnalyzeResponse:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    currency = (payload.currency or DEFAULT_CURRENCY).upper()
    query, query_type = build_search_query(payload)
    key = cache_key(payload)

    try:
        raw_results = get_cached(key)
        if raw_results is None:
            raw_results = call_serpapi(query=query, country=payload.country, currency=currency)
            set_cached(key, raw_results)

        alternatives, excluded = parse_and_filter_results(payload, raw_results)

        market_min_price = alternatives[0].price if alternatives else None
        market_avg_price = mean([a.price for a in alternatives]) if alternatives else None

        verdict = decide_verdict(payload.current_price, market_min_price, market_avg_price)

        savings = None
        if payload.current_price is not None and market_min_price is not None:
            savings = round(payload.current_price - market_min_price, 2)

        return AnalyzeResponse(
            verdict=verdict,
            query_type=query_type,
            market_min_price=market_min_price,
            market_avg_price=round(market_avg_price, 2) if market_avg_price is not None else None,
            current_page_price=payload.current_price,
            currency=currency,
            savings_vs_min=savings,
            top_3_alternatives=alternatives[:3],
            excluded_results=excluded,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled analyze failure")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
