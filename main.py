from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
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
ALLOWED_CROSS_BORDER_SOURCES = {"amazon.de", "next.co.uk"}
BLOCKED_SOURCES = {"ebay"}
PREFERRED_CHILD_STORES = {
    "zalando",
    "next",
    "name it",
    "nameit",
    "jollyroom",
    "boozt",
    "kids-world",
    "kidsworld",
    "luksusbaby",
}
TOKEN_STOPWORDS = {
    "the",
    "and",
    "with",
    "for",
    "til",
    "med",
    "uden",
    "hos",
    "der",
    "som",
    "eller",
    "børn",
    "kids",
    "child",
    "children",
}
NON_DISTINCTIVE_MODEL_TOKENS = {
    "gtx",
    "gore",
    "tex",
    "warm",
    "winter",
    "snow",
    "shoe",
    "shoes",
    "boot",
    "boots",
    "støvle",
    "støvler",
    "vinterstøvler",
    "vandtæt",
    "vandtætte",
    "waterproof",
    "junior",
    "jr",
    "kids",
    "kid",
    "child",
    "children",
    "pull",
    "zip",
    "low",
    "mid",
    "high",
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "gray",
    "grey",
    "silver",
    "gold",
    "beige",
    "brown",
    "navy",
    "night",
    "sky",
}

app = FastAPI(title="Shopping-Politiet", version="1.0.0")
WEBAPP_INDEX = Path(__file__).resolve().parent / "webapp" / "index.html"

# Simple in-memory cache. In production, replace with Redis for shared, multi-instance cache.
CACHE: Dict[str, Dict[str, Any]] = {}


def parse_localized_price(value: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9,.-]", "", value).strip()
    cleaned = re.sub(r"^[,.]+|[,.]+$", "", cleaned)
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        # Use the right-most separator as decimal marker and strip thousands separators.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    elif cleaned.count(",") > 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", "")
    elif cleaned.count(".") > 1 and cleaned.count(",") == 0:
        cleaned = cleaned.replace(".", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


class ProductHtmlExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: Dict[tuple[str, str], str] = {}
        self.ldjson_blocks: List[str] = []
        self._in_ldjson_script = False
        self._script_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        attrs_map = {str(k).lower(): (v or "") for k, v in attrs}
        tag_l = tag.lower()

        if tag_l == "meta":
            content = attrs_map.get("content", "").strip()
            if not content:
                return
            for attr_name in ("property", "name", "itemprop"):
                attr_value = attrs_map.get(attr_name, "").strip().lower()
                if attr_value:
                    self.meta[(attr_name, attr_value)] = content
            return

        if tag_l == "script":
            script_type = attrs_map.get("type", "").lower()
            if "application/ld+json" in script_type:
                self._in_ldjson_script = True
                self._script_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_ldjson_script:
            self._script_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self._in_ldjson_script:
            block = "".join(self._script_chunks).strip()
            if block:
                self.ldjson_blocks.append(unescape(block))
            self._in_ldjson_script = False
            self._script_chunks = []

    def get_meta(self, attr_name: str, attr_value: str) -> Optional[str]:
        return self.meta.get((attr_name.lower(), attr_value.lower()))


def find_product_node(node: Any) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    if isinstance(node, list):
        for item in node:
            found = find_product_node(item)
            if found:
                return found
        return None
    if isinstance(node, dict):
        node_type = node.get("@type")
        if isinstance(node_type, str) and node_type.lower() == "product":
            return node
        if isinstance(node_type, list) and any(str(t).lower() == "product" for t in node_type):
            return node
        if "@graph" in node:
            found = find_product_node(node.get("@graph"))
            if found:
                return found
        for value in node.values():
            found = find_product_node(value)
            if found:
                return found
    return None


def fetch_html(url: str) -> str:
    req = Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
            )
        },
    )
    with urlopen(req, timeout=12) as response:
        raw = response.read(2_500_000)
        charset = response.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def extract_product_data_from_html(source_url: str, html: str) -> ExtractUrlResponse:
    parser = ProductHtmlExtractor()
    parser.feed(html)

    data = ExtractUrlResponse(source_url=source_url)

    for block in parser.ldjson_blocks:
        try:
            parsed = json.loads(block)
        except Exception:
            continue
        product = find_product_node(parsed)
        if not product:
            continue

        offers = product.get("offers")
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        if not isinstance(offers, dict):
            offers = {}

        brand = product.get("brand")
        if isinstance(brand, dict):
            brand = brand.get("name")

        data = ExtractUrlResponse(
            product_name=str(product.get("name", "")).strip() or None,
            current_price=str(offers.get("price") or product.get("price") or "").strip() or None,
            currency=str(offers.get("priceCurrency") or product.get("priceCurrency") or "").strip() or None,
            brand=str(brand or "").strip() or None,
            gtin13=str(product.get("gtin13") or "").strip() or None,
            gtin8=str(product.get("gtin8") or "").strip() or None,
            sku=str(product.get("sku") or "").strip() or None,
            source_url=source_url,
        )
        break

    if not data.product_name:
        data.product_name = parser.get_meta("property", "og:title") or parser.get_meta("name", "title")
    if not data.current_price:
        data.current_price = (
            parser.get_meta("property", "product:price:amount")
            or parser.get_meta("property", "og:price:amount")
            or parser.get_meta("name", "price")
        )
    if not data.currency:
        data.currency = (
            parser.get_meta("property", "product:price:currency")
            or parser.get_meta("property", "og:price:currency")
            or parser.get_meta("name", "currency")
        )
    if not data.brand:
        data.brand = parser.get_meta("name", "brand") or parser.get_meta("property", "product:brand")

    if data.gtin13:
        digits = re.sub(r"\D", "", data.gtin13)
        data.gtin13 = digits or None
    if data.gtin8:
        digits = re.sub(r"\D", "", data.gtin8)
        data.gtin8 = digits or None

    return data


def infer_brand_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    host_labels = [label for label in host.split(".") if label]
    host_skip = {
        "www",
        "shop",
        "store",
        "m",
        "dk",
        "da",
        "de",
        "en",
        "no",
        "se",
        "fi",
        "com",
        "net",
        "org",
        "eu",
    }
    for label in host_labels:
        if label in host_skip:
            continue
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}", label):
            continue
        if len(label) < 3:
            continue
        if "-" in label:
            continue
        if label == "ecco":
            return "ECCO"
        return label.capitalize()

    path_tokens = [t.strip().lower() for t in parsed.path.split("/") if t.strip()]
    if not path_tokens:
        return None

    skip_tokens = {
        "dk",
        "da",
        "de",
        "en",
        "no",
        "se",
        "fi",
        "nl",
        "fr",
        "it",
        "es",
        "pt",
        "products",
        "product",
        "p",
        "shop",
        "sale",
        "new",
    }

    for token in path_tokens:
        if token in skip_tokens:
            continue
        if any(ch.isdigit() for ch in token):
            continue
        if not re.fullmatch(r"[a-z][a-z0-9-]{1,30}", token):
            continue
        if len(token) < 3:
            continue
        if "-" in token:
            continue
        if token == "www":
            continue
        # Preserve known stylings.
        if token == "ecco":
            return "ECCO"
        return token.capitalize()
    return None


def infer_product_name_from_url(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    path_tokens = [t.strip() for t in parsed.path.split("/") if t.strip()]
    if not path_tokens:
        return None

    skip_tokens = {"produkt", "product", "products", "dk", "da", "de", "en", "shop"}
    candidate: Optional[str] = None
    for token in path_tokens:
        token_l = token.lower()
        if token_l in skip_tokens:
            continue
        if re.fullmatch(r"[0-9]{4,16}", token_l):
            continue
        if re.fullmatch(r"[a-z0-9_-]{3,120}", token_l):
            candidate = token
            if "-" in token_l or "_" in token_l:
                break

    if not candidate:
        return None

    cleaned = re.sub(r"[-_]+", " ", candidate).strip()
    if not cleaned:
        return None

    words: List[str] = []
    for w in cleaned.split():
        if re.fullmatch(r"[0-9]+", w):
            words.append(w)
        elif w.lower() in {"gtx", "gore", "tex", "wp"}:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)


def infer_identifiers_from_url(url: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    digits = re.findall(r"\d+", url)
    gtin13 = next((d for d in digits if len(d) == 13), None)
    gtin8 = next((d for d in digits if len(d) == 8), None)
    sku = next((d for d in digits if 5 <= len(d) <= 12 and d not in {gtin13, gtin8}), None)
    return gtin13, gtin8, sku


def fallback_extract_from_url(url: str) -> ExtractUrlResponse:
    gtin13, gtin8, sku = infer_identifiers_from_url(url)
    return ExtractUrlResponse(
        product_name=infer_product_name_from_url(url),
        current_price=None,
        currency=None,
        brand=infer_brand_from_url(url),
        gtin13=gtin13,
        gtin8=gtin8,
        sku=sku,
        source_url=url,
    )


def merge_extract_data(primary: ExtractUrlResponse, fallback: ExtractUrlResponse) -> ExtractUrlResponse:
    return ExtractUrlResponse(
        product_name=primary.product_name or fallback.product_name,
        current_price=primary.current_price or fallback.current_price,
        currency=primary.currency or fallback.currency,
        brand=primary.brand or fallback.brand,
        gtin13=primary.gtin13 or fallback.gtin13,
        gtin8=primary.gtin8 or fallback.gtin8,
        sku=primary.sku or fallback.sku,
        source_url=primary.source_url,
    )


def extract_product_data_from_url(url: str) -> ExtractUrlResponse:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="URL must be a valid http/https URL")
    fallback = fallback_extract_from_url(url)
    try:
        html = fetch_html(url)
    except HTTPError as exc:
        # Some shops block server-side requests (403/429). Return best-effort URL inference instead.
        if exc.code in {403, 429}:
            return fallback
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: HTTP {exc.code}") from exc
    except Exception as exc:
        if fallback.product_name or fallback.brand or fallback.gtin13 or fallback.sku:
            return fallback
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: {exc}") from exc

    extracted = extract_product_data_from_html(source_url=url, html=html)
    return merge_extract_data(extracted, fallback)


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
            parsed = parse_localized_price(value)
            if parsed is None:
                raise ValueError(f"Invalid price format: {value}")
            return parsed
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


class ExtractUrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)

    @field_validator("url", mode="before")
    @classmethod
    def strip_url(cls, value: Any) -> str:
        s = str(value or "").strip()
        if not s:
            raise ValueError("url is required")
        return s


class ExtractUrlResponse(BaseModel):
    product_name: Optional[str] = None
    current_price: Optional[str] = None
    currency: Optional[str] = None
    brand: Optional[str] = None
    gtin13: Optional[str] = None
    gtin8: Optional[str] = None
    sku: Optional[str] = None
    source_url: str


class Alternative(BaseModel):
    shop: str
    price: float
    currency: str
    url: str
    title: Optional[str] = None
    item_price: Optional[float] = None
    shipping_price: float = 0.0


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


class SearchRequest(AnalyzeRequest):
    limit: int = Field(default=20, ge=1, le=50)


class SearchResponse(BaseModel):
    query_type: str
    currency: str
    total_matches: int
    excluded_results: int
    offers: List[Alternative]


def cache_key(payload: AnalyzeRequest, query: Optional[str] = None) -> str:
    identity = payload.gtin13 or payload.gtin8 or payload.sku or payload.product_name or "unknown"
    query_part = query or identity
    return f"{payload.country}:{identity}:{query_part}:{payload.include_used}:{payload.include_refurbished}"


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
        return parse_localized_price(raw)
    return None


def parse_shipping_price(*values: Any) -> float:
    texts: List[str] = []

    for value in values:
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        if isinstance(value, str):
            t = value.strip()
            if t:
                texts.append(t)
            continue
        if isinstance(value, list):
            for part in value:
                if isinstance(part, str) and part.strip():
                    texts.append(part.strip())

    if not texts:
        return 0.0

    combined = " ".join(texts).lower()
    free_shipping_signals = ["free shipping", "free delivery", "gratis", "fragtfri", "fri levering"]
    if any(signal in combined for signal in free_shipping_signals):
        return 0.0

    # Extract only values that look like money amounts to avoid parsing dates/times.
    money_patterns = [
        r"(?:kr\.?|dkk|€|eur|\$|usd)\s*([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)",
        r"([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?)\s*(?:kr\.?|dkk|€|eur|\$|usd)",
    ]
    for pattern in money_patterns:
        match = re.search(pattern, combined, flags=re.IGNORECASE)
        if match:
            parsed = parse_localized_price(match.group(1))
            if parsed is not None:
                return max(0.0, parsed)

    return 0.0


def normalized_tokens(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    tokens = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
    return {t for t in tokens if t not in TOKEN_STOPWORDS}


def has_brand_match(brand: Optional[str], title: str, source: str) -> bool:
    brand_tokens = normalized_tokens(brand)
    if not brand_tokens:
        return True
    result_tokens = normalized_tokens(f"{title} {source}")
    matched = len(brand_tokens & result_tokens)
    required = max(1, math.ceil(len(brand_tokens) * 0.6))
    return matched >= required


def has_signature_model_match(payload: AnalyzeRequest, title: str) -> bool:
    if not payload.product_name:
        return True

    product_tokens = normalized_tokens(payload.product_name)
    if not product_tokens:
        return True

    brand_tokens = normalized_tokens(payload.brand)
    signature_tokens = product_tokens - brand_tokens - NON_DISTINCTIVE_MODEL_TOKENS
    if not signature_tokens:
        return True

    title_tokens = normalized_tokens(title)
    return bool(signature_tokens & title_tokens)


def is_wrong_model(payload: AnalyzeRequest, title: str) -> bool:
    if not payload.product_name:
        return False
    name_tokens = normalized_tokens(payload.product_name)
    title_tokens = normalized_tokens(title)
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


def is_allowed_source_for_denmark(source: str, shipping: str) -> bool:
    source_l = source.lower().strip()
    shipping_l = shipping.lower().strip()

    if any(blocked in source_l for blocked in BLOCKED_SOURCES):
        return False
    if any(allowed in source_l for allowed in ALLOWED_CROSS_BORDER_SOURCES):
        return True

    # Native DK domains/sources are always allowed.
    if ".dk" in source_l or "/dk" in source_l or source_l.endswith(" dk"):
        return True

    # Danish shops can exist on .com domains; allow if shipping clearly includes Denmark.
    denmark_positive_signals = ["denmark", "danmark", "ships to denmark", "levering til danmark"]
    if any(signal in shipping_l for signal in denmark_positive_signals):
        return True

    # Explicit negative shipping signal -> reject.
    denmark_negative_signals = ["does not ship to denmark", "cannot ship to denmark", "no shipping to denmark"]
    if any(signal in shipping_l for signal in denmark_negative_signals):
        return False

    # Allow known Danish indicators in source names even without .dk TLD.
    if any(marker in source_l for marker in ["denmark", "danmark"]):
        return True

    return False


def is_preferred_child_store(source: str) -> bool:
    source_l = source.lower().strip()
    return any(marker in source_l for marker in PREFERRED_CHILD_STORES)


def parse_and_filter_results(payload: AnalyzeRequest, raw_results: List[Dict[str, Any]]) -> tuple[List[Alternative], int]:
    filtered: List[Alternative] = []
    excluded = 0

    for r in raw_results:
        title = str(r.get("title", "")).strip()
        source = str(r.get("source", "Unknown shop")).strip() or "Unknown shop"
        link = str(r.get("link") or r.get("product_link") or "").strip()
        condition = " ".join(
            [
                str(r.get("condition", "")).lower(),
                str(r.get("second_hand_condition", "")).lower(),
            ]
        ).strip()
        shipping = " ".join([str(r.get("shipping", "")).lower(), str(r.get("delivery", "")).lower()]).strip()

        price = parse_price(r.get("price") or r.get("extracted_price"))
        if price is None or not link:
            excluded += 1
            continue
        shipping_price = parse_shipping_price(r.get("shipping"), r.get("delivery"))
        total_price = round(price + shipping_price, 2)

        if payload.country.lower() == "dk":
            if not is_allowed_source_for_denmark(source, shipping):
                excluded += 1
                continue

        if ("used" in condition or "used" in title.lower()) and not payload.include_used:
            excluded += 1
            continue

        if ("refurbished" in condition or "refurbished" in title.lower()) and not payload.include_refurbished:
            excluded += 1
            continue

        if payload.brand and not has_brand_match(payload.brand, title=title, source=source):
            excluded += 1
            continue

        if not has_signature_model_match(payload, title=title):
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
                price=total_price,
                currency=payload.currency or DEFAULT_CURRENCY,
                url=link,
                title=title or None,
                item_price=round(price, 2),
                shipping_price=round(shipping_price, 2),
            )
        )

    preferred = [alt for alt in filtered if is_preferred_child_store(alt.shop)]
    if preferred:
        filtered = preferred

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


@app.get("/", include_in_schema=False, response_model=None)
def root():
    if WEBAPP_INDEX.exists():
        return RedirectResponse(url="/app", status_code=307)
    return {"status": "ok"}


@app.get("/app", include_in_schema=False)
def web_app() -> FileResponse:
    if not WEBAPP_INDEX.exists():
        raise HTTPException(status_code=404, detail="Web app not found")
    return FileResponse(WEBAPP_INDEX)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/extract-web", response_model=ExtractUrlResponse, include_in_schema=False)
def extract_web(payload: ExtractUrlRequest) -> ExtractUrlResponse:
    return extract_product_data_from_url(payload.url)


def normalize_payload_for_denmark(payload: AnalyzeRequest) -> AnalyzeRequest:
    return payload.model_copy(update={"country": DEFAULT_COUNTRY, "currency": DEFAULT_CURRENCY})


def resolve_alternatives(payload: AnalyzeRequest, currency: str) -> tuple[List[Alternative], int, str]:
    query, query_type = build_search_query(payload)
    search_variants: List[tuple[str, str]] = [(query, query_type)]
    if query_type in {"gtin_exact", "sku_brand", "sku"} and payload.product_name:
        if payload.brand:
            search_variants.append((f'"{payload.brand}" {payload.product_name}', "name_brand_fuzzy_fallback"))
        search_variants.append((payload.product_name, "name_fuzzy_fallback"))

    alternatives: List[Alternative] = []
    excluded = 0
    selected_query_type = query_type

    # Try primary search first; fallback from SKU/GTIN to name-based search when matches collapse to zero.
    for search_query, variant_type in search_variants:
        key = cache_key(payload, search_query)
        raw_results = get_cached(key)
        if raw_results is None:
            raw_results = call_serpapi(query=search_query, country=payload.country, currency=currency)
            set_cached(key, raw_results)

        current_alts, current_excluded = parse_and_filter_results(payload, raw_results)
        if current_alts:
            alternatives = current_alts
            excluded = current_excluded
            selected_query_type = variant_type
            break

        # Preserve the latest excluded count for diagnostics if all variants fail.
        excluded = current_excluded

    return alternatives, excluded, selected_query_type


def analyze_impl(payload: AnalyzeRequest) -> AnalyzeResponse:
    # Product logic is Denmark-only by requirement.
    payload = normalize_payload_for_denmark(payload)
    currency = DEFAULT_CURRENCY

    try:
        alternatives, excluded, selected_query_type = resolve_alternatives(payload, currency)

        market_min_price = alternatives[0].price if alternatives else None
        market_avg_price = mean([a.price for a in alternatives]) if alternatives else None

        verdict = decide_verdict(payload.current_price, market_min_price, market_avg_price)

        savings = None
        if payload.current_price is not None and market_min_price is not None:
            savings = round(payload.current_price - market_min_price, 2)

        return AnalyzeResponse(
            verdict=verdict,
            query_type=selected_query_type,
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


def search_impl(payload: SearchRequest) -> SearchResponse:
    base_payload = AnalyzeRequest.model_validate(payload.model_dump(exclude={"limit"}))
    base_payload = normalize_payload_for_denmark(base_payload)
    currency = DEFAULT_CURRENCY

    try:
        alternatives, excluded, selected_query_type = resolve_alternatives(base_payload, currency)
        return SearchResponse(
            query_type=selected_query_type,
            currency=currency,
            total_matches=len(alternatives),
            excluded_results=excluded,
            offers=alternatives[: payload.limit],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled search failure")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@app.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, x_api_key: Optional[str] = Header(default=None)) -> AnalyzeResponse:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return analyze_impl(payload)


@app.post("/v1/analyze-web", response_model=AnalyzeResponse, include_in_schema=False)
def analyze_web(payload: AnalyzeRequest) -> AnalyzeResponse:
    return analyze_impl(payload)


@app.post("/v1/search", response_model=SearchResponse)
def search(payload: SearchRequest, x_api_key: Optional[str] = Header(default=None)) -> SearchResponse:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY is not configured")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return search_impl(payload)


@app.post("/v1/search-web", response_model=SearchResponse, include_in_schema=False)
def search_web(payload: SearchRequest) -> SearchResponse:
    return search_impl(payload)
