from __future__ import annotations

import json
import hashlib
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


def load_local_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env_file(Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("API_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_COUNTRY = "dk"
DEFAULT_CURRENCY = "DKK"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "1800"))
LLM_QUERY_CACHE_TTL_SECONDS = int(os.getenv("LLM_QUERY_CACHE_TTL_SECONDS", "21600"))
SERPAPI_MAX_RETRIES = int(os.getenv("SERPAPI_MAX_RETRIES", "3"))
SERPAPI_RETRY_BACKOFF_SECONDS = float(os.getenv("SERPAPI_RETRY_BACKOFF_SECONDS", "0.5"))
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
MAX_QUERY_VARIANTS = 10
EARLY_STOP_MIN_UNIQUE_MATCHES = 8
LLM_QUERY_TRIGGER_MIN_MATCHES = 3
QUERY_TERM_GROUPS = [
    {"hoodie", "sweatshirt", "sweater", "jumper", "trøje", "troeje"},
    {"blå", "blaa", "blue", "royal", "navy"},
]
QUERY_STRIP_TOKENS = {
    "dreng",
    "drenge",
    "boy",
    "boys",
    "pige",
    "piger",
    "girl",
    "girls",
    "barn",
    "børn",
    "boern",
    "kid",
    "kids",
}
PRODUCT_TYPE_GROUPS: Dict[str, set[str]] = {
    "winter_boot": {"vinterstøvle", "vinterstøvler", "winter boot", "winter boots", "snow boot", "snow boots"},
    "rain_boot": {"gummistøvle", "gummistøvler", "rain boot", "rain boots", "wellies", "wellington"},
    "boot_general": {"støvle", "støvler", "boot", "boots"},
    "shoe": {"sko", "shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers"},
    "hoodie_top": {"hoodie", "sweatshirt", "sweater", "jumper", "trøje", "troeje"},
    "jacket": {"jakke", "jacket", "coat", "anorak", "parka"},
    "sandals": {"sandal", "sandaler", "sandals"},
}
PRODUCT_FEATURE_GROUPS: Dict[str, set[str]] = {
    "winter": {"vinter", "winter", "snow", "sne"},
    "waterproof": {"vandtæt", "vandtætte", "waterproof", "gtx", "gore-tex", "goretex"},
    "rain": {"regn", "rain", "gummi", "wellies", "wellington"},
}
COLOR_GROUPS: Dict[str, set[str]] = {
    "black": {"black", "sort", "noir", "nero"},
    "grey": {"grey", "gray", "grå", "gra", "antracit", "anthracite"},
    "white": {"white", "hvid", "blanc"},
    "blue": {"blue", "blå", "blaa", "navy", "royal", "indigo", "marine"},
    "red": {"red", "rød", "roed", "burgundy", "bordeaux"},
    "green": {"green", "grøn", "groen", "olive", "khaki"},
    "pink": {"pink", "rosa", "fuchsia"},
    "purple": {"purple", "lilla", "violet"},
    "yellow": {"yellow", "gul"},
    "orange": {"orange"},
    "brown": {"brown", "brun", "tan", "cognac"},
    "beige": {"beige", "sand", "cream", "creme", "ivory"},
}
MIN_MATCH_SCORE_DEFAULT = 0.50
MIN_MATCH_SCORE_BRAND = 0.58
MIN_MATCH_SCORE_STRONG_ID = 0.40
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
GENERIC_APPAREL_QUERY_TOKENS = {
    "hoodie",
    "sweatshirt",
    "sweater",
    "jumper",
    "shirt",
    "trøje",
    "troeje",
    "blå",
    "blaa",
    "blue",
    "royal",
    "navy",
    "dreng",
    "drenge",
    "boy",
    "boys",
    "pige",
    "piger",
    "girl",
    "girls",
    "barn",
    "børn",
    "boern",
    "kid",
    "kids",
}
NON_DISTINCTIVE_MODEL_TOKENS = {
    "gtx",
    "gore",
    "tex",
    "warm",
    "vinter",
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
LLM_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


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
    store_filters: Optional[List[str]] = None
    query_variants: Optional[List[str]] = None
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

    @field_validator("store_filters", mode="before")
    @classmethod
    def normalize_store_filters(cls, value: Any) -> Optional[List[str]]:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            raise ValueError("store_filters must be a list of strings")

        cleaned: List[str] = []
        for item in values:
            marker = str(item).strip().lower()
            if marker and marker not in cleaned:
                cleaned.append(marker)
        return cleaned or None

    @field_validator("query_variants", mode="before")
    @classmethod
    def normalize_query_variants(cls, value: Any) -> Optional[List[str]]:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        else:
            raise ValueError("query_variants must be a list of strings")

        cleaned: List[str] = []
        seen: set[str] = set()
        for item in values:
            query = str(item).strip()
            if not query:
                continue
            query_key = query.lower()
            if query_key in seen:
                continue
            seen.add(query_key)
            cleaned.append(query)
            if len(cleaned) >= MAX_QUERY_VARIANTS:
                break
        return cleaned or None


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
    image_url: Optional[str] = None
    match_score: Optional[float] = None


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
    excluded_reasons: Dict[str, int] = Field(default_factory=dict)


class SearchRequest(AnalyzeRequest):
    limit: int = Field(default=20, ge=1, le=50)


class SearchResponse(BaseModel):
    query_type: str
    currency: str
    total_matches: int
    excluded_results: int
    excluded_reasons: Dict[str, int] = Field(default_factory=dict)
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


def llm_cache_key(payload: AnalyzeRequest) -> str:
    identity = {
        "product_name": payload.product_name or "",
        "brand": payload.brand or "",
        "sku": payload.sku or "",
        "gtin13": payload.gtin13 or "",
        "gtin8": payload.gtin8 or "",
        "store_filters": payload.store_filters or [],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"llm:{digest}"


def get_llm_cached(key: str) -> Optional[List[str]]:
    record = LLM_QUERY_CACHE.get(key)
    if not record:
        return None
    if time.time() - record["ts"] > LLM_QUERY_CACHE_TTL_SECONDS:
        LLM_QUERY_CACHE.pop(key, None)
        return None
    queries = record.get("queries")
    if not isinstance(queries, list):
        return None
    return [str(q) for q in queries if str(q).strip()]


def set_llm_cached(key: str, queries: List[str]) -> None:
    LLM_QUERY_CACHE[key] = {"ts": time.time(), "queries": queries}


def normalize_query_lines(raw_text: str) -> List[str]:
    lines = re.split(r"[\r\n]+", raw_text)
    cleaned: List[str] = []
    seen: set[str] = set()
    for line in lines:
        item = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        item = clean_query_text(item)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
        if len(cleaned) >= MAX_QUERY_VARIANTS:
            break
    return cleaned


def build_query_boost_prompt(payload: AnalyzeRequest) -> str:
    return (
        "Du er query-generator for dansk e-commerce prissøgning.\n"
        "Lav 8 korte søgequeries for samme produkt.\n\n"
        "Krav:\n"
        "- Fokus på danske butikker og shopping-søgning.\n"
        "- Behold brand/model hvis kendt.\n"
        "- Lav variationer af produkttype (fx hoodie/sweatshirt/sweater).\n"
        "- Lav variationer af farveord (fx blå/blue/royal/navy), men kun relevante.\n"
        "- Fjern støj (fx køn/alder ord) i nogle varianter.\n"
        "- Ingen forklaringer.\n"
        "- Output kun rå queries, én pr linje.\n\n"
        f"Produktnavn: {payload.product_name or ''}\n"
        f"Brand: {payload.brand or ''}\n"
        f"SKU: {payload.sku or ''}\n"
        f"GTIN: {payload.gtin13 or payload.gtin8 or ''}\n"
    )


def extract_response_output_text(data: Dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: List[str] = []
    for item in data.get("output", []) if isinstance(data.get("output"), list) else []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(content, dict):
                continue
            for key in ("text", "output_text"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return "\n".join(chunks)


def request_llm_query_variants(payload: AnalyzeRequest) -> List[str]:
    if not OPENAI_API_KEY or not payload.product_name:
        return []

    prompt = build_query_boost_prompt(payload)
    body = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "max_output_tokens": 220,
    }

    req = Request(
        url="https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=18) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            data = json.loads(response.read().decode(charset, errors="replace"))
    except Exception as exc:
        logger.warning("LLM query boost failed: %s", exc)
        return []

    if not isinstance(data, dict):
        return []
    return normalize_query_lines(extract_response_output_text(data))


def get_auto_query_variants(payload: AnalyzeRequest) -> List[str]:
    if not OPENAI_API_KEY or not payload.product_name:
        return []
    key = llm_cache_key(payload)
    cached = get_llm_cached(key)
    if cached is not None:
        return cached
    queries = request_llm_query_variants(payload)
    # Cache empty responses too to avoid repeated slow/noisy LLM retries.
    set_llm_cached(key, queries)
    return queries


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


def clean_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def expand_term_group(query: str, group: set[str]) -> List[str]:
    query_l = query.lower()
    hits = [term for term in group if re.search(rf"\b{re.escape(term)}\b", query_l)]
    if not hits:
        return []

    variants: List[str] = []
    for hit in hits:
        for replacement in group:
            if replacement == hit:
                continue
            variant = re.sub(
                rf"\b{re.escape(hit)}\b",
                replacement,
                query,
                flags=re.IGNORECASE,
            )
            variant = clean_query_text(variant)
            if variant and variant.lower() != query_l:
                variants.append(variant)
    return variants


def strip_age_size_tokens(query: str) -> str:
    q = query
    q = re.sub(r"\b\d{1,3}\s*(?:år|aar|yrs?|year|years)\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b\d{2,3}\s*/\s*\d{2,3}\b", " ", q)
    q = re.sub(r"\b(?:str|size)\s*[:.]?\s*\d{2,3}(?:\s*/\s*\d{2,3})?\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\b\d{2,3}\s*cm\b", " ", q, flags=re.IGNORECASE)
    return clean_query_text(q)


def strip_generic_query_tokens(query: str) -> str:
    words = [w for w in re.findall(r"\w+", query, flags=re.UNICODE) if w.lower() not in QUERY_STRIP_TOKENS]
    return clean_query_text(" ".join(words))


def build_search_variants(payload: AnalyzeRequest, extra_variants: Optional[List[str]] = None) -> List[tuple[str, str]]:
    base_query, base_type = build_search_query(payload)
    variants: List[tuple[str, str]] = [(base_query, base_type)]

    if payload.query_variants:
        for query in payload.query_variants:
            variants.append((query, "gpt_variant"))
    if extra_variants:
        for query in extra_variants:
            variants.append((query, "llm_auto"))

    if payload.product_name:
        base_name = clean_query_text(payload.product_name)
        heuristic_names: List[str] = [base_name]

        stripped_age = strip_age_size_tokens(base_name)
        if stripped_age and stripped_age.lower() != base_name.lower():
            heuristic_names.append(stripped_age)

        stripped_generic = strip_generic_query_tokens(stripped_age or base_name)
        if stripped_generic and stripped_generic.lower() not in {q.lower() for q in heuristic_names}:
            heuristic_names.append(stripped_generic)

        for q in list(heuristic_names):
            for group in QUERY_TERM_GROUPS:
                heuristic_names.extend(expand_term_group(q, group))

        for query in heuristic_names:
            query_clean = clean_query_text(query)
            if not query_clean:
                continue
            variants.append((query_clean, "name_expanded"))
            if payload.brand:
                variants.append((f'"{payload.brand}" {query_clean}', "name_brand_expanded"))

    if base_type in {"gtin_exact", "sku_brand", "sku"} and payload.product_name:
        if payload.brand:
            variants.append((f'"{payload.brand}" {payload.product_name}', "name_brand_fuzzy_fallback"))
        variants.append((payload.product_name, "name_fuzzy_fallback"))

    unique: List[tuple[str, str]] = []
    seen: set[str] = set()
    for query, qtype in variants:
        normalized = clean_query_text(query)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((normalized, qtype))
        if len(unique) >= MAX_QUERY_VARIANTS:
            break
    return unique


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

    retries = max(1, SERPAPI_MAX_RETRIES)
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            search = GoogleSearch(params)
            data = search.get_dict()
            if not isinstance(data, dict):
                raise HTTPException(status_code=502, detail="Malformed search response")

            error_text = str(data.get("error") or "").lower()
            should_retry = (
                "429" in error_text
                or "rate" in error_text
                or "limit" in error_text
                or "403" in error_text
                or "temporar" in error_text
                or "timeout" in error_text
            )
            if should_retry and attempt < retries:
                sleep_s = SERPAPI_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "SerpApi retryable error for query '%s' (attempt %s/%s): %s",
                    query,
                    attempt,
                    retries,
                    error_text,
                )
                time.sleep(max(0.0, sleep_s))
                continue

            if error_text:
                raise HTTPException(status_code=502, detail=f"Search provider failure: {data.get('error')}")

            return data.get("shopping_results", []) or []
        except HTTPException:
            raise
        except Exception as exc:  # network/timeouts and client errors
            last_error = exc
            if attempt < retries:
                sleep_s = SERPAPI_RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    "SerpApi call failed (attempt %s/%s) for query '%s': %s",
                    attempt,
                    retries,
                    query,
                    exc,
                )
                time.sleep(max(0.0, sleep_s))
                continue
            logger.exception("SerpApi call failed")
            raise HTTPException(status_code=502, detail=f"Search provider failure: {exc}") from exc

    raise HTTPException(status_code=502, detail=f"Search provider failure: {last_error}")


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


def meaningful_match_tokens(text: Optional[str]) -> set[str]:
    tokens = normalized_tokens(text)
    return {
        t
        for t in tokens
        if t not in GENERIC_APPAREL_QUERY_TOKENS and not any(ch.isdigit() for ch in t)
    }


def contains_keyword(text: str, keyword: str) -> bool:
    text_l = text.lower()
    kw = keyword.lower()
    if " " in kw or "-" in kw:
        return kw in text_l
    return re.search(rf"\b{re.escape(kw)}\b", text_l) is not None


def extract_type_groups(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    text_l = text.lower()
    groups: set[str] = set()
    for group, keywords in PRODUCT_TYPE_GROUPS.items():
        if any(contains_keyword(text_l, kw) for kw in keywords):
            groups.add(group)
    if "boot_general" in groups and ("winter_boot" in groups or "rain_boot" in groups):
        groups.discard("boot_general")
    return groups


def extract_feature_groups(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    text_l = text.lower()
    groups: set[str] = set()
    for group, keywords in PRODUCT_FEATURE_GROUPS.items():
        if any(contains_keyword(text_l, kw) for kw in keywords):
            groups.add(group)
    return groups


def extract_color_groups(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    text_l = text.lower()
    groups: set[str] = set()
    for group, keywords in COLOR_GROUPS.items():
        if any(contains_keyword(text_l, kw) for kw in keywords):
            groups.add(group)
    return groups


def extract_age_years(text: Optional[str]) -> set[int]:
    if not text:
        return set()
    years: set[int] = set()
    for match in re.finditer(r"\b(\d{1,2})\s*(?:år|aar|yrs?|year|years)\b", text.lower()):
        try:
            years.add(int(match.group(1)))
        except Exception:
            continue
    return years


def extract_size_markers(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    markers: set[str] = set()
    for m in re.finditer(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text):
        a, b = m.group(1), m.group(2)
        markers.add(f"{a}/{b}")
        markers.add(a)
        markers.add(b)
    for m in re.finditer(r"\b(?:str|size)\s*[:.]?\s*(\d{2,3})\b", text.lower()):
        markers.add(m.group(1))
    for m in re.finditer(r"\b(\d{2,3})\s*-\s*(\d{2,3})\b", text):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 < abs(a - b) <= 25:
            markers.add(str(a))
            markers.add(str(b))
            markers.add(f"{a}-{b}")
    return markers


def has_broad_size_range(text: Optional[str]) -> bool:
    if not text:
        return False
    for m in re.finditer(r"\b(\d{2,3})\s*-\s*(\d{2,3})\b", text):
        start, end = int(m.group(1)), int(m.group(2))
        if abs(end - start) >= 8:
            return True
    return False


def has_age_size_compatibility(query_text: Optional[str], title_text: str) -> bool:
    query_ages = extract_age_years(query_text)
    title_ages = extract_age_years(title_text)
    if query_ages and title_ages and not (query_ages & title_ages):
        return False

    query_sizes = extract_size_markers(query_text)
    title_sizes = extract_size_markers(title_text)
    if query_sizes and title_sizes and not (query_sizes & title_sizes):
        return False

    return True


def has_product_type_conflict(query_text: Optional[str], title_text: str) -> bool:
    query_groups = extract_type_groups(query_text)
    title_groups = extract_type_groups(title_text)
    if not query_groups or not title_groups:
        return False
    if query_groups & title_groups:
        return False

    # Special-case: general boots can match specific boot subtypes.
    if "boot_general" in query_groups and {"winter_boot", "rain_boot"} & title_groups:
        return False
    if "boot_general" in title_groups and {"winter_boot", "rain_boot"} & query_groups:
        return False

    return True


def has_required_feature_match(query_text: Optional[str], title_text: str) -> bool:
    query_features = extract_feature_groups(query_text)
    if not query_features:
        return True
    title_features = extract_feature_groups(title_text)
    if not title_features:
        return True

    if "waterproof" in query_features and "waterproof" not in title_features:
        return False
    if "winter" in query_features and "rain" in title_features and "winter" not in title_features:
        return False
    return True


def has_color_conflict(query_text: Optional[str], title_text: str, link_text: str = "") -> bool:
    query_colors = extract_color_groups(query_text)
    if not query_colors:
        return False

    title_colors = extract_color_groups(f"{title_text} {link_text}")
    if not title_colors:
        # Unknown color in result title/link -> do not hard reject.
        return False

    return not bool(query_colors & title_colors)


def get_brand_tokens(brand: Optional[str]) -> set[str]:
    if not brand:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", brand.lower()) if len(t) >= 2}


def has_brand_match(brand: Optional[str], title: str, source: str, link: str = "") -> bool:
    brand_tokens = get_brand_tokens(brand)
    if not brand_tokens:
        return True
    haystack = f"{title} {source} {link}".lower()
    if brand and brand.strip().lower() in haystack:
        return True
    result_tokens = {t for t in re.findall(r"[a-z0-9]+", haystack) if len(t) >= 2}
    matched = len(brand_tokens & result_tokens)
    required = max(1, math.ceil(len(brand_tokens) * 0.7))
    return matched >= required


def has_signature_model_match(payload: AnalyzeRequest, title: str) -> bool:
    if not payload.product_name:
        return True

    product_tokens = meaningful_match_tokens(payload.product_name)
    if not product_tokens:
        return True

    brand_tokens = normalized_tokens(payload.brand)
    signature_tokens = product_tokens - brand_tokens - NON_DISTINCTIVE_MODEL_TOKENS
    if not signature_tokens:
        return True

    title_tokens = meaningful_match_tokens(title)
    if signature_tokens & title_tokens:
        return True

    title_l = title.lower()
    for token in signature_tokens:
        if len(token) >= 5 and token in title_l:
            return True
    return False


def is_wrong_model(payload: AnalyzeRequest, title: str) -> bool:
    if not payload.product_name:
        return False
    name_tokens = meaningful_match_tokens(payload.product_name)
    title_tokens = meaningful_match_tokens(title)
    if not name_tokens:
        return False
    overlap = len(name_tokens & title_tokens) / max(len(name_tokens), 1)
    # Be less strict when user already narrowed search by brand/store filter.
    threshold = 0.45
    if payload.brand or payload.store_filters:
        threshold = 0.30
    # Broad fuzzy queries with few words should not be over-penalized.
    if not (payload.gtin13 or payload.gtin8 or payload.sku) and len(name_tokens) <= 3:
        threshold = min(threshold, 0.30)
    return overlap < threshold


def minimum_match_score(payload: AnalyzeRequest) -> float:
    if payload.gtin13 or payload.gtin8 or payload.sku:
        return MIN_MATCH_SCORE_STRONG_ID
    if payload.brand:
        return MIN_MATCH_SCORE_BRAND
    return MIN_MATCH_SCORE_DEFAULT


def compute_match_score(payload: AnalyzeRequest, title: str, source: str, link: str) -> float:
    score = 0.0

    if payload.brand:
        if has_brand_match(payload.brand, title=title, source=source, link=link):
            score += 0.30
    else:
        score += 0.08

    query_groups = extract_type_groups(payload.product_name)
    title_groups = extract_type_groups(title)
    if query_groups:
        if query_groups & title_groups:
            score += 0.24
        elif "boot_general" in query_groups and {"winter_boot", "rain_boot"} & title_groups:
            score += 0.18
        elif not title_groups:
            score += 0.08
    else:
        score += 0.08

    query_features = extract_feature_groups(payload.product_name)
    title_features = extract_feature_groups(title)
    if query_features:
        ratio = len(query_features & title_features) / len(query_features)
        score += 0.16 * ratio
    else:
        score += 0.05

    query_colors = extract_color_groups(payload.product_name)
    title_colors = extract_color_groups(f"{title} {link}")
    if query_colors:
        if title_colors and (query_colors & title_colors):
            score += 0.14
        elif not title_colors:
            score += 0.02
    else:
        score += 0.04

    query_tokens = meaningful_match_tokens(payload.product_name)
    title_tokens = meaningful_match_tokens(title)
    if query_tokens:
        overlap_ratio = len(query_tokens & title_tokens) / len(query_tokens)
        score += 0.17 * overlap_ratio
    else:
        score += 0.08

    query_has_age_size = bool(extract_age_years(payload.product_name) or extract_size_markers(payload.product_name))
    title_has_age_size = bool(extract_age_years(title) or extract_size_markers(title))
    if query_has_age_size:
        if has_age_size_compatibility(payload.product_name, title):
            score += 0.13 if title_has_age_size else 0.06
    else:
        score += 0.05

    return round(min(score, 1.0), 3)


def contains_accessory_signal(product_name: Optional[str], title: str) -> bool:
    if not product_name:
        return False
    title_l = title.lower()
    product_l = product_name.lower()
    accessory_terms = {"case", "cover", "belt", "charger", "strap", "cable", "holder", "refill"}
    if any(t in title_l for t in accessory_terms) and not any(t in product_l for t in accessory_terms):
        return True
    return False


def is_allowed_source_for_denmark(source: str, shipping: str, link: str = "") -> bool:
    source_l = source.lower().strip()
    shipping_l = shipping.lower().strip()
    link_l = link.lower().strip()

    if any(blocked in source_l or blocked in link_l for blocked in BLOCKED_SOURCES):
        return False
    if any(allowed in source_l or allowed in link_l for allowed in ALLOWED_CROSS_BORDER_SOURCES):
        return True

    # Native DK domains/sources are always allowed.
    if (
        ".dk" in source_l
        or source_l.endswith(" dk")
        or ".dk/" in link_l
        or link_l.startswith("https://dk.")
        or link_l.startswith("http://dk.")
    ):
        return True

    # Locale-specific pages on non-.dk domains can still be Denmark storefronts.
    locale_markers = [
        "/da-dk/",
        "/da_dk/",
        "locale=da-dk",
        "locale=da_dk",
        "lang=da",
        "country=dk",
        "market=dk",
    ]
    if any(marker in link_l for marker in locale_markers):
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


def matches_store_filters(source: str, link: str, store_filters: Optional[List[str]]) -> bool:
    if not store_filters:
        return True
    haystack = f"{source} {link}".lower()
    return any(marker in haystack for marker in store_filters)


def extract_image_url(raw_result: Dict[str, Any]) -> Optional[str]:
    direct_candidates: List[Any] = [
        raw_result.get("thumbnail"),
        raw_result.get("image"),
        raw_result.get("image_url"),
    ]
    for candidate in direct_candidates:
        if isinstance(candidate, str) and candidate.strip().startswith(("http://", "https://")):
            return candidate.strip()

    list_candidates: List[Any] = [
        raw_result.get("thumbnails"),
        raw_result.get("serpapi_thumbnails"),
    ]
    for candidate_list in list_candidates:
        if not isinstance(candidate_list, list):
            continue
        for item in candidate_list:
            if isinstance(item, str) and item.strip().startswith(("http://", "https://")):
                return item.strip()
            if isinstance(item, dict):
                for key in ("thumbnail", "image", "url", "link"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip().startswith(("http://", "https://")):
                        return value.strip()

    return None


def parse_and_filter_results(
    payload: AnalyzeRequest,
    raw_results: List[Dict[str, Any]],
) -> tuple[List[Alternative], int, Dict[str, int]]:
    filtered: List[Alternative] = []
    excluded = 0
    excluded_reasons: Dict[str, int] = {}

    def exclude(reason: str) -> None:
        nonlocal excluded
        excluded += 1
        excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    for r in raw_results:
        title = str(r.get("title", "")).strip()
        source = str(r.get("source", "Unknown shop")).strip() or "Unknown shop"
        link = str(r.get("link") or r.get("product_link") or "").strip()
        if not link.lower().startswith(("http://", "https://")):
            exclude("invalid_link")
            continue
        condition = " ".join(
            [
                str(r.get("condition", "")).lower(),
                str(r.get("second_hand_condition", "")).lower(),
            ]
        ).strip()
        shipping = " ".join([str(r.get("shipping", "")).lower(), str(r.get("delivery", "")).lower()]).strip()

        price = parse_price(r.get("price") or r.get("extracted_price"))
        if price is None or not link:
            exclude("missing_price_or_link")
            continue
        shipping_price = parse_shipping_price(r.get("shipping"), r.get("delivery"))
        total_price = round(price + shipping_price, 2)

        if payload.country.lower() == "dk":
            if not is_allowed_source_for_denmark(source, shipping, link):
                exclude("not_allowed_in_denmark")
                continue

        if not matches_store_filters(source=source, link=link, store_filters=payload.store_filters):
            exclude("store_filter_mismatch")
            continue

        if ("used" in condition or "used" in title.lower()) and not payload.include_used:
            exclude("used_filtered")
            continue

        if ("refurbished" in condition or "refurbished" in title.lower()) and not payload.include_refurbished:
            exclude("refurbished_filtered")
            continue

        if payload.brand and not has_brand_match(payload.brand, title=title, source=source, link=link):
            exclude("brand_mismatch")
            continue

        if has_product_type_conflict(payload.product_name, title):
            exclude("product_type_conflict")
            continue

        if not has_required_feature_match(payload.product_name, title):
            exclude("required_feature_missing")
            continue

        if has_color_conflict(payload.product_name, title, link):
            exclude("color_conflict")
            continue

        if not has_age_size_compatibility(payload.product_name, title):
            exclude("age_or_size_mismatch")
            continue

        if not has_signature_model_match(payload, title=title):
            exclude("signature_model_missing")
            continue

        if contains_accessory_signal(payload.product_name, title):
            exclude("accessory_mismatch")
            continue

        if payload.gtin13 or payload.gtin8:
            gtin = payload.gtin13 or payload.gtin8
            if gtin and gtin not in json.dumps(r).replace(" ", ""):
                # keep result if Google omitted gtin in blob but only when title strongly matches
                if is_wrong_model(payload, title):
                    exclude("wrong_model")
                    continue
        elif payload.product_name and is_wrong_model(payload, title):
            exclude("wrong_model")
            continue

        match_score = compute_match_score(payload, title=title, source=source, link=link)
        if match_score < minimum_match_score(payload):
            exclude("match_score_too_low")
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
                image_url=extract_image_url(r),
                match_score=match_score,
            )
        )

    preferred = [alt for alt in filtered if is_preferred_child_store(alt.shop)]
    if preferred:
        filtered = preferred

    filtered.sort(key=lambda x: (-(x.match_score or 0.0), x.price))
    return filtered, excluded, excluded_reasons


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


def alternative_dedupe_key(alt: Alternative) -> str:
    if alt.url:
        return alt.url.strip().lower()
    return f"{alt.shop.lower().strip()}::{(alt.title or '').lower().strip()}"


def dedupe_alternatives(alternatives: List[Alternative]) -> List[Alternative]:
    deduped: Dict[str, Alternative] = {}
    for alt in alternatives:
        key = alternative_dedupe_key(alt)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = alt
            continue
        existing_score = existing.match_score or 0.0
        candidate_score = alt.match_score or 0.0
        if candidate_score > existing_score:
            deduped[key] = alt
            continue
        if candidate_score == existing_score and alt.price < existing.price:
            deduped[key] = alt
    return sorted(deduped.values(), key=lambda x: (-(x.match_score or 0.0), x.price))


def resolve_alternatives(payload: AnalyzeRequest, currency: str) -> tuple[List[Alternative], int, Dict[str, int], str]:
    search_variants = build_search_variants(payload)
    if not search_variants:
        raise HTTPException(status_code=400, detail="No valid search variants available")

    all_alternatives: List[Alternative] = []
    excluded_total = 0
    excluded_reasons_total: Dict[str, int] = {}
    selected_query_type = search_variants[0][1]
    processed_queries: set[str] = set()

    def run_variants(variants: List[tuple[str, str]]) -> None:
        nonlocal excluded_total, selected_query_type, all_alternatives, excluded_reasons_total
        for search_query, variant_type in variants:
            normalized_query = clean_query_text(search_query)
            query_key = normalized_query.lower()
            if not normalized_query or query_key in processed_queries:
                continue
            processed_queries.add(query_key)

            key = cache_key(payload, normalized_query)
            raw_results = get_cached(key)
            if raw_results is None:
                raw_results = call_serpapi(query=normalized_query, country=payload.country, currency=currency)
                set_cached(key, raw_results)

            current_alts, current_excluded, current_reasons = parse_and_filter_results(payload, raw_results)
            excluded_total += current_excluded
            for reason, count in current_reasons.items():
                excluded_reasons_total[reason] = excluded_reasons_total.get(reason, 0) + count
            if current_alts:
                if not all_alternatives:
                    selected_query_type = variant_type
                all_alternatives.extend(current_alts)
                if len(dedupe_alternatives(all_alternatives)) >= EARLY_STOP_MIN_UNIQUE_MATCHES:
                    break

    run_variants(search_variants)

    if len(dedupe_alternatives(all_alternatives)) < LLM_QUERY_TRIGGER_MIN_MATCHES and not payload.query_variants:
        auto_variants = get_auto_query_variants(payload)
        if auto_variants:
            run_variants(build_search_variants(payload, extra_variants=auto_variants))

    return dedupe_alternatives(all_alternatives), excluded_total, excluded_reasons_total, selected_query_type


def analyze_impl(payload: AnalyzeRequest) -> AnalyzeResponse:
    # Product logic is Denmark-only by requirement.
    payload = normalize_payload_for_denmark(payload)
    currency = DEFAULT_CURRENCY

    try:
        alternatives, excluded, excluded_reasons, selected_query_type = resolve_alternatives(payload, currency)

        market_min_price = min((a.price for a in alternatives), default=None)
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
            excluded_reasons=excluded_reasons,
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
        alternatives, excluded, excluded_reasons, selected_query_type = resolve_alternatives(base_payload, currency)
        return SearchResponse(
            query_type=selected_query_type,
            currency=currency,
            total_matches=len(alternatives),
            excluded_results=excluded,
            excluded_reasons=excluded_reasons,
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
