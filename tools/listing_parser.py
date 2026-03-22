"""
Tool 1: Listing Parser
Fetches a Swedish real estate listing URL and uses an LLM to extract
structured property data from the HTML — works across any agency site.

Many Swedish agency sites (Erik Olsson, Widerlov, etc.) are JavaScript-rendered
SPAs that return minimal HTML to simple HTTP clients. We handle this by:
1. Trying a direct fetch with realistic browser headers
2. Extracting data from JSON-LD / structured data if present in raw HTML
3. Parsing URL slug for clues (rooms, address, etc.)
4. Using LLM to extract from whatever content we get
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
import json
import re


class PropertyData(BaseModel):
    address: Optional[str] = Field(None, description="Street address with floor")
    asking_price: Optional[int] = Field(None, description="Asking price in SEK")
    size_sqm: Optional[float] = Field(None, description="Living area in m2")
    rooms: Optional[int] = Field(None, description="Number of rooms")
    monthly_fee: Optional[int] = Field(None, description="Monthly BRF fee in SEK")
    brf_name: Optional[str] = Field(None, description="Name of the BRF")
    building_year: Optional[int] = Field(None, description="Year the building was constructed")
    floor: Optional[str] = Field(None, description="Floor number")
    debt_per_sqm: Optional[int] = Field(None, description="BRF debt per m2 in SEK")
    maintenance_fund_per_sqm: Optional[int] = Field(None, description="Maintenance fund per m2 in SEK")


def _extract_json_ld(html: str) -> Optional[dict]:
    """Extract structured data from JSON-LD script tags if present."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            # Look for RealEstateListing, Product, or similar schema types
            if isinstance(data, dict):
                schema_type = data.get("@type", "")
                if any(t in str(schema_type).lower() for t in ["realestate", "product", "residence", "apartment"]):
                    return data
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        schema_type = item.get("@type", "")
                        if any(t in str(schema_type).lower() for t in ["realestate", "product", "residence", "apartment"]):
                            return item
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _extract_from_meta_tags(html: str) -> dict:
    """Extract property data from Open Graph and other meta tags."""
    soup = BeautifulSoup(html, "html.parser")
    meta_data = {}

    for meta in soup.find_all("meta"):
        name = meta.get("property", meta.get("name", "")).lower()
        content = meta.get("content", "")
        if content:
            if "title" in name or "og:title" in name:
                meta_data["title"] = content
            elif "description" in name or "og:description" in name:
                meta_data["description"] = content
            elif "price" in name:
                meta_data["price"] = content

    # Also grab the page title
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        meta_data["page_title"] = title_tag.string.strip()

    return meta_data


def _parse_url_slug(url: str) -> dict:
    """Extract clues from the URL slug — many Swedish sites encode data in the URL."""
    hints = {}

    # Erik Olsson format: /homes/Lagenhet-4rum-Sankt-Eriksgatan-83-5-tr-Stockholm-...
    slug_match = re.search(r'/homes?/(.+?)(?:\?|$)', url)
    if not slug_match:
        slug_match = re.search(r'/objekt/(.+?)(?:\?|$)', url)
    if not slug_match:
        slug_match = re.search(r'/bostad(?:er)?/(.+?)(?:\?|$)', url)

    if slug_match:
        slug = slug_match.group(1).replace("-", " ")
        hints["url_slug"] = slug

        # Try to extract rooms
        rooms_match = re.search(r'(\d+)\s*rum', slug, re.IGNORECASE)
        if rooms_match:
            hints["rooms_from_url"] = int(rooms_match.group(1))

        # Try to extract floor
        floor_match = re.search(r'(\d+)\s*tr', slug, re.IGNORECASE)
        if floor_match:
            hints["floor_from_url"] = floor_match.group(1) + " tr"

    return hints


async def _fetch_page(url: str) -> tuple:
    """Fetch page with realistic browser headers. Returns (html, status_code)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=20.0,
        headers=headers,
    ) as client:
        response = await client.get(url)
        return response.text, response.status_code


@tool
async def parse_listing(url: str) -> dict:
    """Fetch a Swedish real estate listing URL and extract structured property data.
    Works with any Swedish agency site (ESNY, Widerlov, Notar, Hemnet, Erik Olsson, etc.).
    Uses multiple strategies: JSON-LD, meta tags, URL parsing, and LLM extraction."""

    all_context = {"url": url}

    # Step 1: Fetch the page
    try:
        html, status_code = await _fetch_page(url)
        if status_code >= 400:
            all_context["fetch_error"] = f"HTTP {status_code}"
    except Exception as e:
        html = ""
        all_context["fetch_error"] = str(e)

    # Step 2: Try JSON-LD structured data
    if html:
        json_ld = _extract_json_ld(html)
        if json_ld:
            all_context["json_ld"] = json.dumps(json_ld, ensure_ascii=False)[:3000]

    # Step 3: Extract meta tags
    if html:
        meta = _extract_from_meta_tags(html)
        if meta:
            all_context["meta_tags"] = json.dumps(meta, ensure_ascii=False)[:2000]

    # Step 4: Parse URL slug
    url_hints = _parse_url_slug(url)
    if url_hints:
        all_context["url_hints"] = json.dumps(url_hints, ensure_ascii=False)

    # Step 5: Extract text content from HTML
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg", "path"]):
            tag.decompose()
        text_content = soup.get_text(separator="\n", strip=True)[:6000]
        if len(text_content) > 100:
            all_context["page_text"] = text_content

    # Step 6: Use LLM to extract structured data from all gathered context
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extraction_prompt = f"""Extract property listing data from a Swedish real estate page.
I've gathered data from multiple sources (JSON-LD, meta tags, URL slug, page text).
Some sources may be empty if the site blocks scraping — use whatever is available.

Return a JSON object with these fields (use null if not found):
- address: street address including floor (e.g. "Sankt Eriksgatan 83, 5 tr")
- asking_price: asking price in SEK as integer (e.g. 4295000)
- size_sqm: living area in m2 as number (e.g. 67)
- rooms: number of rooms as integer (e.g. 4)
- monthly_fee: monthly BRF fee in SEK as integer (e.g. 4850)
- brf_name: name of the BRF/housing association (e.g. "Brf Sjostaden")
- building_year: construction year as integer (e.g. 1968)
- floor: floor as string (e.g. "5 tr")
- debt_per_sqm: BRF debt per m2 in SEK if available
- maintenance_fund_per_sqm: maintenance fund per m2 if available

IMPORTANT: If a field can be inferred from the URL slug (e.g. rooms, floor, address),
use that even if the page text is empty. The URL often contains reliable data.

Return ONLY valid JSON, no markdown or explanation.

Gathered data:
{json.dumps(all_context, ensure_ascii=False, indent=2)[:8000]}"""

    result = await llm.ainvoke(extraction_prompt)

    try:
        data = json.loads(result.content.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        data = {}

    # If we got almost nothing, flag it so the frontend can show manual input
    non_null_fields = sum(1 for v in data.values() if v is not None)
    if non_null_fields < 3:
        data["_parsing_incomplete"] = True
        data["_parsing_note"] = "Limited data extracted. The site may block automated access. Please verify or complete the fields manually."

    return data
