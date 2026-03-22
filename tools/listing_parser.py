"""
Tool 1: Listing Parser
Fetches a Swedish real estate listing URL and uses an LLM to extract
structured property data from the HTML and linked PDF documents.

Strategy:
1. Fetch the listing page HTML
2. Extract JSON-LD, meta tags, and page text
3. Find and download linked PDFs (objektsbeskrivning, BRF docs etc.)
4. Extract text from PDFs — this is where fee, debt, fund data usually lives
5. Pass all gathered context to LLM for structured extraction
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
import json
import re
import io

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sv-SE,sv;q=0.9,en-US;q=0.8,en;q=0.7",
}

# Keywords that indicate a useful PDF (objektsbeskrivning, BRF docs, etc.)
PDF_KEYWORDS = [
    "objekts", "prospekt", "beskrivning", "brf", "förening", "ekonomi",
    "årsredovisning", "annual", "document", "doc", "pdf", "bilag",
]


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
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
            for item in items:
                if isinstance(item, dict):
                    schema_type = str(item.get("@type", "")).lower()
                    if any(t in schema_type for t in ["realestate", "product", "residence", "apartment"]):
                        return item
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _extract_meta(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    meta = {}
    for tag in soup.find_all("meta"):
        name = tag.get("property", tag.get("name", "")).lower()
        content = tag.get("content", "")
        if content:
            if "title" in name:
                meta["title"] = content
            elif "description" in name:
                meta["description"] = content
            elif "price" in name:
                meta["price"] = content
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        meta["page_title"] = title_tag.string.strip()
    return meta


def _find_pdf_links(html: str, base_url: str) -> list:
    """Find all PDF links on the page, prioritizing property/BRF documents."""
    soup = BeautifulSoup(html, "html.parser")
    pdf_links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        link_text = a.get_text(strip=True).lower()
        aria_label = a.get("aria-label", "").lower()
        combined_text = f"{link_text} {aria_label} {href}".lower()

        # Only consider PDF links
        if not (href.endswith(".pdf") or "pdf" in href.lower() or ".pdf" in combined_text):
            continue

        # Make absolute URL
        if href.startswith("http"):
            full_url = href
        elif href.startswith("//"):
            full_url = "https:" + href
        elif href.startswith("/"):
            # Extract base domain from base_url
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
        else:
            full_url = base_url.rstrip("/") + "/" + href

        # Score by relevance
        score = sum(1 for kw in PDF_KEYWORDS if kw in combined_text)
        pdf_links.append((score, full_url, link_text))

    # Sort by relevance score, take top 3
    pdf_links.sort(reverse=True)
    return [url for _, url, _ in pdf_links[:3]]


async def _extract_pdf_text(url: str, client: httpx.AsyncClient) -> Optional[str]:
    """Download a PDF and extract its text content."""
    if PdfReader is None:
        return None
    try:
        response = await client.get(url, timeout=20.0, headers=BROWSER_HEADERS)
        if response.status_code != 200:
            return None
        if "pdf" not in response.headers.get("content-type", "").lower() and not url.lower().endswith(".pdf"):
            return None

        pdf_bytes = io.BytesIO(response.content)
        reader = PdfReader(pdf_bytes)

        text_parts = []
        for page in reader.pages[:8]:  # Max 8 pages to limit token usage
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception:
                continue

        full_text = "\n".join(text_parts)
        # Return first 5000 chars — enough to find fee/debt/fund data
        return full_text[:5000] if full_text else None
    except Exception:
        return None


def _parse_url_slug(url: str) -> dict:
    hints = {}
    for pattern in [r'/homes?/(.+?)(?:\?|$)', r'/objekt/(.+?)(?:\?|$)', r'/bostad(?:er)?/(.+?)(?:\?|$)']:
        slug_match = re.search(pattern, url)
        if slug_match:
            slug = slug_match.group(1).replace("-", " ")
            hints["url_slug"] = slug
            rooms_match = re.search(r'(\d+)\s*rum', slug, re.IGNORECASE)
            if rooms_match:
                hints["rooms_from_url"] = int(rooms_match.group(1))
            floor_match = re.search(r'(\d+)\s*tr', slug, re.IGNORECASE)
            if floor_match:
                hints["floor_from_url"] = floor_match.group(1) + " tr"
            break
    return hints


@tool
async def parse_listing(url: str) -> dict:
    """Fetch a Swedish real estate listing URL and extract structured property data.
    Follows PDF links (objektsbeskrivning, BRF docs) to extract monthly fee,
    debt/m2, maintenance fund, and other BRF financial details."""

    all_context = {"url": url}
    html = ""

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=20.0,
        headers=BROWSER_HEADERS,
    ) as client:
        # Step 1: Fetch the listing page
        try:
            response = await client.get(url)
            if response.status_code < 400:
                html = response.text
        except Exception as e:
            all_context["fetch_error"] = str(e)

        # Step 2: Extract structured data from HTML
        if html:
            json_ld = _extract_json_ld(html)
            if json_ld:
                all_context["json_ld"] = json.dumps(json_ld, ensure_ascii=False)[:2000]

            meta = _extract_meta(html)
            if meta:
                all_context["meta_tags"] = json.dumps(meta, ensure_ascii=False)[:1000]

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg"]):
                tag.decompose()
            page_text = soup.get_text(separator="\n", strip=True)[:4000]
            if len(page_text) > 100:
                all_context["page_text"] = page_text

        # Step 3: Parse URL slug for clues
        url_hints = _parse_url_slug(url)
        if url_hints:
            all_context["url_hints"] = json.dumps(url_hints, ensure_ascii=False)

        # Step 4: Find and extract PDF documents
        if html:
            pdf_urls = _find_pdf_links(html, url)
            pdf_texts = []
            for pdf_url in pdf_urls:
                pdf_text = await _extract_pdf_text(pdf_url, client)
                if pdf_text:
                    pdf_texts.append({
                        "source": pdf_url,
                        "text": pdf_text,
                    })

            if pdf_texts:
                # Combine PDF texts, limit total size
                combined = "\n\n---\n\n".join(
                    f"PDF: {p['source']}\n{p['text']}" for p in pdf_texts
                )
                all_context["pdf_documents"] = combined[:6000]

    # Step 5: LLM extraction from all gathered context
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extraction_prompt = f"""Extract property listing data from a Swedish real estate listing.
I've gathered data from multiple sources: JSON-LD metadata, meta tags, page text, URL slug,
and most importantly — linked PDF documents which often contain the full BRF financial details.

Look especially in the PDF documents for:
- Monthly fee (avgift, månadsavgift): often listed as "X kr/mån"
- BRF debt per m2 (skuld per kvm, belåning per kvm)
- Maintenance fund per m2 (underhållsfond per kvm, fond per kvm)
- Building year (byggår, byggnadsår)
- BRF name (förening, bostadsrättsförening)

Return a JSON object with these fields. Use null if a value is genuinely not found — do NOT use 0 as a default.

CRITICAL: 0 and null mean very different things for a home buyer:
- null = data not available, buyer should look it up
- 0 = the BRF has zero debt or zero maintenance fund (very unusual, only use if explicitly stated)

Only return 0 for debt_per_sqm or maintenance_fund_per_sqm if the source text explicitly says "0 kr", "ingen skuld", "skuldfri" or similar. Otherwise return null.

Fields:
- address: street address including floor (e.g. "Sankt Eriksgatan 83, 5 tr")
- asking_price: asking price in SEK as integer (e.g. 4295000)
- size_sqm: living area in m2 as number (e.g. 67)
- rooms: number of rooms as integer (e.g. 4)
- monthly_fee: monthly BRF fee in SEK as integer (e.g. 4850)
- brf_name: name of the BRF (e.g. "Brf Majsol")
- building_year: construction year as integer (e.g. 1968)
- floor: floor as string (e.g. "5 tr")
- debt_per_sqm: BRF debt per m2 in SEK — null if not explicitly stated in source
- maintenance_fund_per_sqm: maintenance fund per m2 in SEK — null if not explicitly stated in source

Return ONLY valid JSON, no markdown or explanation.

Gathered data:
{json.dumps(all_context, ensure_ascii=False, indent=2)[:9000]}"""

    result = await llm.ainvoke(extraction_prompt)

    try:
        data = json.loads(result.content.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        data = {}

    non_null = sum(1 for v in data.values() if v is not None)
    if non_null < 3:
        data["_parsing_incomplete"] = True
        data["_parsing_note"] = "Limited data could be extracted. The site may block automated access. Please verify fields manually."

    return data
