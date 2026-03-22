"""
Tool 1: Listing Parser
Fetches a Swedish real estate listing URL and uses an LLM to extract
structured property data from the HTML — works across any agency site.
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional


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


@tool
async def parse_listing(url: str) -> dict:
    """Fetch a Swedish real estate listing URL and extract structured property data.
    Works with any Swedish agency site (ESNY, Widerlov, Notar, Hemnet, etc.)."""

    # Fetch the page
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PropertyAnalyzer/1.0)"}
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    # Clean HTML to reduce token usage
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
        tag.decompose()

    # Keep only text-heavy content, truncate to ~8000 chars
    text_content = soup.get_text(separator="\n", strip=True)[:8000]

    # Use LLM to extract structured data
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    extraction_prompt = f"""Extract property listing data from this Swedish real estate page text.
Return a JSON object with these fields (use null if not found):
- address: street address including floor (e.g. "Birger Jarlsgatan 42, 3 tr")
- asking_price: asking price in SEK as integer (e.g. 4295000)
- size_sqm: living area in m2 as number (e.g. 67)
- rooms: number of rooms as integer (e.g. 2)
- monthly_fee: monthly BRF fee in SEK as integer (e.g. 4850)
- brf_name: name of the BRF/housing association (e.g. "Brf Sjostaden")
- building_year: construction year as integer (e.g. 1968)
- floor: floor as string (e.g. "3 tr")
- debt_per_sqm: BRF debt per m2 in SEK if available
- maintenance_fund_per_sqm: maintenance fund per m2 if available

Return ONLY valid JSON, no markdown or explanation.

Page text:
{text_content}"""

    result = await llm.ainvoke(extraction_prompt)

    import json
    try:
        data = json.loads(result.content.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        data = {}

    return data
