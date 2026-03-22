"""
Tool 4: Area Intelligence
Finds nearby amenities, transit, schools, and infrastructure using
OpenStreetMap Overpass API. Also checks Vindbrukskollen for wind turbines.
"""

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Optional
import json
import math
import re
import logging

logger = logging.getLogger(__name__)


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

AMENITY_QUERIES = {
    "transit": {
        "label": "Public Transit",
        "tags": [
            'node["railway"="station"]',
            'node["railway"="halt"]',
            'node["station"="subway"]',
            'node["railway"="tram_stop"]',
            'node["amenity"="bus_station"]',
            'node["highway"="bus_stop"]',
        ],
        "radius": 1500,
    },
    "school": {
        "label": "Schools & Kindergartens",
        "tags": [
            'nwr["amenity"="school"]',
            'nwr["amenity"="kindergarten"]',
        ],
        "radius": 1200,
    },
    "grocery": {
        "label": "Grocery & Shopping",
        "tags": [
            'nwr["shop"="supermarket"]',
            'nwr["shop"="convenience"]',
        ],
        "radius": 1000,
    },
    "park": {
        "label": "Parks & Green Spaces",
        "tags": [
            'nwr["leisure"="park"]',
            'nwr["leisure"="nature_reserve"]',
        ],
        "radius": 1000,
    },
    "healthcare": {
        "label": "Healthcare",
        "tags": [
            'nwr["amenity"="hospital"]',
            'nwr["amenity"="clinic"]',
            'nwr["amenity"="pharmacy"]',
        ],
        "radius": 1500,
    },
    "restaurant": {
        "label": "Restaurants & Cafés",
        "tags": [
            'nwr["amenity"="restaurant"]',
            'nwr["amenity"="cafe"]',
        ],
        "radius": 800,
    },
}


def _clean_address_for_geocoding(address: str) -> str:
    """Remove floor info, apartment numbers, and other noise from address."""
    cleaned = re.sub(r',?\s*\d+\s*(av\s*\d+\s*)?tr\b', '', address, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*\d+\s*tr\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*(nb|ög|bv)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*lgh\s*\d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s*$', '', cleaned).strip()
    return cleaned


async def _geocode_address(address: str) -> Optional[tuple]:
    """Convert a Swedish address to lat/lng using Nominatim."""
    cleaned = _clean_address_for_geocoding(address)

    queries = [
        f"{cleaned}, Sweden",
        f"{cleaned}, Stockholm, Sweden",
        f"{cleaned}, Göteborg, Sweden",
        f"{cleaned}, Malmö, Sweden",
    ]

    street_only = re.match(r'^([\w\säöåÄÖÅ]+\s+\d+)', cleaned)
    if street_only:
        queries.insert(1, f"{street_only.group(1)}, Sweden")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for query in queries:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": query,
                        "format": "json",
                        "limit": 1,
                        "countrycodes": "se",
                    },
                    headers={"User-Agent": "PropertyAnalyzer/1.0 (hackathon project)"},
                )
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", address, e)
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _element_center(el: dict) -> Optional[tuple]:
    """Get the lat/lon of an Overpass element (node, way, or relation)."""
    if "lat" in el and "lon" in el:
        return el["lat"], el["lon"]
    center = el.get("center")
    if center:
        return center["lat"], center["lon"]
    return None


async def _query_overpass(lat: float, lon: float, category: str, config: dict, client: httpx.AsyncClient) -> list:
    """Run an Overpass query for one amenity category and return findings."""
    radius = config["radius"]
    tag_union = "".join(f"{tag}(around:{radius},{lat},{lon});" for tag in config["tags"])
    query = f"[out:json][timeout:10];({tag_union});out center 20;"

    try:
        response = await client.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=15.0,
        )
        if response.status_code != 200:
            logger.warning("Overpass returned %d for category '%s'", response.status_code, category)
            return []

        elements = response.json().get("elements", [])
        results = []
        seen_names = set()

        for el in elements:
            tags = el.get("tags", {})
            name = tags.get("name", "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            center = _element_center(el)
            dist = _haversine_km(lat, lon, center[0], center[1]) if center else None

            results.append({
                "name": name,
                "distance_km": round(dist, 2) if dist is not None else None,
            })

        results.sort(key=lambda x: x["distance_km"] if x["distance_km"] is not None else 999)
        return results[:5]

    except Exception as e:
        logger.warning("Overpass query failed for '%s': %s", category, e)
        return []


async def _query_wind_turbines(lat: float, lon: float) -> list:
    """Check Vindbrukskollen for nearby wind power installations."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            delta = 0.045
            response = await client.get(
                "https://vbk.lansstyrelsen.se/api/WindPowerPlants",
                params={
                    "latMin": lat - delta,
                    "latMax": lat + delta,
                    "lonMin": lon - delta,
                    "lonMax": lon + delta,
                },
            )
            if response.status_code == 200:
                turbines = response.json()
                results = []
                for t in turbines[:5]:
                    t_lat = t.get("latitude", t.get("lat"))
                    t_lon = t.get("longitude", t.get("lon"))
                    if t_lat and t_lon:
                        dist = _haversine_km(lat, lon, float(t_lat), float(t_lon))
                        results.append({
                            "name": t.get("name", "Wind turbine"),
                            "status": t.get("status", "unknown"),
                            "height_m": t.get("totalHeight"),
                            "distance_km": round(dist, 1),
                        })
                return results
    except Exception as e:
        logger.warning("Vindbrukskollen query failed: %s", e)
    return []


@tool
async def analyze_area(address: str) -> dict:
    """Find nearby amenities, transit, schools, shops, parks, and infrastructure
    around a Swedish property address using OpenStreetMap data."""

    coords = await _geocode_address(address)
    if coords is None:
        return {
            "error": "Could not geocode address",
            "nearby": {},
            "coordinates": None,
        }

    lat, lon = coords
    nearby = {}

    async with httpx.AsyncClient() as client:
        for category, config in AMENITY_QUERIES.items():
            items = await _query_overpass(lat, lon, category, config, client)
            nearby[category] = {
                "label": config["label"],
                "items": items,
                "count": len(items),
                "search_radius_m": config["radius"],
            }

    wind = await _query_wind_turbines(lat, lon)
    if wind:
        nearby["wind_power"] = {
            "label": "Wind Power Installations",
            "items": wind,
            "count": len(wind),
            "search_radius_m": 5000,
        }

    # LLM summary of the neighbourhood
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    summary_prompt = f"""Based on this nearby-amenities data for a Swedish property at {address},
write a concise neighbourhood summary (3-5 sentences) highlighting:
- Transit accessibility
- Family friendliness (schools, parks)
- Daily convenience (grocery, healthcare)
- Any notable positives or negatives

Data:
{json.dumps(nearby, indent=2, default=str)[:4000]}

Return ONLY a JSON object with:
- "summary": string (the neighbourhood summary)
- "walkability_score": int 1-10 (10 = everything within walking distance)
- "family_score": int 1-10 (10 = ideal for families)

Return ONLY valid JSON."""

    try:
        result = await llm.ainvoke(summary_prompt)
        assessment = json.loads(result.content.strip().strip("```json").strip("```"))
    except Exception:
        assessment = {
            "summary": "Area data collected. Review categories for details.",
            "walkability_score": None,
            "family_score": None,
        }

    return {
        "coordinates": {"lat": lat, "lon": lon},
        "nearby": nearby,
        "assessment": assessment,
    }
