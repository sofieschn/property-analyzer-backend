"""
Tool 4: Area Intelligence
Checks for nearby developments, wind turbines, and infrastructure projects
using Vindbrukskollen and Stockholm Open Data.
"""

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Optional
import json
import math


async def _geocode_address(address: str) -> Optional[tuple]:
    """Convert a Swedish address to lat/lng using a free geocoding service."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": f"{address}, Stockholm, Sweden",
                    "format": "json",
                    "limit": 1,
                },
                headers={"User-Agent": "PropertyAnalyzer/1.0"},
            )
            if response.status_code == 200:
                data = response.json()
                if data:
                    return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


@tool
async def analyze_area(address: str) -> dict:
    """Find nearby planned developments, wind turbines, and infrastructure
    projects around a Swedish property address."""

    coords = await _geocode_address(address)
    if coords is None:
        return {
            "error": "Could not geocode address",
            "findings": [],
            "coordinates": None,
        }

    lat, lon = coords
    findings = []

    # --- Vindbrukskollen: wind power installations ---
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Search within ~5km bounding box
            delta = 0.045  # roughly 5km
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
                for t in turbines[:5]:  # Limit to 5 nearest
                    t_lat = t.get("latitude", t.get("lat"))
                    t_lon = t.get("longitude", t.get("lon"))
                    if t_lat and t_lon:
                        dist = _haversine_km(lat, lon, float(t_lat), float(t_lon))
                        findings.append({
                            "type": "WIND_POWER",
                            "title": t.get("name", "Wind turbine"),
                            "description": f"Status: {t.get('status', 'unknown')}. "
                                          f"Height: {t.get('totalHeight', 'N/A')}m.",
                            "distance_km": round(dist, 1),
                            "impact": "neutral",
                        })
    except Exception:
        pass

    # --- Stockholm Open Data: detaljplaner (zoning plans) ---
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://opendata.stockholm.se/api/detaljplaner",
                params={
                    "lat": lat,
                    "lng": lon,
                    "radius": 1000,  # meters
                    "format": "json",
                },
            )
            if response.status_code == 200:
                plans = response.json()
                if isinstance(plans, list):
                    for plan in plans[:5]:
                        findings.append({
                            "type": "ZONING_PLAN",
                            "title": plan.get("plannamn", plan.get("name", "Detaljplan")),
                            "description": plan.get("beskrivning", plan.get("description", "")),
                            "distance_km": round(plan.get("distance", 0) / 1000, 1),
                            "impact": "neutral",
                        })
    except Exception:
        pass

    # --- Stockholm Open Data: building permits ---
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://opendata.stockholm.se/api/bygglov",
                params={
                    "lat": lat,
                    "lng": lon,
                    "radius": 1000,
                    "format": "json",
                },
            )
            if response.status_code == 200:
                permits = response.json()
                if isinstance(permits, list):
                    for permit in permits[:5]:
                        findings.append({
                            "type": "BUILDING_PERMIT",
                            "title": permit.get("description", "Building permit"),
                            "description": permit.get("details", ""),
                            "distance_km": 0,
                            "impact": "neutral",
                        })
    except Exception:
        pass

    # If we found items, use LLM to assess impact
    if findings:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        impact_prompt = f"""Assess the impact of each nearby development on a residential property buyer.
For each item, set impact to "positive", "negative", or "neutral".

Positive examples: new metro station, school, park, improved infrastructure
Negative examples: large wind farm very close, heavy construction next door, highway
Neutral examples: distant wind turbines, minor zoning changes

Items:
{json.dumps(findings, indent=2)}

Return ONLY a JSON array with the same items but with the "impact" field updated. No explanation."""

        result = await llm.ainvoke(impact_prompt)
        try:
            assessed = json.loads(result.content.strip().strip("```json").strip("```"))
            if isinstance(assessed, list) and len(assessed) == len(findings):
                findings = assessed
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "coordinates": {"lat": lat, "lon": lon},
        "findings": findings,
        "search_radius_km": 5,
    }
