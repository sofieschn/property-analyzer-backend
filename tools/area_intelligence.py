"""
Tool 4: Area Intelligence
Finds nearby amenities, transit, schools, and infrastructure using
OpenStreetMap Overpass API (single batched query).
"""

import asyncio
import httpx
from langchain_core.tools import tool
from typing import Optional
import json
import math
import re
import logging

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

AMENITY_CATEGORIES = {
    "transit": {
        "label": "Public Transit",
        "tags": [
            ("node", "railway", "station"),
            ("node", "railway", "halt"),
            ("node", "station", "subway"),
            ("node", "railway", "tram_stop"),
            ("node", "amenity", "bus_station"),
            ("node", "highway", "bus_stop"),
        ],
        "radius": 1500,
    },
    "school": {
        "label": "Schools & Kindergartens",
        "tags": [
            ("nwr", "amenity", "school"),
            ("nwr", "amenity", "kindergarten"),
        ],
        "radius": 1200,
    },
    "grocery": {
        "label": "Grocery & Shopping",
        "tags": [
            ("nwr", "shop", "supermarket"),
            ("nwr", "shop", "convenience"),
        ],
        "radius": 1000,
    },
    "park": {
        "label": "Parks & Green Spaces",
        "tags": [
            ("nwr", "leisure", "park"),
            ("nwr", "leisure", "nature_reserve"),
        ],
        "radius": 1000,
    },
    "healthcare": {
        "label": "Healthcare",
        "tags": [
            ("nwr", "amenity", "hospital"),
            ("nwr", "amenity", "clinic"),
            ("nwr", "amenity", "pharmacy"),
            ("nwr", "amenity", "doctors"),
        ],
        "radius": 1500,
    },
    "restaurant": {
        "label": "Restaurants & Cafés",
        "tags": [
            ("nwr", "amenity", "restaurant"),
            ("nwr", "amenity", "cafe"),
        ],
        "radius": 800,
    },
}

_TYPE_LABELS = {
    "supermarket": "Supermarket", "convenience": "Convenience store",
    "park": "Park", "nature_reserve": "Nature reserve",
    "hospital": "Hospital", "clinic": "Clinic", "pharmacy": "Pharmacy",
    "doctors": "Doctor's office", "school": "School", "kindergarten": "Kindergarten",
    "restaurant": "Restaurant", "cafe": "Café",
    "station": "Station", "halt": "Train stop", "subway": "Subway station",
    "tram_stop": "Tram stop", "bus_station": "Bus station", "bus_stop": "Bus stop",
}


def _clean_address_for_geocoding(address: str) -> str:
    cleaned = re.sub(r',?\s*\d+\s*(av\s*\d+\s*)?tr\b', '', address, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*\d+\s*tr\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*(nb|ög|bv)\b', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r',?\s*lgh\s*\d+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*,\s*$', '', cleaned).strip()
    return cleaned


async def _geocode_address(address: str) -> Optional[tuple]:
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
        async with httpx.AsyncClient(timeout=5.0) as client:
            for query in queries:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": 1, "countrycodes": "se"},
                    headers={"User-Agent": "PropertyAnalyzer/1.0"},
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
    if "lat" in el and "lon" in el:
        return el["lat"], el["lon"]
    center = el.get("center")
    if center:
        return center["lat"], center["lon"]
    bounds = el.get("bounds")
    if bounds:
        return (bounds["minlat"] + bounds["maxlat"]) / 2, (bounds["minlon"] + bounds["maxlon"]) / 2
    return None


def _get_display_name(tags: dict) -> str:
    name = tags.get("name")
    if name:
        return name
    for key in ("shop", "amenity", "leisure", "healthcare", "railway", "highway", "station"):
        val = tags.get(key)
        if val and val in _TYPE_LABELS:
            return _TYPE_LABELS[val]
    return ""


def _tag_matches_category(tags: dict, category_tags: list) -> bool:
    for _, key, value in category_tags:
        if tags.get(key) == value:
            return True
    return False


def _build_overpass_query(lat: float, lon: float) -> str:
    """Single Overpass query for all categories — avoids rate limiting."""
    parts = []
    for config in AMENITY_CATEGORIES.values():
        radius = config["radius"]
        for elem_type, key, value in config["tags"]:
            parts.append(f'{elem_type}["{key}"="{value}"](around:{radius},{lat},{lon});')
    return f"[out:json][timeout:20];({' '.join(parts)});out center body 80;"


async def _query_all_amenities(lat: float, lon: float) -> dict:
    """Single batched Overpass request, results sorted into categories."""
    query = _build_overpass_query(lat, lon)
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(OVERPASS_URL, data={"data": query})
            if response.status_code != 200:
                logger.warning("Overpass returned %d", response.status_code)
                return {}
            elements = response.json().get("elements", [])
    except Exception as e:
        logger.warning("Overpass query failed: %s", e)
        return {}

    results = {}
    for cat_key, config in AMENITY_CATEGORIES.items():
        cat_items = []
        seen = set()
        for el in elements:
            tags = el.get("tags", {})
            if not _tag_matches_category(tags, config["tags"]):
                continue
            display_name = _get_display_name(tags)
            if not display_name:
                continue
            dedup = display_name.lower()
            if dedup in seen:
                continue
            seen.add(dedup)
            center = _element_center(el)
            dist = _haversine_km(lat, lon, center[0], center[1]) if center else None
            if dist is not None and dist > config["radius"] / 1000:
                continue
            cat_items.append({"name": display_name, "distance_km": round(dist, 2) if dist else None})

        cat_items.sort(key=lambda x: x["distance_km"] if x["distance_km"] is not None else 999)
        results[cat_key] = {
            "label": config["label"],
            "items": cat_items[:5],
            "count": len(cat_items),
            "search_radius_m": config["radius"],
        }
    return results


async def _query_wind_turbines(lat: float, lon: float) -> list:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            delta = 0.045
            response = await client.get(
                "https://vbk.lansstyrelsen.se/api/WindPowerPlants",
                params={"latMin": lat - delta, "latMax": lat + delta,
                        "lonMin": lon - delta, "lonMax": lon + delta},
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
                            "distance_km": round(dist, 1),
                        })
                return results
    except Exception as e:
        logger.warning("Vindbrukskollen failed: %s", e)
    return []


@tool
async def analyze_area(address: str) -> dict:
    """Find nearby amenities, transit, schools, shops, parks, and infrastructure
    around a Swedish property address using OpenStreetMap data."""

    coords = await _geocode_address(address)
    if coords is None:
        return {"error": "Could not geocode address", "nearby": {}, "coordinates": None}

    lat, lon = coords

    # Run Overpass + wind turbines in parallel
    nearby, wind = await asyncio.gather(
        _query_all_amenities(lat, lon),
        _query_wind_turbines(lat, lon),
    )

    if wind:
        nearby["wind_power"] = {
            "label": "Wind Power Installations",
            "items": wind,
            "count": len(wind),
            "search_radius_m": 5000,
        }

    return {
        "coordinates": {"lat": lat, "lon": lon},
        "nearby": nearby,
    }
