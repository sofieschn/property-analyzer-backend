"""
Tool 3: Sustainability
Checks energy performance against MEPS thresholds.
Attempts to fetch real data from Boverket's public search, falls back to
estimation based on building year if unavailable.
"""

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import Optional
import json
import re


# Swedish MEPS thresholds (kWh/m2/year) for residential buildings
MEPS_THRESHOLDS = {
    "residential": {
        2030: 160,
        2033: 130,
    },
    "commercial": {
        2030: 204,
        2033: 167,
    }
}

# Approximate renovation cost ranges per energy class step (SEK)
RENOVATION_COSTS = {
    "G_to_F": (100_000, 200_000),
    "F_to_E": (120_000, 250_000),
    "E_to_D": (150_000, 300_000),
    "D_to_C": (250_000, 450_000),
    "C_to_B": (350_000, 600_000),
    "B_to_A": (500_000, 900_000),
}

ENERGY_CLASS_RANGES = {
    "A": (0, 50),
    "B": (50, 75),
    "C": (75, 100),
    "D": (100, 135),
    "E": (135, 180),
    "F": (180, 235),
    "G": (235, 999),
}


def _get_energy_class(kwh_per_sqm: float) -> str:
    for cls, (low, high) in ENERGY_CLASS_RANGES.items():
        if low <= kwh_per_sqm < high:
            return cls
    return "G"


def _estimate_renovation_cost(current_class: str, target_class: str) -> Optional[tuple]:
    classes = list(ENERGY_CLASS_RANGES.keys())
    current_idx = classes.index(current_class)
    target_idx = classes.index(target_class)

    if target_idx >= current_idx:
        return None

    total_low = 0
    total_high = 0
    for i in range(current_idx, target_idx, -1):
        key = f"{classes[i]}_to_{classes[i-1]}"
        if key in RENOVATION_COSTS:
            low, high = RENOVATION_COSTS[key]
            total_low += low
            total_high += high

    return (total_low, total_high) if total_low > 0 else None


async def _try_boverket_search(address: str) -> Optional[dict]:
    """Try to get energy declaration data from Boverket's public search page.
    This scrapes the public-facing search at sokenergideklaration.boverket.se."""
    try:
        # Clean address for search
        clean_addr = re.sub(r',?\s*\d+\s*(av\s*\d+\s*)?tr\b', '', address, flags=re.IGNORECASE).strip()

        async with httpx.AsyncClient(timeout=3.0, follow_redirects=True) as client:
            # Try the public search endpoint
            response = await client.get(
                "https://sokenergideklaration.boverket.se/api/search",
                params={"query": clean_addr},
                headers={"User-Agent": "Mozilla/5.0 (compatible; PropertyAnalyzer/1.0)"},
            )
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    decl = data[0]
                    return {
                        "energy_class": decl.get("energiklass", decl.get("energyClass")),
                        "primary_energy": decl.get("primaerenergital", decl.get("primaryEnergy")),
                        "heating_type": decl.get("uppvaermningssaett", decl.get("heatingType")),
                    }
    except Exception:
        pass

    # Fallback: try the older API endpoint
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                "https://api.boverket.se/energideklarationer/",
                params={"kommun": "Stockholm", "adress": address},
            )
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    decl = data[0]
                    return {
                        "energy_class": decl.get("energiklass"),
                        "primary_energy": decl.get("primaerenergital"),
                        "heating_type": decl.get("uppvaermningssaett"),
                    }
    except Exception:
        pass

    return None


def _estimate_from_building_year(building_year: int) -> dict:
    """Estimate energy performance from building year using Swedish averages."""
    # Based on Swedish building stock averages by era
    estimates = [
        (1930, 195, "District heating (estimated)", "Pre-war buildings typically have poor insulation, high ceilings, and single-pane windows."),
        (1945, 180, "District heating (estimated)", "Early 20th century buildings, often with some renovations but original structure."),
        (1960, 170, "District heating (estimated)", "Post-war construction, better standards but before energy crisis awareness."),
        (1975, 155, "District heating (estimated)", "Miljonprogrammet era, mass-produced with moderate insulation."),
        (1985, 135, "District heating (estimated)", "Built after 1975 energy crisis, improved insulation standards."),
        (2000, 115, "District heating/heat pump (estimated)", "Modern building codes, decent energy performance."),
        (2012, 90, "Heat pump (estimated)", "BBR energy requirements tightened significantly."),
        (2020, 75, "Heat pump (estimated)", "Near-zero energy building standards."),
        (9999, 65, "Heat pump (estimated)", "Latest building codes, excellent energy performance."),
    ]

    for threshold_year, kwh, heating, note in estimates:
        if building_year < threshold_year:
            return {
                "energy_class": _get_energy_class(kwh),
                "primary_energy": kwh,
                "heating_type": heating,
                "is_estimated": True,
                "estimation_note": note,
            }

    return {
        "energy_class": _get_energy_class(65),
        "primary_energy": 65,
        "heating_type": "Heat pump (estimated)",
        "is_estimated": True,
        "estimation_note": "New construction, assumed excellent performance.",
    }


@tool
async def analyze_sustainability(
    address: str,
    building_year: Optional[int] = None,
    building_type: str = "residential",
) -> dict:
    """Check energy performance against Swedish MEPS thresholds.
    Fetches energy declaration from Boverket and calculates compliance risk."""

    # Try to get real data first
    energy_data = await _try_boverket_search(address)

    # Fall back to estimation if no real data
    if energy_data is None and building_year is not None:
        energy_data = _estimate_from_building_year(building_year)
    elif energy_data is None:
        return {
            "error": "Could not determine energy performance. No API data and no building year provided.",
            "energy_class": None,
            "primary_energy": None,
            "meps_status": {},
            "renovation_estimate": None,
            "risks": [],
        }

    # Check MEPS compliance
    thresholds = MEPS_THRESHOLDS.get(building_type, MEPS_THRESHOLDS["residential"])
    primary_energy = energy_data["primary_energy"]
    energy_class = energy_data["energy_class"]

    meps_status = {}
    risks = []

    for year, threshold in thresholds.items():
        compliant = primary_energy <= threshold
        meps_status[str(year)] = {
            "threshold": threshold,
            "current": primary_energy,
            "compliant": compliant,
            "gap": max(0, primary_energy - threshold),
        }
        if not compliant:
            risks.append({
                "flag": f"MEPS_{year}_FAIL",
                "message": f"At {primary_energy} kWh/m²/year, this building exceeds the {year} MEPS threshold "
                           f"of {threshold} kWh/m² by {primary_energy - threshold} kWh/m². "
                           "Renovation will be required to meet regulatory standards.",
                "severity": "high" if year == 2030 else "medium",
            })

    # Estimate renovation cost to reach 2033 compliance
    target_kwh = thresholds[2033]
    target_class = _get_energy_class(target_kwh)
    renovation_estimate = _estimate_renovation_cost(energy_class, target_class)

    return {
        "energy_class": energy_class,
        "primary_energy": primary_energy,
        "heating_type": energy_data.get("heating_type"),
        "is_estimated": energy_data.get("is_estimated", False),
        "estimation_note": energy_data.get("estimation_note"),
        "meps_status": meps_status,
        "renovation_estimate": {
            "low": renovation_estimate[0],
            "high": renovation_estimate[1],
            "target_class": target_class,
        } if renovation_estimate else None,
        "risks": risks,
    }
