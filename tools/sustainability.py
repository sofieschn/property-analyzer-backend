"""
Tool 3: Sustainability
Checks energy performance against MEPS thresholds using Boverket's
Energideklaration API. Estimates renovation costs to reach compliance.
"""

import httpx
from langchain_core.tools import tool
from typing import Optional


# Swedish MEPS thresholds (kWh/m2/year)
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

# Approximate renovation cost ranges per energy class improvement (SEK)
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
        return None  # Already at or above target

    total_low = 0
    total_high = 0
    for i in range(current_idx, target_idx, -1):
        key = f"{classes[i]}_to_{classes[i-1]}"
        if key in RENOVATION_COSTS:
            low, high = RENOVATION_COSTS[key]
            total_low += low
            total_high += high

    return (total_low, total_high) if total_low > 0 else None


@tool
async def analyze_sustainability(
    address: str,
    building_year: Optional[int] = None,
    building_type: str = "residential",
) -> dict:
    """Check energy performance against Swedish MEPS thresholds.
    Fetches energy declaration from Boverket and calculates compliance risk."""

    energy_data = None

    # Attempt to fetch from Boverket's open energy declaration data
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Boverket energy declaration search
            response = await client.get(
                "https://www.boverket.se/api/energideklaration/sok",
                params={"adress": address, "format": "json"},
            )
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    decl = data[0]
                    energy_data = {
                        "energy_class": decl.get("energiklass"),
                        "primary_energy": decl.get("primaerenergital"),
                        "heating_type": decl.get("uppvaermningssaett"),
                    }
    except Exception:
        pass  # Fall back to estimation

    # If API fails, estimate from building year
    if energy_data is None and building_year is not None:
        if building_year < 1960:
            estimated_kwh = 185
        elif building_year < 1975:
            estimated_kwh = 165
        elif building_year < 1990:
            estimated_kwh = 140
        elif building_year < 2005:
            estimated_kwh = 120
        elif building_year < 2015:
            estimated_kwh = 95
        else:
            estimated_kwh = 75

        energy_data = {
            "energy_class": _get_energy_class(estimated_kwh),
            "primary_energy": estimated_kwh,
            "heating_type": "estimated from building year",
            "is_estimated": True,
        }
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
                "message": f"At {primary_energy} kWh/m2/year, this building exceeds the {year} MEPS threshold "
                           f"of {threshold} kWh/m2 by {primary_energy - threshold} kWh/m2. "
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
        "meps_status": meps_status,
        "renovation_estimate": {
            "low": renovation_estimate[0],
            "high": renovation_estimate[1],
            "target_class": target_class,
        } if renovation_estimate else None,
        "risks": risks,
    }
