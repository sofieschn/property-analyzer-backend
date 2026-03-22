"""
Tool 2: BRF Economy
Analyzes the financial health of a BRF using listing data and public records.
Calculates total monthly cost, flags risks like high debt or upcoming stambyte.
"""

from langchain_core.tools import tool
from typing import Optional


@tool
async def analyze_brf_economy(
    asking_price: int,
    monthly_fee: int,
    size_sqm: float,
    building_year: Optional[int] = None,
    debt_per_sqm: Optional[int] = None,
    maintenance_fund_per_sqm: Optional[int] = None,
    stambyte_done_year: Optional[int] = None,
    stambyte_planned_year: Optional[int] = None,
) -> dict:
    """Analyze the BRF's financial health based on listing data.
    Calculates total monthly cost and flags economic risks.
    Uses real stambyte data from the listing when available,
    falls back to age-based estimation only when no data exists."""

    # --- Mortgage calculation ---
    loan_amount = asking_price * 0.85
    monthly_interest_rate = 0.042 / 12
    monthly_mortgage_interest = loan_amount * monthly_interest_rate
    annual_amortization_rate = 0.02
    monthly_amortization = (loan_amount * annual_amortization_rate) / 12
    total_monthly_cost = monthly_fee + monthly_mortgage_interest + monthly_amortization

    # --- Risk assessment ---
    risks = []
    risk_level = "LOW"

    # Debt per m2 assessment
    if debt_per_sqm is None:
        risks.append({
            "flag": "DEBT_DATA_MISSING",
            "message": "BRF debt per m² could not be found in the listing. "
                       "Request the BRF annual report (årsredovisning) to verify the association's debt level "
                       "before making an offer. High debt can lead to significant fee increases.",
            "severity": "medium"
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"
    elif debt_per_sqm > 10000:
        risks.append({
            "flag": "HIGH_DEBT",
            "message": f"BRF debt is {debt_per_sqm} SEK/m², well above the 10,000 SEK/m² warning threshold. "
                       "This indicates the association carries significant loans that may lead to fee increases.",
            "severity": "high"
        })
        risk_level = "HIGH"
    elif debt_per_sqm > 7000:
        risks.append({
            "flag": "ELEVATED_DEBT",
            "message": f"BRF debt at {debt_per_sqm} SEK/m² is moderately high. Monitor for potential fee increases.",
            "severity": "medium"
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"

    # Maintenance fund assessment
    if maintenance_fund_per_sqm is None:
        risks.append({
            "flag": "FUND_DATA_MISSING",
            "message": "Maintenance fund per m² could not be found in the listing. "
                       "A low or empty maintenance fund means the BRF may struggle to cover future repairs "
                       "without special assessments (extra uttaxering). Request the årsredovisning to verify.",
            "severity": "medium"
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"
    elif maintenance_fund_per_sqm < 1000:
        risks.append({
            "flag": "LOW_FUND",
            "message": f"Maintenance fund at {maintenance_fund_per_sqm} SEK/m² is below recommended levels. "
                       "Large repairs may require special assessments or fee increases.",
            "severity": "medium"
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"

    # --- Stambyte assessment ---
    # Priority: use real data from listing; only estimate from building age as last resort
    stambyte_risk = False

    if stambyte_planned_year is not None:
        years_until = stambyte_planned_year - 2026
        if years_until <= 0:
            stambyte_risk = True
            risks.append({
                "flag": "STAMBYTE_ONGOING",
                "message": f"A stambyte is planned/underway (scheduled for {stambyte_planned_year}). "
                           "Expect increased monthly fees during the renovation period, typically "
                           "+500–1,000 SEK/month for 10–15 years. Cost: 5,000–8,000 SEK/m².",
                "severity": "high"
            })
            risk_level = "HIGH"
        elif years_until <= 5:
            stambyte_risk = True
            risks.append({
                "flag": "STAMBYTE_PLANNED",
                "message": f"A stambyte is planned for {stambyte_planned_year} ({years_until} year(s) away). "
                           "Budget for increased monthly fees. Cost: typically 5,000–8,000 SEK/m², "
                           "often financed through higher BRF fees (+500–1,000 SEK/month for 10–15 years).",
                "severity": "high"
            })
            risk_level = "HIGH"
        else:
            risks.append({
                "flag": "STAMBYTE_FUTURE",
                "message": f"A stambyte is planned for {stambyte_planned_year} ({years_until} years away). "
                           "Not an immediate concern but factor into long-term costs.",
                "severity": "low"
            })

    elif stambyte_done_year is not None:
        years_since = 2026 - stambyte_done_year
        if years_since < 10:
            risks.append({
                "flag": "STAMBYTE_RECENT",
                "message": f"Stambyte was completed in {stambyte_done_year} ({years_since} years ago). "
                           "Pipes are in good condition. If the BRF took a loan for the renovation, "
                           "check whether the monthly fee still includes loan repayments.",
                "severity": "low"
            })
        elif years_since < 40:
            risks.append({
                "flag": "STAMBYTE_OK",
                "message": f"Stambyte was completed in {stambyte_done_year} ({years_since} years ago). "
                           "Pipes should be in acceptable condition for another 20–30 years.",
                "severity": "low"
            })
        else:
            stambyte_risk = True
            risks.append({
                "flag": "STAMBYTE_AGING",
                "message": f"The last stambyte was in {stambyte_done_year} ({years_since} years ago). "
                           "Pipes typically last 50–60 years, so another renovation may be needed "
                           "within the next 10–20 years.",
                "severity": "medium"
            })
            if risk_level == "LOW":
                risk_level = "MEDIUM"

    elif building_year is not None:
        # No stambyte data from listing — estimate from building age
        building_age = 2026 - building_year
        if building_age > 50:
            stambyte_risk = True
            risks.append({
                "flag": "STAMBYTE_UNKNOWN",
                "message": f"Building is from {building_year} ({building_age} years old) and no stambyte "
                           "information was found in the listing. Pipes typically need replacement every "
                           "50–60 years. Request the BRF annual report (årsredovisning) to check whether "
                           "a stambyte has been done or is planned. Cost if needed: 5,000–8,000 SEK/m².",
                "severity": "high"
            })
            risk_level = "HIGH"
        elif building_age >= 40:
            stambyte_risk = True
            risks.append({
                "flag": "STAMBYTE_APPROACHING",
                "message": f"Building is from {building_year} ({building_age} years old). "
                           "A stambyte may be needed within 5–15 years. Check the BRF annual report "
                           "for the maintenance plan.",
                "severity": "medium"
            })
            if risk_level == "LOW":
                risk_level = "MEDIUM"

    # Monthly fee relative to size
    fee_per_sqm = monthly_fee / size_sqm if size_sqm > 0 else 0
    if fee_per_sqm > 80:
        risks.append({
            "flag": "HIGH_FEE",
            "message": f"Monthly fee is {fee_per_sqm:.0f} SEK/m², which is above average for Stockholm. "
                       "This could indicate the BRF is covering high costs or paying off loans.",
            "severity": "medium"
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"

    return {
        "monthly_fee": monthly_fee,
        "debt_per_sqm": debt_per_sqm,
        "maintenance_fund_per_sqm": maintenance_fund_per_sqm,
        "building_year": building_year,
        "estimated_total_monthly_cost": round(total_monthly_cost),
        "mortgage_monthly": round(monthly_mortgage_interest + monthly_amortization),
        "loan_amount": round(loan_amount),
        "stambyte_risk": stambyte_risk,
        "risk_level": risk_level,
        "risks": risks,
    }
