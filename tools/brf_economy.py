"""
Tool 2: BRF Economy
Analyzes the financial health of a BRF using listing data and public records.
Calculates total monthly cost, flags risks like high debt or upcoming stambyte.
"""

from langchain_core.tools import tool
from typing import Optional
import math


@tool
async def analyze_brf_economy(
    asking_price: int,
    monthly_fee: int,
    size_sqm: float,
    building_year: Optional[int] = None,
    debt_per_sqm: Optional[int] = None,
    maintenance_fund_per_sqm: Optional[int] = None,
) -> dict:
    """Analyze the BRF's financial health based on listing data.
    Calculates total monthly cost and flags economic risks."""

    # --- Mortgage calculation ---
    # Swedish standard: 85% loan-to-value, average rate ~4.2% (2026)
    loan_amount = asking_price * 0.85
    monthly_interest_rate = 0.042 / 12
    # Swedish mortgages: interest-only is common, but we show full cost
    monthly_mortgage_interest = loan_amount * monthly_interest_rate
    # Mandatory amortization: 2% if LTV > 70%, 1% if LTV > 50%
    annual_amortization_rate = 0.02
    monthly_amortization = (loan_amount * annual_amortization_rate) / 12
    total_monthly_cost = monthly_fee + monthly_mortgage_interest + monthly_amortization

    # --- Risk assessment ---
    risks = []
    risk_level = "LOW"

    # Debt per m2 assessment
    if debt_per_sqm is not None:
        if debt_per_sqm > 10000:
            risks.append({
                "flag": "HIGH_DEBT",
                "message": f"BRF debt is {debt_per_sqm} SEK/m2, well above the 10,000 SEK/m2 warning threshold. "
                           "This indicates the association carries significant loans that may lead to fee increases.",
                "severity": "high"
            })
            risk_level = "HIGH"
        elif debt_per_sqm > 7000:
            risks.append({
                "flag": "ELEVATED_DEBT",
                "message": f"BRF debt at {debt_per_sqm} SEK/m2 is moderately high. Monitor for potential fee increases.",
                "severity": "medium"
            })
            if risk_level == "LOW":
                risk_level = "MEDIUM"

    # Maintenance fund assessment
    if maintenance_fund_per_sqm is not None:
        if maintenance_fund_per_sqm < 1000:
            risks.append({
                "flag": "LOW_FUND",
                "message": f"Maintenance fund at {maintenance_fund_per_sqm} SEK/m2 is below recommended levels. "
                           "Large repairs may require special assessments or fee increases.",
                "severity": "medium"
            })
            if risk_level == "LOW":
                risk_level = "MEDIUM"

    # Stambyte (pipe replacement) assessment based on building year
    stambyte_risk = False
    stambyte_note = None
    if building_year is not None:
        building_age = 2026 - building_year
        if building_age > 50 and building_age < 70:
            stambyte_risk = True
            stambyte_note = (
                f"Building is from {building_year} ({building_age} years old). "
                "Pipes typically last 50-60 years. A stambyte (pipe replacement) is likely within the next 5 years. "
                "Cost: typically 5,000-8,000 SEK/m2, often financed through increased monthly fees (+500-1000 SEK/month for 10-15 years)."
            )
            risks.append({
                "flag": "STAMBYTE_LIKELY",
                "message": stambyte_note,
                "severity": "high"
            })
            risk_level = "HIGH"
        elif building_age >= 40 and building_age <= 50:
            stambyte_risk = True
            stambyte_note = (
                f"Building is from {building_year} ({building_age} years old). "
                "A stambyte may be needed within 5-15 years. Check BRF annual report for maintenance plan."
            )
            risks.append({
                "flag": "STAMBYTE_POSSIBLE",
                "message": stambyte_note,
                "severity": "medium"
            })
            if risk_level == "LOW":
                risk_level = "MEDIUM"

    # Monthly fee relative to size
    fee_per_sqm = monthly_fee / size_sqm if size_sqm > 0 else 0
    if fee_per_sqm > 80:
        risks.append({
            "flag": "HIGH_FEE",
            "message": f"Monthly fee is {fee_per_sqm:.0f} SEK/m2, which is above average for Stockholm. "
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
        "stambyte_note": stambyte_note,
        "risk_level": risk_level,
        "risks": risks,
    }
