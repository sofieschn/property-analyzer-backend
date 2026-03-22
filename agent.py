"""
LangChain agent that orchestrates all four tools to produce
a complete property analysis.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools.listing_parser import parse_listing
from tools.brf_economy import analyze_brf_economy
from tools.sustainability import analyze_sustainability
from tools.area_intelligence import analyze_area
import json


SYSTEM_PROMPT = """You are a Swedish property analysis assistant. Your job is to help
home buyers understand the full picture of what they're signing up for when purchasing
a BRF apartment in Sweden.

You have access to four tools:
1. parse_listing - extracts property data from a listing URL
2. analyze_brf_economy - evaluates BRF financial health and risks
3. analyze_sustainability - checks energy performance vs MEPS regulations
4. analyze_area - finds nearby developments and infrastructure

When given a listing URL:
1. First, parse the listing to extract property data
2. Then run the economy, sustainability, and area tools in parallel using the extracted data
3. Compile all results into a structured analysis

When asked a follow-up question about a specific data point, explain your reasoning
clearly and cite the underlying data that supports your assessment."""


async def run_analysis(url: str) -> dict:
    """Run the full property analysis pipeline."""

    # Step 1: Parse the listing
    listing_data = await parse_listing.ainvoke({"url": url})

    if not listing_data or not listing_data.get("address"):
        return {"error": "Could not parse listing. Please try manual input."}

    # Step 2: Run analysis tools (would be parallel with asyncio.gather in production)
    import asyncio

    economy_task = analyze_brf_economy.ainvoke({
        "asking_price": listing_data.get("asking_price", 0),
        "monthly_fee": listing_data.get("monthly_fee", 0),
        "size_sqm": listing_data.get("size_sqm", 0),
        "building_year": listing_data.get("building_year"),
        "debt_per_sqm": listing_data.get("debt_per_sqm"),
        "maintenance_fund_per_sqm": listing_data.get("maintenance_fund_per_sqm"),
    })

    sustainability_task = analyze_sustainability.ainvoke({
        "address": listing_data["address"],
        "building_year": listing_data.get("building_year"),
        "building_type": "residential",
    })

    area_task = analyze_area.ainvoke({
        "address": listing_data["address"],
    })

    economy_result, sustainability_result, area_result = await asyncio.gather(
        economy_task, sustainability_task, area_task
    )

    # Step 3: Generate overall score and summary
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    area_summary = {k: {"count": v.get("count", 0), "closest": v["items"][0]["name"] if v.get("items") else None}
                     for k, v in area_result.get("nearby", {}).items()}

    summary_prompt = f"""Based on this Swedish property analysis, generate:
1. A risk score from 1-10 (10 = safest, 1 = highest risk)
2. A risk label: "Safe investment", "Proceed with caution", or "High risk — investigate further"
3. A 2-3 sentence summary of the most important findings for a buyer

Property: {listing_data.get('address')} — {listing_data.get('asking_price')} SEK

BRF Economy: {json.dumps(economy_result, default=str)}
Sustainability: {json.dumps(sustainability_result, default=str)}
Area (nearby counts): {json.dumps(area_summary, default=str)}

Return ONLY JSON with fields: score (int), label (string), summary (string)"""

    summary_response = await llm.ainvoke(summary_prompt)
    try:
        summary = json.loads(summary_response.content.strip().strip("```json").strip("```"))
    except json.JSONDecodeError:
        summary = {"score": 5, "label": "Proceed with caution", "summary": "Analysis complete. Review individual panels for details."}

    return {
        "property": listing_data,
        "economy": economy_result,
        "sustainability": sustainability_result,
        "area": area_result,
        "summary": summary,
    }


async def chat_about_data(context: str, question: str) -> str:
    """Answer follow-up questions about specific data points."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""The user is looking at this data point from their property analysis:

Context: {context}

Their question: {question}

Provide a clear, helpful explanation in 2-4 sentences. Be specific about WHY this data
point is significant for a home buyer. Reference Swedish regulations, typical BRF
economics, or local market conditions where relevant."""),
    ]

    response = await llm.ainvoke(messages)
    return response.content
