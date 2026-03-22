"""
FastAPI server for the Property Analyzer backend.
Exposes two endpoints:
  POST /api/analyze  — full property analysis from a listing URL
  POST /api/chat     — follow-up questions about specific data points
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Property Analyzer API",
    description="AI-powered Swedish property analysis for home buyers",
    version="1.0.0",
)

# Allow Lovable frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Lovable domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: Optional[str] = None
    # Manual input fallback
    address: Optional[str] = None
    asking_price: Optional[int] = None
    size_sqm: Optional[float] = None
    rooms: Optional[int] = None
    monthly_fee: Optional[int] = None
    brf_name: Optional[str] = None
    building_year: Optional[int] = None


class ChatRequest(BaseModel):
    context: str  # The data point being discussed
    question: str  # The user's follow-up question


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    from agent import run_analysis

    if not request.url and not request.address:
        raise HTTPException(status_code=400, detail="Provide either a listing URL or an address")

    try:
        if request.url:
            result = await run_analysis(request.url)
        else:
            # Manual input path — skip listing parser, go directly to analysis
            from tools.brf_economy import analyze_brf_economy
            from tools.sustainability import analyze_sustainability
            from tools.area_intelligence import analyze_area
            import asyncio
            import json
            from langchain_openai import ChatOpenAI

            listing_data = {
                "address": request.address,
                "asking_price": request.asking_price,
                "size_sqm": request.size_sqm,
                "rooms": request.rooms,
                "monthly_fee": request.monthly_fee,
                "brf_name": request.brf_name,
                "building_year": request.building_year,
            }

            economy_result, sustainability_result, area_result = await asyncio.gather(
                analyze_brf_economy.ainvoke({
                    "asking_price": request.asking_price or 0,
                    "monthly_fee": request.monthly_fee or 0,
                    "size_sqm": request.size_sqm or 0,
                    "building_year": request.building_year,
                }),
                analyze_sustainability.ainvoke({
                    "address": request.address,
                    "building_year": request.building_year,
                    "building_type": "residential",
                }),
                analyze_area.ainvoke({
                    "address": request.address,
                }),
            )

            # Generate summary
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            summary_prompt = f"""Based on this analysis, generate a JSON with:
score (1-10, 10=safest), label (string), summary (2-3 sentences).
Economy: {json.dumps(economy_result, default=str)}
Sustainability: {json.dumps(sustainability_result, default=str)}
Area: {json.dumps(area_result, default=str)}
Return ONLY JSON."""

            resp = await llm.ainvoke(summary_prompt)
            try:
                summary = json.loads(resp.content.strip().strip("```json").strip("```"))
            except Exception:
                summary = {"score": 5, "label": "Proceed with caution", "summary": "Review panels for details."}

            result = {
                "property": listing_data,
                "economy": economy_result,
                "sustainability": sustainability_result,
                "area": area_result,
                "summary": summary,
            }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    from agent import chat_about_data

    try:
        response = await chat_about_data(request.context, request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
