import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

# Database helpers
from database import db, create_document, get_documents

app = FastAPI(title="Travair API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Models --------------------
class ItineraryRequest(BaseModel):
    destination: str
    days: int = Field(..., ge=1, le=30)
    budget: float = Field(..., ge=0)
    preferences: List[str] = []
    travelers: Optional[int] = 1
    travel_month: Optional[str] = None

class ItineraryDay(BaseModel):
    day: int
    title: str
    activities: List[str]
    tips: List[str] = []

class ItineraryResponse(BaseModel):
    destination: str
    days: int
    estimated_cost: float
    weather_notes: Optional[str] = None
    safety_tips: List[str] = []
    plan: List[ItineraryDay]

class RouteShieldRequest(BaseModel):
    origin: str
    destination: str
    date: str
    transport_mode: str = Field(default="flight", description="flight|train|bus|cab")

class RiskItem(BaseModel):
    type: str
    severity: str
    description: str
    mitigation: str

class RouteShieldResponse(BaseModel):
    risks: List[RiskItem]
    alerts: List[str] = []
    alternatives: List[str] = []
    support_contacts: List[str] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class AssistantRequest(BaseModel):
    messages: List[ChatMessage]
    locale: Optional[str] = "en"

class Guide(BaseModel):
    name: str
    city: str
    rating: float
    price_per_hour: float
    languages: List[str]
    available: bool = True

class BookingRequest(BaseModel):
    type: str = Field(..., description="flight|hotel|bus|cab|tour|package")
    details: Dict[str, Any]
    user_id: Optional[str] = None

class BookingResponse(BaseModel):
    booking_id: str
    status: str

# -------------------- Utils --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _call_openai_system(prompt: str, temperature: float = 0.6) -> Optional[str]:
    """Lightweight call to OpenAI with graceful fallback when key missing."""
    if not OPENAI_API_KEY:
        return None
    try:
        import requests
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are Travair, an expert AI travel planner and safety assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenAI error:", e)
        return None

# -------------------- Routes --------------------
@app.get("/")
def root():
    return {"name": "Travair API", "status": "ok"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
        else:
            response["database"] = "❌ Not Initialized"
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:80]}"
    return response

@app.post("/api/itinerary", response_model=ItineraryResponse)
def generate_itinerary(req: ItineraryRequest):
    # Try AI generation
    user_prompt = (
        f"Create a day-wise travel plan for {req.destination} for {req.days} days. "
        f"Budget: {req.budget}. Preferences: {', '.join(req.preferences) if req.preferences else 'general'}. "
        f"Include hidden attractions, best times, weather-based suggestions, emergency tips and a price estimate."
    )
    ai_text = _call_openai_system(user_prompt)

    # If AI unavailable, create a heuristic plan
    plan: List[ItineraryDay] = []
    for i in range(1, req.days + 1):
        plan.append(ItineraryDay(
            day=i,
            title=f"Day {i} in {req.destination}",
            activities=[
                "Morning: Iconic landmark visit",
                "Afternoon: Local market & street food",
                "Evening: Scenic point or cultural show"
            ],
            tips=["Carry a reusable water bottle", "Use official taxis or rideshare"]
        ))

    est = max(150.0 * (req.days or 1), 200.0)
    weather = f"Check seasonal patterns for {req.travel_month or 'your travel dates'} and pack accordingly."
    safety_tips = [
        "Keep digital copies of documents",
        "Share live location with a trusted contact",
        "Avoid poorly lit areas late night"
    ]

    if ai_text:
        # Try to enrich with AI summary embedded as an extra tip on Day 1
        plan[0].tips.append("AI summary: " + ai_text[:180] + ("…" if len(ai_text) > 180 else ""))

    # Persist minimal trip document
    try:
        create_document("trip", {
            "destination": req.destination,
            "days": req.days,
            "budget": req.budget,
            "preferences": req.preferences,
            "estimated_cost": est,
        })
    except Exception as e:
        print("DB insert error:", e)

    return ItineraryResponse(
        destination=req.destination,
        days=req.days,
        estimated_cost=est,
        weather_notes=weather,
        safety_tips=safety_tips,
        plan=plan
    )

@app.post("/api/routeshield", response_model=RouteShieldResponse)
def route_shield(req: RouteShieldRequest):
    # Simple heuristic + optional AI enrichment
    risks = [
        RiskItem(type="weather", severity="medium", description="Possible showers", mitigation="Carry light rain gear"),
        RiskItem(type="delay", severity="low", description=f"{req.transport_mode.title()} congestion during peak hours", mitigation="Start 30-45 mins early"),
    ]
    ai = _call_openai_system(
        f"Analyze route from {req.origin} to {req.destination} on {req.date} via {req.transport_mode}. "
        f"List key risks with mitigation in 3 bullets, and suggest 2 alternates.")
    alerts: List[str] = []
    alternatives: List[str] = [
        "Consider earlier departure window",
        "Keep offline maps for the region"
    ]
    if ai:
        alerts.append(ai[:220] + ("…" if len(ai) > 220 else ""))
    support = ["112 (Emergency)", "Local police helpline", "Nearest embassy (if international)"]
    return RouteShieldResponse(risks=risks, alerts=alerts, alternatives=alternatives, support_contacts=support)

@app.post("/api/assistant")
def assistant(req: AssistantRequest):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "Hello")
    ai = _call_openai_system(
        f"User message: {last_user}. Provide a concise helpful travel assistant reply in {req.locale}.",
        temperature=0.7
    )
    reply = ai or "I'm here to help with your trip. Ask me about plans, safety, or bookings."
    return {"reply": reply}

@app.get("/api/guides", response_model=List[Guide])
def get_guides(city: Optional[str] = None):
    # Try DB; if empty, seed a few virtual guides (not inserted)
    try:
        results = get_documents("guide", {"city": city} if city else {})
        if results:
            # Map DB docs to Pydantic shape
            mapped: List[Guide] = []
            for g in results:
                mapped.append(Guide(
                    name=g.get("name", "Local Guide"),
                    city=g.get("city", ""),
                    rating=float(g.get("rating", 4.6)),
                    price_per_hour=float(g.get("price_per_hour", 12.0)),
                    languages=g.get("languages", ["en"]),
                    available=bool(g.get("available", True))
                ))
            return mapped
    except Exception as e:
        print("DB guides error:", e)

    fallback = [
        Guide(name="Aarav", city=city or "Goa", rating=4.8, price_per_hour=15, languages=["en", "hi"], available=True),
        Guide(name="Maya", city=city or "Jaipur", rating=4.7, price_per_hour=14, languages=["en", "hi"], available=True),
        Guide(name="Ishan", city=city or "Delhi", rating=4.5, price_per_hour=12, languages=["en", "hi"], available=False),
    ]
    return fallback

@app.post("/api/bookings", response_model=BookingResponse)
def create_booking(req: BookingRequest):
    try:
        booking_id = create_document("booking", {
            "type": req.type,
            "details": req.details,
            "user_id": req.user_id,
            "status": "confirmed",
        })
        return BookingResponse(booking_id=booking_id, status="confirmed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Booking failed: {str(e)[:100]}")

@app.get("/api/price-compare")
def price_compare(destination: str, nights: int = 3):
    # Mock comparison
    travair = 100 * nights
    other_a = int(travair * 1.1)
    other_b = int(travair * 0.95)
    savings = min(other_a, other_b) - travair
    return {
        "destination": destination,
        "nights": nights,
        "travair_price": travair,
        "competitors": {
            "MakeMyTrip": other_a,
            "OtherPortal": other_b
        },
        "savings_vs_best": max(0, savings)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
