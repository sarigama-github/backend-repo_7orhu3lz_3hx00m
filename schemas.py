"""
Travair Schemas

Each Pydantic model here corresponds to a MongoDB collection with the
lowercased class name, e.g. Trip -> "trip".
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Trip(BaseModel):
    destination: str
    days: int = Field(..., ge=1, le=30)
    budget: float = Field(..., ge=0)
    preferences: List[str] = []
    estimated_cost: Optional[float] = None

class Guide(BaseModel):
    name: str
    city: str
    rating: float = Field(..., ge=0, le=5)
    price_per_hour: float = Field(..., ge=0)
    languages: List[str] = ["en"]
    available: bool = True

class Booking(BaseModel):
    type: str = Field(..., description="flight|hotel|bus|cab|tour|package")
    details: Dict[str, Any]
    user_id: Optional[str] = None
    status: str = "pending"
