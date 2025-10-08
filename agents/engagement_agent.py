# agents/engagement_agent.py
from fastapi import APIRouter
from pydantic import BaseModel
from memory.long_term import get_lead_preferences

router = APIRouter()

class EngageRequest(BaseModel):
    lead_id: str
    category: str

@router.post("/engage")
async def engage(req: EngageRequest):
    prefs = get_lead_preferences(req.lead_id)

    message = f"Hi {req.lead_id}, thanks for your interest."
    if req.category == "Campaign Qualified":
        message += " We recommend scheduling a demo."
    elif req.category == "General Inquiry":
        message += " We'd be happy to answer your questions."
    else:
        message += " Thanks for checking in."

    # Add personalization if available
    if prefs:
        if "preferred_channels" in prefs:
            channel = prefs["preferred_channels"][0]
            message += f" We'll reach you via your preferred channel: {channel}."
        if "best_contact_time" in prefs:
            message += f" Best time to contact: {prefs['best_contact_time']}."

    return {
        "lead_id": req.lead_id,
        "status": "engaged",
        "message": message
    }
