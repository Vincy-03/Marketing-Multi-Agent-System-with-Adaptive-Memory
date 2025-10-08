# agents/campaign_optimization_agent.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Metrics(BaseModel):
    campaign_id: str
    impressions: int
    clicks: int
    conversions: int

@router.post("/optimize")
async def optimize(m: Metrics):
    ctr = (m.clicks / m.impressions) if m.impressions > 0 else 0
    conv_rate = (m.conversions / m.clicks) if m.clicks > 0 else 0

    suggestions = []
    action = "keep_running"

    if ctr < 0.02:  # less than 2%
        action = "suggest_ab_test"
        suggestions.append("Run A/B test on subject lines or CTA")
    if conv_rate < 0.05:  # less than 5%
        suggestions.append("Improve landing page / reduce friction")
    if ctr >= 0.02 and conv_rate >= 0.05:
        suggestions.append("Campaign performing well")

    return {
        "campaign_id": m.campaign_id,
        "ctr": ctr,
        "conv_rate": conv_rate,
        "action": action,
        "suggestions": suggestions
    }
