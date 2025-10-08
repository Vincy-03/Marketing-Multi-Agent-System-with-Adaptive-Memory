from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents.lead_triage_agent import router as lead_router
from agents.engagement_agent import router as engage_router
from agents.campaign_optimization_agent import router as optimize_router

# =============================
# App Setup & Swagger Settings
# =============================
app = FastAPI(
    title="Marketing Multi-Agent System",
    version="1.0.0",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,   # hide large schema models
        "docExpansion": "none",           # keep sections collapsed
        "defaultModelRendering": "model", # cleaner model view
        "tryItOutEnabled": True,          # enable Try it out
        "displayRequestDuration": True,   # show request duration
    }
)

# =============================
# Enable CORS (fix Swagger issue)
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (safe for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Routers for each agent
# =============================
app.include_router(lead_router, prefix="", tags=["Lead Triage Agent"])
app.include_router(engage_router, prefix="", tags=["Engagement Agent"])
app.include_router(optimize_router, prefix="", tags=["Campaign Optimization Agent"])

# =============================
# Root Endpoint
# =============================
@app.get("/")
def root():
    return {"message": "PurpleMerit assessment demo - agents available"}
