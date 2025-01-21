from fastapi import FastAPI
from app.api.recommendations import router as recommendation_router

app = FastAPI()

app.include_router(recommendation_router, prefix="/api/v1", tags=["recommendation"])

@app.get("/")
async def root():
    return {"message": "Welcome to Recommendation System"}
