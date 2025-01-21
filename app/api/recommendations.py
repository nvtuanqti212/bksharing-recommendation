from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.recommendation.services import get_content_based_recommendations

router = APIRouter()

@router.get("/recommendations/content-based-filtering")
async def recommend_content_based(db: Session = Depends(get_db)):
    recommendations = get_content_based_recommendations(db)
    for rec in recommendations:
        print(f"Mentee {rec['mentee_id']} is recommended mentors: {rec['recommended_mentors']}")

    return {"recommendations": recommendations}
