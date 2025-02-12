import structlog

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.recommendation.services import get_content_based_recommendations, get_collaborative_filtering_recommendations


logger = structlog.get_logger('recommendations.py')

router = APIRouter()

# Get recommendations for mentee by accountId
@router.get ("/content-based-filtering/{accountId}")
async def recommend_content_based_by_account_id(accountId: int, db: Session = Depends(get_db)):
    recommendations = get_content_based_recommendations(db, accountId)

    # Convert Pandas DataFrame to list of dicts to avoid numpy types issues
    recommendations_dict = recommendations.to_dict(orient="records")
    return {"recommendations": recommendations_dict}

# Get recommendations for mentee by accountId  
@router.get ("/collaborative-filtering/{accountId}")
async def recommend_collaborative_by_account_id(accountId: int, db: Session = Depends(get_db)):
    recommendations = get_collaborative_filtering_recommendations(db, accountId)

    if recommendations is None:
        return {"recommendations": []}

    # Convert Pandas DataFrame to list of dicts to avoid numpy types issues
    recommendations_dict = recommendations.to_dict(orient="records")
    return {"recommendations": recommendations_dict}

@router.get ("/test")
async def test():
    return {"message": "Hello world!"}


