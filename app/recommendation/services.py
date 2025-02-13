import structlog
import json
from app.recommendation.algorithms.content_based_filtering import content_based_filtering
from app.recommendation.algorithms.collaborative_based_filtering import collaborative_filtering
import pandas as pd
from sqlalchemy import text
from app.core.config import settings

logger = structlog.get_logger('content_based_filtering')

def get_content_based_recommendations(db, accountId):
    # Fetch mentee By mentee id
    mentee_query_string = text(''' 
      select 
      acc.id    as accountId,
      acc.name         as account_name,
      acc.learning_goal     as learning_goal,
      stu.major             as major,
      stu.educational_level as educational_level,
      pa.name               as profile_name,
      pa.type               as type,
      pa.description        as description,
      pa.organization       as organization,
      pa.position           as position,
      pa.major              as major
         from accounts acc
         join students stu on acc.id = stu.account_id
         join profile_achievements pa on acc.id = pa.account_id
         where acc.id = :accountId and
      acc.status not in ('INACTIVE', 'SUSPENDED') and acc.is_active = true and
      stu.is_active = true and pa.is_active = true''')

    mentee = pd.DataFrame(db.execute(mentee_query_string, {'accountId': accountId}).fetchall())

    # Fetch mentees and mentors data from database
    mentors_query_string = text('''
      select acc.id    as account_id,
      acc.name         as account_name,
      men.target_level as target_level,
      pa.name          as profile_name,
      pa.major         as major,
      pa.organization  as organization,
      pa.position      as position,
      pa.type          as type
         from accounts acc
         join mentors men on acc.id = men.account_id
         join profile_achievements pa on acc.id = pa.account_id 
      where acc.status not in ('INACTIVE', 'SUSPENDED') and acc.is_active = true and
      men.is_active = true and pa.is_active = true''')

    mentors = pd.DataFrame(db.execute(mentors_query_string).fetchall())

    # Call Content-Based Filtering
    return content_based_filtering(mentee, mentors, settings.TOP_K_RECOMMENDATION)


def get_collaborative_filtering_recommendations(db, accountId):
    # Fetch clicks data from database
    clicks_query_string = text('''
      select student_account_id, mentor_account_id, no_clicks as click_count
          from account_clicks
    ''')

    clicks_df =  pd.DataFrame(db.execute(clicks_query_string).fetchall())

    # mentors_query_string = text('''
    #   select acc.id    as account_id,
    #   acc.name         as account_name
    #      from accounts acc join mentors men on acc.id = men.account_id ''')
    
    # mentors_df = pd.DataFrame(db.execute(mentors_query_string).fetchall())

    # Fetch student data from database
    student_query_string = text('''
      select acc.id    as account_id,
      acc.name         as account_name
         from accounts acc join students stu on acc.id = stu.account_id
                                and acc.account_type = 'STUDENT'
         where acc.id = :accountId''')
    
    student_record = pd.DataFrame(db.execute(student_query_string, {'accountId': accountId}).fetchall())

    if student_record.empty:
        logger.error(f"Student with account id {accountId} not found")
        return None

    recommendations = collaborative_filtering(student_record, clicks_df, top_k_neighbors=5, top_n=10, lambda_param=10,
                                              transform_similarity=True)
    
    logger.info(
        f"Recommended mentors for student {student_record['account_id']}:\n" +
        json.dumps(recommendations.to_dict(orient='records'), indent=4, ensure_ascii=False)
    )

    return recommendations
                            

