from app.recommendation.algorithms.content_based_filtering import content_based_filtering
import pandas as pd
from sqlalchemy import text
from app.core.config import settings

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
                     where acc.id = :accountId''')

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
         join profile_achievements pa on acc.id = pa.account_id''')

    mentors = pd.DataFrame(db.execute(mentors_query_string).fetchall())

    # Call Content-Based Filtering
    return content_based_filtering(mentee, mentors, settings.TOP_K_RECOMMENDATION)


