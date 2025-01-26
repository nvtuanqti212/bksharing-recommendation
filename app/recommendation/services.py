from app.recommendation.algorithms.content_based_filtering import content_based_filtering
import pandas as pd

def get_content_based_recommendations(db):
    # Fetch mentees and mentors data from database
    mentees = pd.DataFrame(db.query("SELECT * FROM students_with_achievements"))
    mentors = pd.DataFrame(db.query("SELECT * FROM mentors_with_achievements"))

    # Call Content-Based Filtering
    return content_based_filtering(mentees, mentors)
