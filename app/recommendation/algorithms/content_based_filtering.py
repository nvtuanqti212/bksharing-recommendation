from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def combine_achievements(achievements: list) -> str:
    """
    Combine a list of achievements into a single string.
    :param achievements: List of dictionaries with achievement details.
    :return: Combined string of achievements.
    """
    combined = []
    for achievement in achievements:
        combined.append(f"{achievement['type']}: {achievement['name']} at {achievement.get('organization', '')} ({achievement.get('position', '')})")
    return ", ".join(combined)

def content_based_filtering(mentees: pd.DataFrame, mentors: pd.DataFrame, top_k: int = 10):
    """
    Recommend mentors for mentees using Content-Based Filtering.

    :param mentees: DataFrame containing mentee data.
    :param mentors: DataFrame containing mentor data.
    :param top_k: Number of top recommendations.
    :return: Recommendations for each mentee.
    """
    # Step 1: Combine achievements into a single field
    mentees['achievements_combined'] = mentees['achievements'].apply(combine_achievements)
    mentors['achievements_combined'] = mentors['achievements'].apply(combine_achievements)

    # Step 2: Combine all features into a single string
    mentees['combined_features'] = (
        mentees['preferences'] + " " +
        mentees['learningGoal'] + " " +
        mentees['educationalLevel'] + " " +
        mentees['major'] + " " +
        mentees['achievements_combined']
    )
    
    mentors['combined_features'] = (
        mentors['expertise'] + " " +
        mentors['description'] + " " +
        mentors['targetLevels'] + " " +
        mentors['achievements_combined']
    )

    # Step 3: Vectorize the combined features using TF-IDF
    vectorizer = TfidfVectorizer()
    mentee_vectors = vectorizer.fit_transform(mentees['combined_features'])
    mentor_vectors = vectorizer.transform(mentors['combined_features'])

    # Step 4: Calculate cosine similarity
    recommendations = []
    for mentee_idx, mentee_vector in enumerate(mentee_vectors):
        similarity_scores = cosine_similarity(mentee_vector, mentor_vectors).flatten()
        
        # Step 5: Sort mentors by similarity scores and pick top_k
        top_matches = similarity_scores.argsort()[-top_k:][::-1]
        recommended_mentors = mentors.iloc[top_matches]['mentor_id'].tolist()

        recommendations.append({
            "mentee_id": mentees.iloc[mentee_idx]['mentee_id'],
            "recommended_mentors": recommended_mentors
        })

    return recommendations
