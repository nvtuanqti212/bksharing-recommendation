import structlog
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from app.core.config import settings

logger = structlog.get_logger('content_based_filtering')


def process_achievements_by_type(achievements) -> dict: 
    """
    Split achievements into categories and weight them.
    Expected achievement dict format:
      {
        'profile_name': ...,
        'type': one of 'EXPERIENCE', 'EDUCATION', or other for certifications/projects
        'description': ...,
        'organization': ...,
        'position': ...,
        'major': ...
      }
    :param achievements: List of achievement dictionaries.
    :return: Dictionary with combined text for each category.
    """
    experience = []
    education = []
    certification = []
    
    for idx, achievement in enumerate(achievements, start=1):
        # Convert pandas Series to dict, if necessary
        if isinstance(achievement, pd.Series):
            achievement = achievement.to_dict()
        # Ensure achievement is a dictionary
        if not isinstance(achievement, dict):
            logger.warning(f"Invalid achievement format at index {idx}: {achievement}")
            continue

        if achievement['type'] == 'EXPERIENCE':
            experience.append(f"{achievement['organization']} {achievement['position']}")
        elif achievement['type'] == 'EDUCATION':
            education.append(f"{achievement['organization']} {achievement['major']}")
        elif achievement['type'] == 'CERTIFICATION':
            certification.append(f"{achievement['profile_name']} {achievement['organization']}")
        else:
            logger.warning(f"Unknown achievement type '{achievement['type']}' at index {idx}")


    # Combine text for each category
    return {
        'experience': " ".join(experience),
        'education': " ".join(education),
        'certification': " ".join(certification)
    }


def combine_weighted_features(base_text: str, achievements_processed: dict) -> str:
    """
    Combine base features with processed achievements, applying a weight for each type.
    
    Weights can be adjusted as needed. For example:
      experience: 1.5
      education: 1.0
      certification: 0.5
    
    Since we are dealing with text, one way to “weight” a feature is to repeat the text
    proportionally to its weight value (after scaling to an integer factor).
    
    :param base_text: Other textual features (e.g., learning goals, major, etc.)
    :param achievements_processed: Dictionary with keys 'experience', 'education', 'certification'
                                   each containing the combined achievement text.
    :return: Combined feature string with weighted achievements.
    """
    # Define weight factors for each type
    weights = {
        'experience': settings.PROFILE_EXPERIENCE_WEIGHT,
        'education': settings.PROFILE_EDUCATION_WEIGHT,
        'certification': settings.PROFILE_CERTIFICATION_WEIGHT
    }
    
    weighted_text = []
    for key, weight in weights.items():
        # Scale weight factor for repetition; adjust multiplier as needed.
        # We use int(round(...)) to convert the scaled weight to an integer repetition factor.
        repetition_factor = int(round(weight * 10))
        achievement_text = achievements_processed[key]
        if achievement_text:
            # Repeat the text to emphasize its importance
            weighted_text.append((achievement_text + " ") * repetition_factor)
    
    combined = base_text + " " + " ".join(weighted_text)
    return combined


def process_mentor_data(mentors: pd.DataFrame) -> pd.DataFrame:
    """
    Process mentor data to aggregate achievements and create combined text features.
    
    :param mentors: DataFrame containing mentors and their achievements.
    :return: Processed DataFrame with a combined text field for each mentor.
    """
    processed_mentors = []

    # Group by mentor ID (or account ID)
    grouped = mentors.groupby('account_id')

    for account_id, group in grouped:
        # Aggregate achievements for the current mentor
        achievements = group[['profile_name', 'type', 'organization', 'position', 'major']].to_dict('records')
        aggregated_achievements = process_achievements_by_type(achievements)
        
        # Extract common mentor-specific information (assume it’s the same for all rows in the group)
        common_info = group.iloc[0][['account_name', 'target_level']].to_dict()
        # Check target_level is not empty
        if common_info['target_level']:
            if isinstance(common_info['target_level'], (set, list)):
                target_level = ", ".join(common_info['target_level'])
            else:
            # Split the string by commas if it's a string representation of a set
                target_level = ", ".join(common_info['target_level'].strip('{}').split(', '))
        common_info_str = f"Name: {common_info['account_name']} Target Level: {target_level}"
        
        # Combine weighted features for the mentor
        combined_text = combine_weighted_features(common_info_str, aggregated_achievements)
        
        # Append the processed data for this mentor
        processed_mentors.append({
            'account_id': account_id,
            'account_name': common_info['account_name'],
            'combined_features': combined_text
        })

    # Convert processed data into a DataFrame
    return pd.DataFrame(processed_mentors)


def content_based_filtering(mentee: pd.DataFrame, mentors: pd.DataFrame, top_k: int = 5):
    logger.info(f"Recommend mentors for mentors: {mentors}")
    """
    Recommend mentors for a given mentee using content-based filtering.
    Processing steps:
      1. Process the mentee's achievement record.
      2. Process each mentor's achievement record.
      3. Combine achievements with other profile features.
      4. Build TF-IDF vectors / CountVectorizer and compute cosine similarity. 
      5. Return top_k mentor recommendations.
    :param mentee: A pandas Series representing a single mentee's record.
    :param mentors: A pandas DataFrame with mentor records.
    :param top_k: Number of top recommendations to return.
    :return: DataFrame of recommended mentors with key columns.
    """

    # Check empty mentee or mentor data
    if mentee.empty:
        logger.warning("Empty mentee data")
        return pd.DataFrame()
    if mentors.empty:
        logger.warning("Empty mentor data")
        return pd.DataFrame()

    # Process mentee achievement (wrap single record dict in a list)
    achievements = mentee[['profile_name', 'type', 'description', 'organization', 'position', 'major']].to_dict('records')

    # Process mentee achievements by type including experience, education, and certification
    mentee_achievements = process_achievements_by_type(achievements)

    # Base textual features for mentee
    # Extract common information for the mentee (assume the first row contains the common fields)
    common_info = mentee.iloc[0][['learning_goal', 'major', 'educational_level']].to_dict()
    # The major field is weighted at 1.0 by default, 

    common_info_str = " ".join(
        f"{value} " * (int(round(settings.PROFILE_MAJOR_WEIGHT * 10)) if key == 'major' else 1)
        for key, value in common_info.items() if value is not None
    )

    mentee_text = combine_weighted_features(common_info_str, mentee_achievements)

    # Process mentor achievements: a mentor has multiple achievements and 
    # we query all mentors in database
    mentors = mentors.copy()  # avoid modifying original DataFrame
    mentors_processed = process_mentor_data(mentors)

    # Print processed mentor data for debugging with combined features
    logger.info(
        f"Processed mentor data:\n{json.dumps(mentors_processed.to_dict(orient='records'), indent=4, ensure_ascii=False)}"
    )
    
    # Build Count Vectors for the mentee and mentors
    vectorizer = TfidfVectorizer(stop_words='english') # Stop words are common words (e.g. "the", "and", "is") 

    # Fit the vectorizer on the mentee text and transform the mentor text
    # NOTE There are some words like university, college, etc. that are common in education and experience
    # but may not be relevant for the recommendation. We can adjust the vectorizer settings to exclude them.
    mentee_vector = vectorizer.fit_transform([mentee_text])
    logger.info(f"Mentee vector features: {vectorizer.get_feature_names_out()}")
    logger.info(f"Mentee vector shape: {mentee_vector.shape}")
    # Transform mentor text following the mentee vectorizer
    mentor_vectors = vectorizer.transform(mentors_processed["combined_features"])
    logger.info(f"Mentor vector features: {vectorizer.get_feature_names_out()}")
    logger.info(f"Mentor vectors shape: {mentor_vectors.shape}")
    
    # Compute cosine similarity between the mentee and each mentor
    similarities = cosine_similarity(mentee_vector, mentor_vectors)
    logger.info(f"Similarity matrix shape: {similarities}")
    
    # Get indices of top_k mentors with highest similarity values
    top_indices = similarities.argsort(axis=1).flatten()[-top_k:][::-1]
    logger.info(f"Top indices: {top_indices}")

    # Get the corresponding similarity scores
    similarity_scores = similarities[0][top_indices]
    
    # Select the recommended mentors and add the similarity score column
    recommendations = mentors_processed.iloc[top_indices].copy()
    recommendations["similarity"] = similarity_scores
    
    # Return recommendations containing key mentor information along with similarity score
    return recommendations[['account_id', 'account_name', 'similarity']]


