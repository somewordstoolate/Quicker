from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity_match(query: str, candidates: list, threshold=0.9) -> str | None:
    """
    Check whether there are similar strings in the candidates whose similarity exceeds the threshold. If so, return the most similar string.

    Args:
        query (str): The string to compare.
        candidates (list): A list of strings to compare.
        threshold (float): The threshold of similarity. The default is 0.9.

    Returns:
        str: The most similar string in the candidates. If there is no similar string, return None.
    """

    # Initialize the most similar string and the highest similarity.
    most_similar_str = None
    highest_similarity = 0
    vectorizer = TfidfVectorizer()

    # Compare the similarity between the query and each candidate.
    for candidate in candidates:
        tfidf_matrix = vectorizer.fit_transform([query, candidate])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity = cos_sim[0][0]

        # Update the most similar string and the highest similarity if the similarity exceeds the threshold.
        if similarity > threshold and similarity > highest_similarity:
            most_similar_str = candidate
            highest_similarity = similarity

    return most_similar_str


if __name__ == "__main__":
    query = "Patients with dementia and agitation\/aggressive behavior"
    candidates = [
        "Patients with dementia and agitation/aggressive behavior",
        "Patients with dementia and agitation/aggressive behavior.",
        "Patients with dementia and agitation\aggressive behavior",
    ]
    result = similarity_match(query, candidates)
    print(result)  # Output: I want to buy a computer
