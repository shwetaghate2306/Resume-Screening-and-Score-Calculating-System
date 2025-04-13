import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load Resumes
def load_resumes(resumes_folder):
    resumes = {}
    for filename in os.listdir(resumes_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(resumes_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:

                resumes[filename] = file.read()
    return resumes

# Step 2: Load Job Description
def load_job_description(jd_path):
    with open(jd_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 3: Rank Resumes based on Similarity to Job Description
def rank_resumes(resumes, job_description):
    documents = list(resumes.values()) + [job_description]
    filenames = list(resumes.keys())

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity: Last doc is the JD
    cosine_sim = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1:])

    # Create results with score
    scores = [(filenames[i], round(float(cosine_sim[i][0]), 4)) for i in range(len(filenames))]
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked

# Step 4: Display Top Matches
def display_results(ranked):
    print("Ranked Resumes:")
    for i, (name, score) in enumerate(ranked):
        print(f"{i+1}. {name} - Score: {score}")

# Main
if __name__ == "__main__":
    resumes_folder = "resumes"          # Folder containing .txt resumes
    jd_path = "job_description.txt"    # File containing the job description

    resumes = load_resumes(resumes_folder)
    jd = load_job_description(jd_path)
    ranked_resumes = rank_resumes(resumes, jd)
    display_results(ranked_resumes)
