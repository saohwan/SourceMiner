import os
import re
from difflib import SequenceMatcher
from time import time
from git import Repo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_tokens(source):
    return re.sub(r"[^\w\s]", "", source).lower().split()


def tokenize_directory(directory_path, extensions):
    print(f"입력받은 파일 경로: {directory_path}")
    token_tuples = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if any(re.match(pattern, file) for pattern in extensions):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                        tokens = extract_tokens(source)
                        token_tuples.append((file_path, tokens))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    return token_tuples


def clone_repository(repo_url, target_dir):
    print(f"Cloning Git repository to {target_dir}...")
    Repo.clone_from(repo_url, target_dir)
    print("Git repository cloning completed.")


def calculate_similarity(tokens1, tokens2, vectorizer):
    transformed_tokens1 = vectorizer.transform([" ".join(tokens1)])
    transformed_tokens2 = vectorizer.transform([" ".join(tokens2)])

    similarity_value = cosine_similarity(transformed_tokens1, transformed_tokens2)[0, 0]
    return similarity_value


def main():
    repo_url = input("Enter Git repository URL: ")
    target_dir = input("Enter directory path to clone to: ")
    print(f"target_dir: {target_dir}")

    clone_repository(repo_url, target_dir)

    extensions = [r".*\.(c|cpp|py|java|js|html|css|php|sql|txt)$"]
    source_directory_path = os.path.expanduser("~/dev/learn-airflow")

    target_tokens = tokenize_directory(target_dir, extensions)
    print(f"target_tokens: {target_tokens}")
    source_tokens_list = tokenize_directory(source_directory_path, extensions)
    print(f"source_tokens_list: {source_tokens_list}")

    average_similarity = 0.0
    start_time = time()
    elapsed_time = 0.0

    vectorizer = CountVectorizer()

    if len(target_tokens) == 0 or len(source_tokens_list) == 0:
        print("Unable to calculate average similarity due to missing files.")
    else:
        vectorizer.fit([" ".join(tokens) for _, tokens in target_tokens])

        for target_file_path, target_file_tokens in target_tokens:
            max_similarity = 0.0
            for _, source_tokens in source_tokens_list:
                similarity_value = calculate_similarity(
                    target_file_tokens, source_tokens, vectorizer
                )
                print(f"Similarity:  {similarity_value}")
                max_similarity = max(max_similarity, similarity_value)
                elapsed_time = time() - start_time

                print(
                    f"Similarity between {target_file_path} and {source_directory_path}: {similarity_value:.2f}"
                )
                print(
                    f"Max Similarity for {target_file_path}: {max_similarity * 100:.2f}%")
                print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

            average_similarity += max_similarity

        total_files = len(target_tokens)
        print(f"Average Similarity: {average_similarity / total_files * 100:.2f}%")
        print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
