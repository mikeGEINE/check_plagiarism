import os
import nbformat
import ast
import tokenize
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import Levenshtein

def extract_code_cells(notebook_path):
    """Extract code cells from a single Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = nbformat.read(f, as_version=4)
    code_cells = [cell['source'] for cell in nb_data['cells'] if cell['cell_type'] == 'code']
    return "\n".join(code_cells)

def load_notebooks_from_dir(directory):
    """Load code cells from all Jupyter Notebooks in a directory."""
    notebooks = {}
    for file in os.listdir(directory):
        if file.endswith(".ipynb"):
            path = os.path.join(directory, file)
            notebooks[file] = extract_code_cells(path)
    return notebooks

# TEXT-BASED SIMILARITY
def text_based_similarity(notebooks):
    vectorizer = TfidfVectorizer()
    notebook_names = list(notebooks.keys())
    tfidf_matrix = vectorizer.fit_transform(notebooks.values())
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(similarity_matrix, index=notebook_names, columns=notebook_names)

def levenshtein_based_similarity(notebooks):
    def levenshtein_similarity(code1, code2):
        """
        Calculate the similarity between two strings using Levenshtein distance.
        Similarity is calculated as 1 - (Levenshtein distance / max(len(code1), len(code2))).
        """
        lev_distance = Levenshtein.distance(code1, code2)
        max_len = max(len(code1), len(code2))
        return 1 - (lev_distance / max_len)

    notebook_names = list(notebooks.keys())
    code_cells = list(notebooks.values())

    similarity_matrix = np.zeros((len(code_cells), len(code_cells)))

    for i in range(len(code_cells)):
        for j in range(len(code_cells)):
            similarity_matrix[i][j] = levenshtein_similarity(code_cells[i], code_cells[j])

    return pd.DataFrame(similarity_matrix, index=notebook_names, columns=notebook_names)

# AST-BASED SIMILARITY
def ast_based_similarity(notebooks):
    def ast_node_count(code):
        try:
            tree = ast.parse(code)
            return sum(1 for _ in ast.walk(tree))  # Count AST nodes
        except SyntaxError:
            return 0

    notebook_names = list(notebooks.keys())
    counts = [ast_node_count(code) for code in notebooks.values()]
    similarity_matrix = np.zeros((len(counts), len(counts)))

    for i in range(len(counts)):
        for j in range(len(counts)):
            similarity_matrix[i][j] = 1 - abs(counts[i] - counts[j]) / max(counts[i], counts[j], 1)
    return pd.DataFrame(similarity_matrix, index=notebook_names, columns=notebook_names)

# TOKEN-BASED SIMILARITY
def token_based_similarity(notebooks):
    def tokenize_code(code):
        try:
            tokens = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
            return [token.string for token in tokens if token.type == tokenize.NAME]
        except Exception:
            return []

    notebook_names = list(notebooks.keys())
    tokenized_notebooks = {name: tokenize_code(code) for name, code in notebooks.items()}
    all_tokens = [" ".join(tokens) for tokens in tokenized_notebooks.values()]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tokens)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(similarity_matrix, index=notebook_names, columns=notebook_names)

# Save reports with student names
def save_similarity_report(matrix, title, output_dir, student_map, thresholds):
    """Save heatmap and high-similarity pairs to file with custom thresholds."""
    os.makedirs(output_dir, exist_ok=True)

    # Map notebook names to student names for better labeling
    renamed_matrix = matrix.rename(index=student_map, columns=student_map)

    # Plot heatmap with gridlines and better label spacing
    fig, ax = plt.subplots(figsize=(12, 10))  # Increase figure size
    plt.title(title, fontsize=14, pad=20)     # Title with padding
    cax = ax.imshow(renamed_matrix, cmap='hot', interpolation='nearest')

    # Add gridlines for clarity
    ax.set_xticks(np.arange(len(renamed_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(renamed_matrix.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)  # Hide minor ticks

    # Add labels to axes
    ax.set_xticks(np.arange(len(renamed_matrix.columns)))
    ax.set_yticks(np.arange(len(renamed_matrix.index)))
    ax.set_xticklabels(renamed_matrix.columns, rotation=90, ha="right", fontsize=10)
    ax.set_yticklabels(renamed_matrix.index, fontsize=10)

    # Add colorbar
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    # Save the heatmap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}.png"), dpi=300)
    plt.close()

    # Generate report text file
    report_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"High Similarity Pairs ({title}):\n\n")
        for i in range(len(renamed_matrix)):
            for j in range(i + 1, len(renamed_matrix)):
                similarity_score = renamed_matrix.iloc[i, j]
                threshold = thresholds.get(title, 0.8)  # Default threshold if not found in the dict
                if similarity_score > threshold:
                    student1 = renamed_matrix.index[i]
                    student2 = renamed_matrix.columns[j]
                    line = f"{student1} ↔ {student2}: {similarity_score:.2f}\n"
                    f.write(line)
    print(f"Report saved to: {report_path}")


def run_similarity_tests(notebooks, output_dir, student_map, thresholds, tests=['text', 'levenshtein', 'ast', 'token']):
    """Run selected similarity tests based on provided tests list."""
    # Define the test functions
    similarity_results = {}

    if 'text' in tests:
        print("\nCalculating Text-Based Similarity...")
        similarity_results['text'] = text_based_similarity(notebooks)
        save_similarity_report(similarity_results['text'], "text", output_dir, student_map, thresholds)
    if 'levenshtein' in tests:
        print("\nCalculating Levenshtein-Based Similarity...")
        similarity_results['levenshtein'] = levenshtein_based_similarity(notebooks)
        save_similarity_report(similarity_results['levenshtein'], "levenshtein", output_dir, student_map, thresholds)
    if 'ast' in tests:
        print("\nCalculating AST-Based Similarity...")
        similarity_results['ast'] = ast_based_similarity(notebooks)
        save_similarity_report(similarity_results['ast'], "ast", output_dir, student_map, thresholds)
    if 'token' in tests:
        print("\nCalculating Token-Based Similarity...")
        similarity_results['token'] = token_based_similarity(notebooks)
        save_similarity_report(similarity_results['token'], "token", output_dir, student_map, thresholds)

    return similarity_results

def generate_final_report(similarity_results, student_map, thresholds, output_dir):
    """Generate final report with suspicious counts and heatmap."""
    suspicious_counts = pd.DataFrame(index=student_map.values(), columns=student_map.values(), data=0, dtype=int)
    suspicious_names = pd.DataFrame(index=student_map.values(), columns=student_map.values(), data='', dtype=str)

    # Iterate through results and count the number of tests that flagged the pair
    for test_name, matrix in similarity_results.items():
        matrix = matrix.rename(index=student_map, columns=student_map)
        threshold = thresholds.get(test_name, 0.8)
        for i, row in matrix.iterrows():
            for j, value in row.items():
                if value > threshold and i != j:
                    suspicious_counts.loc[i, j] += 1
                    suspicious_counts.loc[j, i] += 1
                    suspicious_names.loc[i, j] += (test_name + ',')
                    suspicious_names.loc[j, i] += (test_name + ',')

    suspicious_counts = suspicious_counts / 2
    suspicious_names = suspicious_names.map(lambda x: ', '.join(set(x.rstrip(',').split(','))))

    # Plot heatmap of suspicious counts
    plt.figure(figsize=(12, 10))
    plt.title("Suspicious Counts Heatmap", fontsize=14, pad=20)
    cax = plt.imshow(suspicious_counts, cmap='Blues', interpolation='nearest')
    plt.colorbar(cax)
    plt.xticks(ticks=range(len(suspicious_counts.columns)), labels=suspicious_counts.columns, rotation=90, ha="right")
    plt.yticks(ticks=range(len(suspicious_counts.index)), labels=suspicious_counts.index)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "suspicious_counts_heatmap.png"), dpi=300)
    plt.close()

    # Generate final report text file
    rankings = pd.Series()
    for i in range(len(suspicious_counts)):
        for j in range(i + 1, len(suspicious_counts)):
            if suspicious_counts.iloc[i, j] >= 1:
                student1 = suspicious_counts.index[i]
                student2 = suspicious_counts.columns[j]
                rankings[f"{student1} ↔ {student2}"] = f"{suspicious_counts.iloc[i, j]} tests ({suspicious_names.iloc[i, j]})"

    rankings = rankings.sort_values(ascending=False)
    final_report_path = os.path.join(output_dir, "final_suspicious_report.txt")
    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write("Pairs flagged as suspicious in more than one test:\n\n")
        for pair, value in rankings.items():
            line = f"{pair}: {value}\n"
            f.write(line)
    print(f"\nFinal report saved to: {final_report_path}")



def parse_arguments():
    """Parse console arguments."""
    parser = argparse.ArgumentParser(description="Check similarity of ipynb files.")
    parser.add_argument("--csv", help="Path to the students' CSV file")
    parser.add_argument("--output", help="Directory to save reports")
    parser.add_argument("--input", help="Directory to load ipynb from")
    parser.add_argument('--tests', type=str, default='text,ast,token,levenshtein',
                        help="Comma-separated list of similarity tests to run (options: text, ast, token, levenshtein)")
    parser.add_argument('--thresholds', type=str, default='token:0.7,ast:0.95,text:0.6,levenshtein:0.4',
                        help="Comma-separated list of thresholds for each test (e.g., token:0.7, ast:0.95, text:0.6, levenshtein:0.4)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_dir = args.input or "inspected"
    output_dir = args.output or "reports"
    csv_file = args.csv or "classroom_roster.csv"
    tests = args.tests.split(',')
    # Parse thresholds
    thresholds = {}
    for threshold in args.thresholds.split(','):
        test, value = threshold.split(':')
        thresholds[test] = float(value)

    print("Loading student names...")
    students = pd.read_csv(csv_file).dropna(subset=["github_username"])
    student_map = {f"{row['github_username']}_hw2.ipynb": row['identifier'] for _, row in students.iterrows()}

    print("Loading notebooks...")
    notebooks = load_notebooks_from_dir(input_dir)
    print(f"Loaded {len(notebooks)} notebooks.")

    similarity_results = run_similarity_tests(notebooks, output_dir, student_map, thresholds, tests)

    generate_final_report(similarity_results, student_map, thresholds, output_dir)

    print("\n✅ Similarity analysis complete! Check the output directory for reports and heatmaps.")

if __name__ == "__main__":
    main()
