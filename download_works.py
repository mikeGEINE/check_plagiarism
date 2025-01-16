import os
import requests
import pandas as pd
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Default configuration values
DEFAULT_MAX_WORKERS = 10
DEFAULT_OUTPUT_DIR = "homeworks"

# GitHub API headers
HEADERS = {}

def fetch_hw2_file(repo_name, username, filename, output_dir, headers):
    """Fetch `filename` from a student's repo."""
    url = f"https://api.github.com/repos/{headers['organization']}/{repo_name}/contents/{filename}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        content = response.json().get('content', '')
        # Extract download_url from the API response
        download_url = response.json().get('download_url')
        if content:
            # Decode base64 content and save file
            file_content = base64.b64decode(content)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{username}_{filename}")
            with open(output_path, "wb") as f:
                f.write(file_content)
            print(f"✅ Downloaded {filename} for {username}")
            return True
        elif download_url:
             # Fetch the raw file content via download_url
            raw_response = requests.get(download_url)
            if raw_response.status_code == 200:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{username}_{filename}")
                with open(output_path, "wb") as f:
                    f.write(raw_response.content)
                print(f"✅ Downloaded {filename} for {username}")
                return True
        else:
            print(f"⚠️ {filename} not found in {repo_name}")
            return False
    else:
        print(f"❌ Failed to fetch {repo_name}: {response.status_code} {response.reason}")
        return False

def parse_arguments():
    """Parse console arguments."""
    parser = argparse.ArgumentParser(description="Fetch students' files from GitHub.")
    parser.add_argument("--token", help="GitHub Personal Access Token")
    parser.add_argument("--organization", help="GitHub organization name")
    parser.add_argument("--csv", help="Path to the students' CSV file")
    parser.add_argument("--assignment", type=str, help="Assignment identifier")
    parser.add_argument("--filename", type=int, help="File to fetch")
    parser.add_argument("--output", help="Directory to save fetched files")
    parser.add_argument("--workers", type=int, help="Number of concurrent threads")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration with priority: CLI > Env vars > .env defaults
    github_token = args.token or os.getenv("GITHUB_TOKEN")
    organization = args.organization or os.getenv("ORGANIZATION")
    csv_file = args.csv or os.getenv("CSV_FILE", "classroom_roster.csv")
    filename = args.filename or os.getenv("FILENAME")
    output_dir = args.output or os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    max_workers = args.workers or int(os.getenv("MAX_WORKERS", DEFAULT_MAX_WORKERS))
    assignment = args.assignment or os.getenv("ASSIGNMENT", "EXAMPLE")

    if not all([github_token, organization, csv_file, filename]):
        print("❗ Error: GITHUB_TOKEN, ORGANIZATION, FILENAME, and CSV_FILE must be provided!")
        exit(1)

    # Prepare GitHub API headers
    global HEADERS
    HEADERS = {
        "Authorization": f"Bearer {github_token}",
        "organization": organization,
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    # Read the CSV file
    students = pd.read_csv(csv_file)
    students = students.dropna(subset=['github_username'])

    # Fetch files in parallel
    print(f"🚀 Starting to fetch {filename} files for {len(students)} students...")
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in students.iterrows():
            username = row['github_username']
            repo_name = f"{assignment}-{username}"
            tasks.append(executor.submit(fetch_hw2_file, repo_name, username, filename, output_dir, HEADERS))

        # Collect results
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"❗ Error during execution: {e}")

    print("✅ All tasks completed!")

if __name__ == "__main__":
    main()
