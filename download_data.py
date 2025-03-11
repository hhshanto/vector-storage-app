import json
import random
from pathlib import Path
import requests
import tarfile
import io
import os
import shutil

def download_race_data(url, save_path):
    print(f"Downloading data from {url}")
    response = requests.get(url)
    response.raise_for_status()
    print(f"Download complete. Extracting to {save_path}")
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        tar.extractall(path=save_path)
    print(f"Extraction complete. Data saved to {save_path}")

def load_and_copy_race_data(folder_path, subset='train/middle', num_passages=10, target_folder='selected_passages'):
    data = []
    subset_path = folder_path / 'RACE' / subset
    target_path = folder_path / target_folder
    target_path.mkdir(exist_ok=True)
    
    print(f"Looking for data in {subset_path}")
    if not subset_path.exists():
        print(f"Error: {subset_path} does not exist")
        return data

    for file_path in subset_path.glob('*.txt'):
        print(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage_data = json.load(f)
                data.append(passage_data)
            
            # Copy the file to the target folder
            shutil.copy2(file_path, target_path)
            
            if len(data) >= num_passages:
                break
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    print(f"Loaded and copied {len(data)} passages to {target_path}")
    return data

def prepare_qa_pairs(data):
    qa_pairs = []
    for passage in data:
        context = passage['article']
        for question, answer, options in zip(passage['questions'], passage['answers'], passage['options']):
            answer_index = ord(answer) - ord('A')  # Convert letter to index
            qa_pairs.append({
                'context': context,
                'question': question,
                'answer': options[answer_index]
            })
    print(f"Prepared {len(qa_pairs)} question-answer pairs")
    return qa_pairs

def save_qa_pairs_to_file(qa_pairs, file_path):
    print(f"Saving {len(qa_pairs)} QA pairs to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, qa in enumerate(qa_pairs, 1):
            f.write(f"Passage-Question-Answer Set {i}:\n")
            f.write(f"Context: {qa['context'][:200]}...\n")  # First 200 characters of context
            f.write(f"Q: {qa['question']}\n")
            f.write(f"A: {qa['answer']}\n")
            f.write("\n")  # Empty line between sets
    print(f"Save complete. File size: {os.path.getsize(file_path)} bytes")

# Path to your data folder
data_folder = Path('data')
data_folder.mkdir(exist_ok=True)

# URL for RACE dataset
race_url = "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"

# Download and extract the RACE data
race_folder = data_folder / 'RACE'
download_race_data(race_url, data_folder)

# Load a subset of the RACE data (10 passages from the middle school subset) and copy to a new folder
selected_folder = 'selected_passages'
race_data = load_and_copy_race_data(data_folder, subset='train/middle', num_passages=100, target_folder=selected_folder)

# Delete the original RACE folder
shutil.rmtree(race_folder)
print(f"Deleted original RACE folder: {race_folder}")

# Prepare question-answer pairs
qa_pairs = prepare_qa_pairs(race_data)

# Save question-answer pairs to a text file
output_file = data_folder / 'race_qa_pairs.txt'
save_qa_pairs_to_file(qa_pairs, output_file)

print(f"Selected passages have been copied to {data_folder / selected_folder}")
print(f"Generated passage-question-answer sets have been saved to {output_file}")

# Print some passage-question-answer sets to console
for i, qa in enumerate(qa_pairs[:3], 1):  # Print first 3 sets
    print(f"\nPassage-Question-Answer Set {i}:")
    print(f"Context: {qa['context'][:100]}...")  # Print first 100 characters of context
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")