import json
import re
from pathlib import Path
from collections import Counter
import statistics

def analyze_passage_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    article = data['article']
    questions = data['questions']
    answers = data['answers']
    options = data['options']

    answer_lengths = []
    for a, opts in zip(answers, options):
        try:
            answer_index = ord(a) - ord('A')
            if 0 <= answer_index < len(opts):
                answer_lengths.append(len(opts[answer_index]))
            else:
                print(f"Warning: Invalid answer index in file {file_path}")
        except TypeError:
            print(f"Warning: Invalid answer format in file {file_path}")

    return {
        'article_length': len(article),
        'num_questions': len(questions),
        'question_lengths': [len(q) for q in questions],
        'answer_lengths': answer_lengths,
        'words': re.findall(r'\w+', article.lower())
    }

def analyze_selected_passages(folder_path):
    passage_files = list(folder_path.glob('*.txt'))
    
    total_articles = len(passage_files)
    total_chars = 0
    total_questions = 0
    all_question_lengths = []
    all_answer_lengths = []
    all_words = []

    for file in passage_files:
        print(f"Processing file: {file}")
        data = analyze_passage_file(file)
        total_chars += data['article_length']
        total_questions += data['num_questions']
        all_question_lengths.extend(data['question_lengths'])
        all_answer_lengths.extend(data['answer_lengths'])
        all_words.extend(data['words'])

    # Calculate statistics
    avg_article_length = total_chars / total_articles
    avg_questions_per_article = total_questions / total_articles
    avg_question_length = statistics.mean(all_question_lengths) if all_question_lengths else 0
    avg_answer_length = statistics.mean(all_answer_lengths) if all_answer_lengths else 0

    mode_question_length = statistics.mode(all_question_lengths) if all_question_lengths else 0
    mode_answer_length = statistics.mode(all_answer_lengths) if all_answer_lengths else 0

    unique_words = len(set(all_words))

    # Most common words (excluding stop words)
    stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'is', 'are'])
    word_counts = Counter(word for word in all_words if word not in stop_words)
    most_common_words = word_counts.most_common(10)

    # Prepare the output
    output = [
        "Dataset Summary",
        "===============",
        "",
        "Data Structure:",
        "1. Selected Passages:",
        "   Each passage file contains a JSON object with the following structure:",
        "   - 'article': The main text passage",
        "   - 'questions': A list of questions about the passage",
        "   - 'answers': A list of correct answer indices (A, B, C, or D)",
        "   - 'options': A list of lists, each containing 4 answer options for a question",
        "",
        "2. QA Pairs File (race_qa_pairs.txt):",
        "   The file contains multiple QA pairs, each structured as follows:",
        "   - 'Passage-Question-Answer Set X:' (where X is a number)",
        "   - 'Context: [First 200 characters of the passage]...'",
        "   - 'Q: [The question]'",
        "   - 'A: [The correct answer]'",
        "   - An empty line separating each set",
        "",
        "Analysis Results:",
        f"Total passages: {total_articles}",
        f"Total characters: {total_chars}",
        f"Total questions: {total_questions}",
        f"Unique words: {unique_words}",
        "",
        "Average statistics:",
        f"  Article length: {avg_article_length:.2f} characters",
        f"  Questions per article: {avg_questions_per_article:.2f}",
        f"  Question length: {avg_question_length:.2f} characters",
        f"  Answer length: {avg_answer_length:.2f} characters",
        "",
        "Mode lengths:",
        f"  Question: {mode_question_length} characters",
        f"  Answer: {mode_answer_length} characters",
        "",
        "Most common words (excluding stop words):"
    ]
    
    for word, count in most_common_words:
        output.append(f"  {word}: {count}")

    return "\n".join(output)

if __name__ == "__main__":
    data_folder = Path('data')
    selected_passages_folder = data_folder / 'selected_passages'
    output_file = 'docs/dataset_summary.txt'
    
    if selected_passages_folder.exists():
        summary = analyze_selected_passages(selected_passages_folder)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Dataset summary has been saved to {output_file}")
        print("\nSummary:")
        print(summary)
    else:
        print(f"Error: {selected_passages_folder} not found. Please make sure you've generated the dataset first.")