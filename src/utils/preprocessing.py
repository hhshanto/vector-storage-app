import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def load_passages(folder_path):
    passages = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            print(f"Loading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                try:
                    passage = json.loads(content)
                    passages.append(passage)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file_path}")
    return passages

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_passage(passage):
    # Clean the text
    cleaned_text = clean_text(passage['article'])
    
    # Tokenize
    tokens = tokenize(cleaned_text)
    
    # Remove stopwords
    tokens_without_stopwords = remove_stopwords(tokens)
    
    # Lemmatize
    lemmatized_tokens = lemmatize(tokens_without_stopwords)
    
    # Join tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    
    # Update the passage dictionary
    passage['processed_article'] = processed_text
    
    return passage

def preprocess_all_passages(folder_path):
    passages = load_passages(folder_path)
    preprocessed_passages = [preprocess_passage(passage) for passage in passages]
    return preprocessed_passages

def save_preprocessed_passages(preprocessed_passages, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_passages, f, indent=2)
    print(f"Preprocessed passages saved to {output_file}")

if __name__ == "__main__":
    # Use paths relative to the project root instead of the script location
    folder_path = "./data/selected_passages"
    output_file = "./data/preprocessed_passages.json"
    
    print(f"Looking for text files in: {os.path.abspath(folder_path)}")
    files = os.listdir(folder_path)
    print(f"Files found: {files}")

    preprocessed_passages = preprocess_all_passages(folder_path)
    
    if preprocessed_passages:
        print(f"Preprocessed {len(preprocessed_passages)} passages.")
        print("\nFirst preprocessed passage:")
        print(json.dumps(preprocessed_passages[0], indent=2))
        
        save_preprocessed_passages(preprocessed_passages, output_file)
    else:
        print("No passages were preprocessed. Check if the files are in the correct format.")