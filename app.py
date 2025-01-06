from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel
import spacy
from collections import Counter
import textstat
from lexical_diversity import lex_div as ld

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Check if GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Feed-Forward Neural Network (FFNN)
class CombinedModel(nn.Module):
    def __init__(self, input_dim):
        super(CombinedModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout to prevent overfitting
            
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 2)  # Output layer with 2 classes (human vs LLM)
        )
    
    def forward(self, combined_input):
        logits = self.network(combined_input)
        return logits

# Load the saved TorchScript model
model_path = "best_model_combination.pt"  # Ensure this file exists in the root directory
model = torch.load(model_path, map_location=device)
model.eval()

# Load spaCy model for stylistic features and entity grid
nlp = spacy.load("en_core_web_sm")

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # To handle cross-origin requests if needed

# Function to encode the essay using BERT and return [CLS] token embeddings
def encode_essay(essay: str):
    print("Encoding essay with BERT...")
    encoded_inputs = tokenizer(essay, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = bert_model(**encoded_inputs)
    bert_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract [CLS] token
    print(f"BERT Embedding shape: {bert_embedding.shape}")
    return bert_embedding

# Readability statistics function
def readability_scores(text: str) -> dict:
    print("Calculating readability scores...")
    scores = {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
    }
    print(f"Readability scores: {scores}")
    return scores

# Diversity and Stylometry feature extraction functions
def style_features_processing(text: str) -> tuple:
    print("Processing stylistic features...")
    doc = nlp(text)
    pos_tokens = [token.pos_ for token in doc if not token.is_punct]
    shape_tokens = [token.shape_ for token in doc if not token.is_punct]

    print(f"POS tokens: {pos_tokens[:10]}")  # Print first 10 tokens for brevity
    print(f"Shape tokens: {shape_tokens[:10]}")
    return " ".join(pos_tokens), " ".join(shape_tokens)

def lex_div_feats_extraction(text: str, features: list):
    print("Extracting lexical diversity features...")
    preprocessed = preprocess(text)
    result = {}
    for feature in features:
        result[feature] = getattr(ld, feature)(preprocessed)
    print(f"Lexical diversity features: {result}")
    return result

def preprocess(text: str) -> list:
    doc = nlp(text)
    preprocessed_text = [f"{w.lemma_}_{w.pos_}" for w in doc if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]
    print(f"Preprocessed text: {preprocessed_text[:10]}")  # Print first 10 words for brevity
    return preprocessed_text

def entity_grid(text):
    print("Calculating entity grid...")
    transitions = list()
    entities = list()
    sentences_counter = 0
    doc = nlp(text)
    sentences = [sent for sent in doc.sents]
    sentences_counter += len(sentences)

    role_mappings = {
        "nsubj": "s",
        "dobj": "o",
        "pobj": "o"
    }

    for sent in sentences:
        dict_sentence = {}
        for token in sent:
            if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
                if token.text not in dict_sentence:
                    token_role = role_mappings.get(token.dep_, "x")
                    dict_sentence[token.text] = token_role
        entities.append(dict_sentence)

    for i in range(len(entities) - 1):
        for key, role_1 in entities[i].items():
            role_2 = entities[i + 1].get(key, "-")
            transitions.append(f"{role_1}->{role_2}")

    count_transitions = Counter(transitions)
    weighted_transitions = {k: v / (sentences_counter - 1) for k, v in count_transitions.items()}
    print(f"Entity grid transitions: {weighted_transitions}")
    return weighted_transitions

@app.route('/predict', methods=['POST'])
def predict():
    # Get the essay from the POST request
    data = request.get_json()
    essay = data.get('essay', '')

    # Perform feature extraction on the essay
    bert_embedding = encode_essay(essay)
    readability_features = readability_scores(essay)
    pos_tokens, shape_tokens = style_features_processing(essay)
    lexical_features = lex_div_feats_extraction(essay, ['TTR', 'MATTR'])
    entity_grid_features = entity_grid(essay)

    # Combine all features into one vector (this needs to match the input_dim of the model)
    combined_input = np.hstack([bert_embedding, list(readability_features.values()), 
                                list(lexical_features.values()), list(entity_grid_features.values())])
    
    # Convert the combined input into a tensor and move it to the device
    combined_input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(combined_input_tensor)
        prediction = torch.argmax(nn.functional.softmax(output, dim=1), dim=1).item()

    # Map prediction to human-readable format
    result = 'Human-written' if prediction == 0 else 'LLM-generated'
    
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(debug=True)
