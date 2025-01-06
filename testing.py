import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import spacy
from collections import Counter
import textstat
from lexical_diversity import lex_div as ld

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check if GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load spaCy model for stylistic features and entity grid
nlp = spacy.load("en_core_web_sm")

# Function to encode the essay using BERT and return [CLS] token embeddings
def encode_essay(essay: str):
    encoded_inputs = tokenizer(essay, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract [CLS] token

# Readability statistics function
def readability_scores(text: str) -> dict:
    """Calculate various readability scores for the given text."""
    scores = {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall": textstat.dale_chall_readability_score(text),
    }
    return scores

# Diversity and Stylometry feature extraction functions
def style_features_processing(text: str) -> tuple:
    doc = nlp(text)
    pos_tokens = []
    shape_tokens = []
    LATIN = ["i.e.", "e.g.", "etc.", "c.f.", "et", "al."]

    for word in doc:
        if word.is_punct or word.is_stop or word.text in LATIN:
            pos_target = word.text
            shape_target = word.text
        else:
            pos_target = word.pos_
            shape_target = word.shape_

        pos_tokens.append(pos_target)
        shape_tokens.append(shape_target)

    return " ".join(pos_tokens), " ".join(shape_tokens)

def lex_div_feats_extraction(text: str, features: list):
    preprocessed = preprocess(text)
    result = {}
    for feature in features:
        result[feature] = getattr(ld, feature)(preprocessed)
    return result

def preprocess(text: str) -> list:
    doc = nlp(text)
    return [f"{w.lemma_}_{w.pos_}" for w in doc if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]

def entity_grid(text):
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

    # Extract entities and their roles
    for sent in sentences:
        dict_sentence = {}
        for token in sent:
            if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
                if token.text not in dict_sentence:
                    token_role = role_mappings.get(token.dep_, "x")
                    dict_sentence[token.text] = token_role
        entities.append(dict_sentence)

    # Generate transitions
    for i in range(len(entities) - 1):
        for key, role_1 in entities[i].items():
            role_2 = entities[i + 1].get(key, "-")
            transitions.append(f"{role_1}->{role_2}")

    # Count transitions
    count_transitions = Counter(transitions)
    weighted_transitions = {k: v / (sentences_counter - 1) for k, v in count_transitions.items()}

    # Define all possible transitions
    all_possible_transitions = [
        "o->-", "o->o", "o->s", "o->x", 
        "s->-", "s->o", "s->s", "s->x", 
        "x->-", "x->o", "x->s", "x->x"
    ]

    # Ensure all transitions are present and in order
    ordered_transitions = {transition: weighted_transitions.get(transition, 0.0) for transition in all_possible_transitions}

    return ordered_transitions

# Function to process all features for a single essay
def process_single_essay(essay: str, features_list):
    # Get BERT embeddings
    bert_embeddings = encode_essay(essay)

    # Print the length of BERT embeddings
    bert_flattened = bert_embeddings.flatten()
    print(f"Length of BERT embeddings: {len(bert_flattened)}")

    # Get readability statistics
    readability = readability_scores(essay)
    print(f"Readability feature length: {len(list(readability.values()))}")

    # Get stylistic features
    pos, shape = style_features_processing(essay)
    stylistic_features = lex_div_feats_extraction(essay, features_list)
    print(f"Stylistic feature length: {len(list(stylistic_features.values()))}")

    # Get entity grid transitions
    transitions = entity_grid(essay)
    print(f"Entity grid transitions length: {len(list(transitions.values()))}")

    # Combine all features into a single vector
    all_features = list(readability.values()) + list(stylistic_features.values()) + list(transitions.values())
    print(f"Total feature_vector length: {len(all_features)}")

    # Convert the list of features into a numpy array (flatten it)
    feature_vector = np.array(all_features)

    # Ensure that all feature vectors are of the same length
    # assert len(bert_flattened) + len(feature_vector) == 798, f"Total length mismatch: {len(bert_flattened) + len(feature_vector)}"

    # Concatenate BERT embeddings with the feature vector
    final_vector = np.concatenate([bert_flattened, feature_vector])

    return final_vector

# Example usage with a single essay
essay = "Tamil Nadu’s upcoming elections are poised to be a defining moment in the state’s political history, with major parties like the DMK, AIADMK, and BJP preparing for a fierce contest. The political landscape, historically dominated by the Dravidian parties, has seen the rise of new players, such as Kamal Haasan’s MNM, along with the continued influence of caste-based politics. Key issues shaping the election include economic recovery post-COVID-19, welfare schemes, education, healthcare, and environmental concerns, all of which are vital to the electorate. Alliances, particularly the DMK-Congress coalition and the AIADMK-BJP alliance, will play a critical role in determining the outcome, with smaller parties potentially acting as kingmakers. Additionally, the younger demographic’s increasing involvement, combined with the use of technology in campaigning, signals a shift in how political battles are fought. The outcome of these elections will not only shape Tamil Nadu’s future but also have significant implications on national politics, as voters evaluate parties’ promises against their past performance and ability to address pressing issues like unemployment, development, and social justice."
features_list = ["ttr", "root_ttr", "log_ttr", "maas_ttr", "msttr", "mattr", "hdd", "mtld", "mtld_ma_wrap", "mtld_ma_bid"]
final_features = process_single_essay(essay, features_list)

print(final_features)
# Final feature vector for the essay, ready for classification
print("Final feature vector shape:", final_features.shape)
