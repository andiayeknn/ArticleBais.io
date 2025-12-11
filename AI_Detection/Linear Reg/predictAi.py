import pickle
import os
import re
import string

# caches the model so we don't reload it every time
_CACHED_MODEL = None

def load_model(model_path: str):
    """
    Loads the model from the file. Uses a global cache to prevent 
    re-loading from disk on every single prediction.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        with open(model_path, 'rb') as f:
            _CACHED_MODEL = pickle.load(f)
            
    return _CACHED_MODEL

def preprocess_text(text: str) -> str:
    """
    Standard NLP preprocessing:
    1. Converts to lowercase
    2. Removes punctuation/special characters
    3. Normalizes whitespace (removes newlines/tabs)
    """
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation using regex
    # This regex replaces anything that isn't a letter, number, or whitespace with an empty string
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Remove extra whitespace (e.g., double spaces, newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_ai_probability(text: str) -> float:
    model_path = os.path.join('Models', 'optimized_probabilistic_ai_detection_model.pkl')
    
    # Load model (cached)
    model = load_model(model_path)
    
    # Preprocess the input
    clean_text = preprocess_text(text)
    
    # Predict
    # Note: We pass [clean_text] because sklearn models expect an iterable (list of samples)
    prob = model.predict_proba([clean_text])[0][1]
    
    return float(prob)

if __name__ == '__main__':
    sample_text = input("Enter text to analyze: ")
    try:
        probability = predict_ai_probability(sample_text)
        print(f"AI Probability: {probability:.4f}")
    except Exception as e:
        print(f"Error: {e}")
