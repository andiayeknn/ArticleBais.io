import pickle


def load_model():
    with open('models/bias_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_bias(text, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
    
    if isinstance(text, str):
        text = [text]
    
    X = vectorizer.transform(text)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    labels = {
        0: 'Highly Conservative',
        1: 'Conservative',
        2: 'Moderate',
        3: 'Liberal',
        4: 'Highly Liberal'
    }
    
    results = []
    for pred, probs in zip(predictions, probabilities):
        results.append({
            'label': int(pred),
            'bias': labels[pred],
            'confidence': float(max(probs)),
            'probabilities': {labels[i]: float(p) for i, p in enumerate(probs)}
        })
    
    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    examples = [
        "The government should cut taxes to stimulate economic growth.",
        "We need more social programs to support vulnerable communities.",
        "Both sides have valid points on economic policy."
    ]
    
    for text in examples:
        result = predict_bias(text)
        print(f"\n{text}")
        print(f"→ {result['bias']} ({result['confidence']:.1%})")
