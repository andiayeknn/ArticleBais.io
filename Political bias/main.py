from src.train_model import main as train
from src.predict import predict_bias
import sys


def demo():
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    
    examples = [
        "Taxes should be reduced to promote business growth.",
        "Healthcare is a human right that government must provide.",
        "We need balanced policies that consider both sides."
    ]
    
    for text in examples:
        result = predict_bias(text)
        print(f"\n{text}")
        print(f"→ {result['bias']} ({result['confidence']:.0%})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        demo()
