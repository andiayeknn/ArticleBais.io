import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json


def evaluate_model(model, X_test, y_test, save_dir='../results/'):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    labels = ['Highly Conservative', 'Conservative', 'Moderate', 'Liberal', 'Highly Liberal']
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    
    print("\n" + classification_report(y_test, y_pred, target_names=labels))
    
    with open(f'{save_dir}classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['HC', 'C', 'M', 'L', 'HL'],
                yticklabels=['HC', 'C', 'M', 'L', 'HL'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}confusion_matrix.png', dpi=300)
    
    print(f"\nSaved to {save_dir}")
    
    return accuracy, report, cm
