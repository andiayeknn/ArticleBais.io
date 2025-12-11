import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_loader import load_political_bias_data, split_data


def vectorize_data(train_df, val_df, test_df):
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(val_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']
    
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


def train_naive_bayes(X_train, y_train):
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, tune=False):
    if tune:
        param_grid = {
            'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs', 'liblinear']
        }
        
        grid = GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=42),
            param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV: {grid.best_score_:.4f}")
        
        return grid.best_estimator_
    else:
        model = LogisticRegression(C=5.0, class_weight='balanced', max_iter=2000, random_state=42)
        model.fit(X_train, y_train)
        return model


def compare_models(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*60)
    print("NAIVE BAYES")
    print("="*60)
    nb_model = train_naive_bayes(X_train, y_train)
    
    nb_train = nb_model.score(X_train, y_train)
    nb_val = nb_model.score(X_val, y_val)
    nb_test = nb_model.score(X_test, y_test)
    
    print(f"Train: {nb_train:.4f} | Val: {nb_val:.4f} | Test: {nb_test:.4f}")
    
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    lr_model = train_logistic_regression(X_train, y_train, tune=True)
    
    lr_train = lr_model.score(X_train, y_train)
    lr_val = lr_model.score(X_val, y_val)
    lr_test = lr_model.score(X_test, y_test)
    
    print(f"Train: {lr_train:.4f} | Val: {lr_val:.4f} | Test: {lr_test:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-"*60)
    print(f"{'Naive Bayes':<25} {nb_train:.4f}     {nb_val:.4f}     {nb_test:.4f}")
    print(f"{'Logistic Regression':<25} {lr_train:.4f}     {lr_val:.4f}     {lr_test:.4f}")
    
    if lr_test > nb_test:
        print(f"\nWinner: Logistic Regression (+{(lr_test-nb_test):.1%})")
        return lr_model
    else:
        print(f"\nWinner: Naive Bayes")
        return nb_model


def save_artifacts(model, vectorizer):
    os.makedirs('../models', exist_ok=True)

    with open('models/bias_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("\nSaved model and vectorizer to ../models/")


def main():
    print("="*60)
    print("POLITICAL BIAS CLASSIFIER - TRAINING")
    print("="*60)
    
    df = load_political_bias_data()
    train_df, val_df, test_df = split_data(df)
    
    print("\n" + "="*60)
    print("VECTORIZATION")
    print("="*60)
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = vectorize_data(
        train_df, val_df, test_df
    )
    
    best_model = compare_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    save_artifacts(best_model, vectorizer)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
