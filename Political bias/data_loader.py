import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_political_bias_data():
    dataset = load_dataset("cajcodes/political-bias")
    df = dataset['train'].to_pandas()
    
    print(f"Loaded {len(df)} samples")
    print(f"Classes: {df['label'].nunique()}")
    print(df['label'].value_counts())
    
    return df


def split_data(df, test_size=0.3, val_size=0.5, random_state=42):
    train_df, temp_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, random_state=random_state, stratify=temp_df['label']
    )
    
    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    return train_df, val_df, test_df
