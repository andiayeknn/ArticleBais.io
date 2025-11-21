# Media Bias and AI Content Detection

Media Bias and AI Content Detection
Analyzed the correlation between political bias and AI-generated content by developing and comparing Naive Bayes and Random Forest classification models. This project, conducted within the AI4ALL Ignite accelerator, involved processing four distinct datasets to engineer a transparency framework for modern media consumption.

## Problem Statement <!--- do not change this line -->

With the exponential rise of generative AI in journalism, distinguishing between human-written and AI-generated articles—and detecting their inherent political leanings—has become a critical challenge. This project addresses the urgent need for media literacy tools that can identify potential manipulation and bias in news consumption, ensuring users can navigate an increasingly automated information landscape with transparency and confidence.

## Key Results <!--- do not change this line -->

Unified Dataset Creation: Successfully aggregated, cleaned, and stratified four disparate sources (Kaggle Media Bias, Media Bias Dataset, AI vs Human Text, and AI Generated News) into a unified dataset with consistent labeling.
Dual-Model Classification System:

Naive Bayes: Implemented for political bias and AI detection using TF-IDF and Count Vectorizers to establish baseline probabilities.

Random Forest: Implemented to analyze feature importance and handle complex interactions between text features.

Correlation Discovery: Investigated the research question "Can increasing political bias be linked to AI-generated articles?" by analyzing the statistical correlation between AI-detection confidence scores and bias intensity.

Ethics & Transparency Framework: Developed a comprehensive framework for responsible model interpretation, including uncertainty quantification and error analysis guidelines to mitigate algorithmic bias.


## Methodologies <!--- do not change this line -->

To accomplish this, we engineered a comprehensive NLP pipeline using Python. We designed a rigorous data cleaning process to remove HTML tags and standardize formatting across multiple datasets using pandas. We then performed feature engineering to extract text characteristics (vocabulary richness, sentence structure) and utilized Scikit-learn to train and tune Naive Bayes and Random Forest models. Finally, we conducted a statistical correlation analysis to quantify the relationship between AI generation probability and political bias scores.

## Data Sources <!--- do not change this line -->

Media Bias Dataset

AI vs Human Text Dataset

AI Generated News Dataset

## Technologies Used <!--- do not change this line -->

pandas (Data cleaning, merging, and manipulation)

Scikit-learn (Naive Bayes, Random Forest, Hyperparameter tuning)

Matplotlib / Seaborn (Visualization for EDA and Correlation Analysis)

NLTK / Spacy (Text preprocessing and feature extraction)


## Authors <!--- do not change this line -->

Amadou Ndiaye (Data Cleaning & Integration, Random Forest Implementation, Error Analysis)

Aisha Nishat (Dataset Acquisition, Naive Bayes Implementation, Correlation Analysis)

Anna Pham (Feature Engineering, Model Evaluation, Ethics & Transparency Framework)
- *John Doe ([john.doe@example.com](mailto:john.doe@example.com))*
- *Jane Smith ([jane.smith@example.com](mailto:jane.smith@example.com))*
