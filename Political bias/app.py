import streamlit as st
import joblib
import pickle
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Article Analyzer",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource
def load_models():
    try:
        bias_model = pickle.load(open('models/bias_classifier.pkl', 'rb'))
        bias_vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        ai_model = joblib.load('models/optimized_probabilistic_ai_detection_model.pkl')
        ai_vectorizer = joblib.load('models/tfidf_vectorizer_for_probabilistic_model.pkl')

        return bias_model, bias_vectorizer, ai_model, ai_vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def predict_bias(text, model, vectorizer):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    labels = {
        0: 'Highly Conservative',
        1: 'Conservative',
        2: 'Moderate',
        3: 'Liberal',
        4: 'Highly Liberal'
    }

    return labels[prediction], probabilities, prediction


def predict_ai(text, model, vectorizer):
    X = vectorizer.transform([text])
    probability = model.predict_proba(X)[0][1]
    prediction = "AI-Generated" if probability >= 0.5 else "Human-Written"

    return prediction, probability


def create_bias_gauge(label, confidence):
    colors = {
        'Highly Conservative': '#8B0000',
        'Conservative': '#DC143C',
        'Moderate': '#FFD700',
        'Liberal': '#4169E1',
        'Highly Liberal': '#00008B'
    }

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{label}</b>", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': colors.get(label, '#FFD700')},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f0f0'},
                {'range': [50, 75], 'color': '#e0e0e0'},
                {'range': [75, 100], 'color': '#d0d0d0'}
            ],
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_ai_gauge(prediction, probability):
    color = '#DC143C' if prediction == "AI-Generated" else '#4169E1'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100 if prediction == "AI-Generated" else (1 - probability) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{prediction}</b>", 'font': {'size': 24}},
        number={'suffix': "%", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f0f0'},
                {'range': [50, 75], 'color': '#e0e0e0'},
                {'range': [75, 100], 'color': '#d0d0d0'}
            ],
        }
    ))

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


st.title("📰 Article Analyzer")
st.markdown("Analyze articles for **political bias** and **AI-generated content**")

bias_model, bias_vectorizer, ai_model, ai_vectorizer = load_models()

if bias_model is None or ai_model is None:
    st.error("Failed to load models. Please ensure model files are in the 'models/' directory.")
    st.stop()

st.markdown("---")

article_text = st.text_area(
    "Paste your article here:",
    height=300,
    placeholder="Enter the article text you want to analyze..."
)

if st.button("🔍 Analyze Article", type="primary"):
    if not article_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🗳️ Political Bias")
                bias_label, bias_probs, bias_idx = predict_bias(
                    article_text, bias_model, bias_vectorizer
                )

                confidence = bias_probs[bias_idx]

                st.plotly_chart(
                    create_bias_gauge(bias_label, confidence),
                    use_container_width=True
                )

                with st.expander("See detailed probabilities"):
                    labels = ['Highly Conservative', 'Conservative', 'Moderate',
                              'Liberal', 'Highly Liberal']
                    for label, prob in zip(labels, bias_probs):
                        st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

            with col2:
                st.subheader("🤖 AI Detection")
                ai_label, ai_prob = predict_ai(
                    article_text, ai_model, ai_vectorizer
                )

                st.plotly_chart(
                    create_ai_gauge(ai_label, ai_prob),
                    use_container_width=True
                )

                with st.expander("See detailed probabilities"):
                    st.progress(1 - ai_prob, text=f"Human-Written: {(1 - ai_prob) * 100:.1f}%")
                    st.progress(ai_prob, text=f"AI-Generated: {ai_prob * 100:.1f}%")

            st.markdown("---")

            col3, col4 = st.columns(2)

            with col3:
                st.metric(
                    label="Political Leaning",
                    value=bias_label,
                    delta=f"{confidence * 100:.1f}% confidence"
                )

            with col4:
                st.metric(
                    label="Authorship",
                    value=ai_label,
                    delta=f"{max(ai_prob, 1 - ai_prob) * 100:.1f}% confidence"
                )

st.markdown("---")
st.markdown("""
### 📊 Model Information
- **Political Bias Model:** Logistic Regression (80.8% accuracy)
  - Dataset: 657 political statements
  - Classes: 5 (Highly Conservative → Highly Liberal)

- **AI Detection Model:** Logistic Regression (Probabilistic)
  - Dataset: 695k samples (Human vs AI-generated)
  - Binary classification with probability scores
""")

st.markdown("---")
st.caption("Built for ArticleBais.io | Models: Political Bias Classifier + AI Detector")