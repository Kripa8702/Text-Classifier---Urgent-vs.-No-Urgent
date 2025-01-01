import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

FEEDBACK_FILE = "../csv/feedback.csv"
ORIGINAL_DATA_FILE = "../csv/final_data.csv"

predicted_output = None

# Load the pipelines

with open("../models/svm_pipeline.pkl", "rb") as file:
    svm_pipeline = pickle.load(file)

with open("../models/best_logreg_pipeline.pkl", "rb") as file:
    logreg_tuned_pipeline = pickle.load(file)

with open("../models/stacked_model_3.pkl", "rb") as file:
    stacked_model_3 = pickle.load(file)

try: 
    with open("../models/feedback_trained_model.pkl", "rb") as file:
        feedback_trained_model = pickle.load(file)

except FileNotFoundError:
    feedback_trained_model = None
    


# Feedback loop
def save_feedback(feedback):
    if feedback == "No":
        correct_label = st.selectbox("What is the correct label?", ["Urgent", "Non-Urgent"])
        label = 1 if correct_label == "Urgent" else 0
        if st.button("Submit Feedback"):
            # Save feedback
            feedback_data = pd.DataFrame([[user_input, label]], columns=["text", "label"])
            try:
                existing_data = pd.read_csv(FEEDBACK_FILE)
                # if the text already exists in the feedback file, update the label
                if user_input in existing_data["text"].values:
                    existing_data.loc[existing_data["text"] == user_input, "label"] = label
                    updated_data = existing_data
                else:
                    updated_data = pd.concat([existing_data, feedback_data], ignore_index=True)
            except FileNotFoundError:
                updated_data = feedback_data
            updated_data.to_csv(FEEDBACK_FILE, index=False)
            st.success("Feedback submitted successfully!")
            retrain_model()

def load_feedback():
    original_data = pd.read_csv(ORIGINAL_DATA_FILE)
    try: 
        feedback_data = pd.read_csv(FEEDBACK_FILE)        
        combined_data = pd.concat([original_data, feedback_data], ignore_index=True)
    except FileNotFoundError:
        combined_data = original_data
    return combined_data

def retrain_pipeline(model, tdidf_vectorizer, X, y):
    text_transformer = Pipeline([
    ('tfidf', tdidf_vectorizer)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text_tfidf', text_transformer, 'text')
        ],
    )
    new_pipeline =Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    new_pipeline.fit(X, y)
    return new_pipeline

def retrain_stacked_model(pipeline_svm, pipeline_log, X, y):
    stacked_model = StackingClassifier(
        estimators=[
            ('svm', pipeline_svm),
            ('logreg_tuned', pipeline_log)
        ],
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    )
    
    stacked_model.fit(X, y)
    return stacked_model

def retrain_model():
    data = load_feedback()
    X = data[["text"]]
    y = data["label"]
    
    updated_svm_pipeline = retrain_pipeline(
        SVC(random_state=42, kernel='linear', probability=True),
        TfidfVectorizer(),
        X,
        y
    )

    updated_logreg_pipeline = retrain_pipeline(
        LogisticRegression(random_state=42, max_iter=1000), 
        TfidfVectorizer(ngram_range=(1, 4)),
        X,
        y
    )

    updated_stacked_model_3 = retrain_stacked_model(updated_svm_pipeline, updated_logreg_pipeline, X, y)
    
    with open("../models/feedback_trained_model.pkl", "wb") as file:
        pickle.dump(updated_stacked_model_3, file)
    

# STREAMLIT APP
st.title("Text Classification: Urgent vs Non-Urgent")

# Text input for user
user_input = st.text_area("Enter your message:", placeholder="Type your message here...")

if st.button("Submit", key="submit"):
    if user_input.strip():
         # Convert the input into a DataFrame
        input_df = pd.DataFrame({'text': [user_input]})
        
        
        prediction_stacked_3 = stacked_model_3.predict(input_df)[0]
        output_stack_3 = "Urgent" if prediction_stacked_3 == 1 else "Non-Urgent"
        predicted_output = output_stack_3
        index_stack_3 = 1 if stacked_model_3.predict(input_df)[0] == 1 else 0
        confidence_stack_3 = stacked_model_3.predict_proba(input_df)[0][index_stack_3]
        
        if feedback_trained_model:
            prediction_feedback = feedback_trained_model.predict(input_df)[0]
            output_feedback = "Urgent" if prediction_feedback == 1 else "Non-Urgent"
            index_feedback = 1 if feedback_trained_model.predict(input_df)[0] == 1 else 0
            confidence_feedback = feedback_trained_model.predict_proba(input_df)[0][index_feedback]
        
        # Display table 
        if feedback_trained_model:
            df = pd.DataFrame({
                'Model': [
                    'Stacked Model-3',
                    'Feedback Trained Model',
                    ],
                'Prediction': [
                    output_stack_3,
                    output_feedback,
                ],
                'Confidence': [
                    confidence_stack_3,
                    confidence_feedback,
                ]
            })
        else:
            df = pd.DataFrame({
                'Model': [
                    'Stacked Model-3',
                    ],
                'Prediction': [
                    output_stack_3,
                ],
                'Confidence': [
                    confidence_stack_3,
                ]
            })
        
        st.table(df)

        
                    
    else:
        st.error("Please enter a valid message before submitting.")
        

# Feedback
if predicted_output:
    st.write(f"The model predicted the message as **{predicted_output}**.")
feedback = st.radio("Is the prediction correct?", ["Yes", "No"], key="feedback")
save_feedback(feedback)

