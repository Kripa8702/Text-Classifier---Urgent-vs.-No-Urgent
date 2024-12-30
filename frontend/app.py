import streamlit as st
import pandas as pd
import pickle

# Load the pipelines
with open("../models/svm_pipeline.pkl", "rb") as file:
    pipeline_svm = pickle.load(file)
    
with open("../models/best_svm_pipeline.pkl", "rb") as file:
    pipeline_svm_best = pickle.load(file)
    
with open("../models/logreg_pipeline.pkl", "rb") as file:
    pipeline_logreg = pickle.load(file)
    
with open("../models/best_logreg_pipeline.pkl", "rb") as file:
    pipeline_logreg_best = pickle.load(file)
    
with open("../models/stacked_model.pkl", "rb") as file:
    stacked_model = pickle.load(file)
    


# Streamlit app
st.title("Text Classification: Urgent vs Non-Urgent")

# Text input for user
user_input = st.text_area("Enter your message:", placeholder="Type your message here...")

if st.button("Submit"):
    if user_input.strip():
         # Convert the input into a DataFrame
        input_df = pd.DataFrame({'text': [user_input]})
        
        # Use the pipeline to predict
        prediction_svm = pipeline_svm.predict(input_df)[0]
        prediction__svm_best = pipeline_svm_best.predict(input_df)[0]
        prediction_logreg = pipeline_logreg.predict(input_df)[0]
        prediction_logreg_best = pipeline_logreg_best.predict(input_df)[0]
        prediction_stacked = stacked_model.predict(input_df)[0]
        
        # Display the result
        result = "Urgent" if prediction_svm == 1 else "Non-Urgent"
        result_best = "Urgent" if prediction__svm_best == 1 else "Non-Urgent"
        result_logreg = "Urgent" if prediction_logreg == 1 else "Non-Urgent"
        result_logreg_best = "Urgent" if prediction_logreg_best == 1 else "Non-Urgent"
        result_stacked = "Urgent" if prediction_stacked == 1 else "Non-Urgent"
        
        
        # Display table 
        df = pd.DataFrame({
            'Model': ['SVM', 'Best SVM', 'Logistic Regression', 'Best Logistic Regression', 'Stacked Model'],
            'Prediction': [
                result, 
                result_best,
                result_logreg, 
                result_logreg_best,
                result_stacked
            ]
        })
        
        st.table(df)
    else:
        st.error("Please enter a valid message before submitting.")

