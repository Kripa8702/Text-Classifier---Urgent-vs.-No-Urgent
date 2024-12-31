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
    
# Stacked models
with open("../models/stacked_model_1.pkl", "rb") as file:
    stacked_model_1 = pickle.load(file)

with open("../models/stacked_model_2.pkl", "rb") as file:
    stacked_model_2 = pickle.load(file)

with open("../models/stacked_model_3.pkl", "rb") as file:
    stacked_model_3 = pickle.load(file)

with open("../models/stacked_model_4.pkl", "rb") as file:
    stacked_model_4 = pickle.load(file)


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

        # Stacked model
        prediction_stacked_1 = "Urgent" if stacked_model_1.predict(input_df)[0] == 1 else "Non-Urgent"
        index_stack_1 = 1 if stacked_model_1.predict(input_df)[0] == 1 else 0

        prediction_stacked_2 = "Urgent" if stacked_model_2.predict(input_df)[0] == 1 else "Non-Urgent"
        index_stack_2 = 1 if stacked_model_2.predict(input_df)[0] == 1 else 0

        prediction_stacked_3 = "Urgent" if stacked_model_3.predict(input_df)[0] == 1 else "Non-Urgent"
        index_stack_3 = 1 if stacked_model_3.predict(input_df)[0] == 1 else 0

        prediction_stacked_4 = "Urgent" if stacked_model_4.predict(input_df)[0] == 1 else "Non-Urgent"
        index_stack_4 = 1 if stacked_model_4.predict(input_df)[0] == 1 else 0

        # Confidence scores
        confidence_stack_1 = stacked_model_1.predict_proba(input_df)[0][index_stack_1]
        confidence_stack_2 = stacked_model_2.predict_proba(input_df)[0][index_stack_2]
        confidence_stack_3 = stacked_model_3.predict_proba(input_df)[0][index_stack_3]
        confidence_stack_4 = stacked_model_4.predict_proba(input_df)[0][index_stack_4]
        
        # Display table 
        df = pd.DataFrame({
            'Model': [
                'Stacked Model-1',
                'Stacked Model-2',
                'Stacked Model-3',
                'Stacked Model-4'
                ],
            'Prediction': [
                prediction_stacked_1,
                prediction_stacked_2,
                prediction_stacked_3,
                prediction_stacked_4
            ],
            'Confidence': [
                confidence_stack_1,
                confidence_stack_2,
                confidence_stack_3,
                confidence_stack_4
            ]
        })
        
        st.table(df)
    else:
        st.error("Please enter a valid message before submitting.")

