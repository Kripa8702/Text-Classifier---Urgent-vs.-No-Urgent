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

with open("../models/stacked_model_3.pkl", "rb") as file:
    stacked_model_3 = pickle.load(file)

with open("../models/naive_bayes_model.pkl", "rb") as file:
    naive_bayes_model = pickle.load(file)

with open("../models/svm_pipeline.pkl", "rb") as file:
    svm_pipeline = pickle.load(file)

with open("../models/best_logreg_pipeline.pkl", "rb") as file:
    bst_logreg_pipeline = pickle.load(file)


# try: 
#     with open("../models/feedback_trained_model.pkl", "rb") as file:
#         feedback_trained_model = pickle.load(file)

# except FileNotFoundError:
#     feedback_trained_model = None
    


# Feedback loop
# def save_feedback(feedback):
#     if feedback == "No":
#         correct_label = st.selectbox("What is the correct label?", ["Urgent", "Non-Urgent"])
#         label = 1 if correct_label == "Urgent" else 0
#         if st.button("Submit Feedback"):
#             # Save feedback
#             feedback_data = pd.DataFrame([[user_input, label]], columns=["text", "label"])
#             try:
#                 existing_data = pd.read_csv(FEEDBACK_FILE)
#                 # if the text already exists in the feedback file, update the label
#                 if user_input in existing_data["text"].values:
#                     existing_data.loc[existing_data["text"] == user_input, "label"] = label
#                     updated_data = existing_data
#                 else:
#                     updated_data = pd.concat([existing_data, feedback_data], ignore_index=True)
#             except FileNotFoundError:
#                 updated_data = feedback_data
#             updated_data.to_csv(FEEDBACK_FILE, index=False)
#             st.success("Feedback submitted successfully!")
#             retrain_model()

# def load_feedback():
#     original_data = pd.read_csv(ORIGINAL_DATA_FILE)
#     try: 
#         feedback_data = pd.read_csv(FEEDBACK_FILE)        
#         combined_data = pd.concat([original_data, feedback_data], ignore_index=True)
#     except FileNotFoundError:
#         combined_data = original_data
#     return combined_data

# def retrain_pipeline(model, tdidf_vectorizer, X, y):
#     text_transformer = Pipeline([
#     ('tfidf', tdidf_vectorizer)
#     ])

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('text_tfidf', text_transformer, 'text')
#         ],
#     )
#     new_pipeline =Pipeline([
#         ('preprocessor', preprocessor),
#         ('classifier', model)
#     ])

#     new_pipeline.fit(X, y)
#     return new_pipeline

# def retrain_stacked_model(pipeline_svm, pipeline_log, X, y):
#     stacked_model = StackingClassifier(
#         estimators=[
#             ('svm', pipeline_svm),
#             ('logreg_tuned', pipeline_log)
#         ],
#         final_estimator=LogisticRegression(random_state=42, max_iter=1000),
#     )
    
#     stacked_model.fit(X, y)
#     return stacked_model

# def retrain_model():
#     data = load_feedback()
#     X = data[["text"]]
#     y = data["label"]
    
#     updated_svm_pipeline = retrain_pipeline(
#         SVC(random_state=42, kernel='linear', probability=True),
#         TfidfVectorizer(),
#         X,
#         y
#     )

#     updated_logreg_pipeline = retrain_pipeline(
#         LogisticRegression(random_state=42, max_iter=1000), 
#         TfidfVectorizer(ngram_range=(1, 4)),
#         X,
#         y
#     )

#     updated_stacked_model_3 = retrain_stacked_model(updated_svm_pipeline, updated_logreg_pipeline, X, y)
    
#     with open("../models/feedback_trained_model.pkl", "wb") as file:
#         pickle.dump(updated_stacked_model_3, file)

def get_confidence_percentage(confidence):
    return f"{confidence * 100:.2f}%"

def load_output_data():
    try:
        output_data = pd.read_csv("../csv/output.csv")
    except FileNotFoundError:
        output_data = pd.DataFrame(columns=["Text", "SVM", "LogReg", "SVM + LogReg", "Naive Bayes"])

    return output_data  

def store_output(text, predictions):
    output_data = load_output_data()
    output_data = pd.concat([output_data, pd.DataFrame([{
        "Text": text,
        "SVM": predictions["SVM"],
        "LogReg": predictions["LogReg"],
        "SVM + LogReg": predictions["SVM + LogReg"],
        "Naive Bayes": predictions["Naive Bayes"]
    }])], ignore_index=True)
    output_data.to_csv("../csv/output.csv", index=False)


# STREAMLIT APP
st.title("Text Classification: Urgent vs Non-Urgent")

st.write(
    """
    This is a simple text classification app that predicts whether a message is **Urgent** or **Non-Urgent**.
    """
)

st.write("")

# Text input for user
user_input = st.text_area("Enter your message:", placeholder="Type your message here...")

if st.button("Submit", key="submit"):
    if user_input.strip():
         # Convert the input into a DataFrame
        input_df = pd.DataFrame({'text': [user_input]})
        
        # SVM + LogReg Model
        prediction_svm = svm_pipeline.predict(input_df)[0]
        output_svm = "Urgent" if prediction_svm == 1 else "Non-Urgent"
        confidence_svm = svm_pipeline.predict_proba(input_df)[0][prediction_svm]

        prediction_logreg = bst_logreg_pipeline.predict(input_df)[0]
        output_logreg = "Urgent" if prediction_logreg == 1 else "Non-Urgent"
        confidence_logreg = bst_logreg_pipeline.predict_proba(input_df)[0][prediction_logreg]

        # Stacked Model
        prediction_stacked_3 = stacked_model_3.predict(input_df)[0]
        output_stack_3 = "Urgent" if prediction_stacked_3 == 1 else "Non-Urgent"
        confidence_stack_3 = stacked_model_3.predict_proba(input_df)[0][prediction_stacked_3]

        # Naive Bayes Model
        prediction_naive_bayes = naive_bayes_model.predict(input_df)[0]
        output_naive_bayes = "Urgent" if prediction_naive_bayes == 1 else "Non-Urgent"
        confidence_naive_bayes = naive_bayes_model.predict_proba(input_df)[0][prediction_naive_bayes]
        
        # Store the output
        store_output(user_input, {
            "SVM": output_svm,
            "LogReg": output_logreg,
            "SVM + LogReg": output_stack_3,
            "Naive Bayes": output_naive_bayes
        })
       
        df = pd.DataFrame({
            'Model': [
                'SVM',
                'LogReg',
                'SVM + LogReg',
                'Naive Bayes'
                ],
            'Prediction': [
                output_svm,
                output_logreg,
                output_stack_3,
                output_naive_bayes
            ],
            'Confidence': [
                get_confidence_percentage(confidence_svm),
                get_confidence_percentage(confidence_logreg),
                get_confidence_percentage(confidence_stack_3),
                get_confidence_percentage(confidence_naive_bayes)
            ]
        })
        st.write("### Predictions")
        st.table(df)
            
    else:
        st.error("Please enter a valid message before submitting.")

# DISPLAY OUTPUT
output_data = load_output_data()
if not output_data.empty:
    st.write("### Previous Predictions")
    st.table(output_data)

# Feedback
# if predicted_output:
#     st.write(f"The model predicted the message as **{predicted_output}**.")
# feedback = st.radio("Is the prediction correct?", ["Yes", "No"], key="feedback")
# save_feedback(feedback)
