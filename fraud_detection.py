import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import base64
from fpdf import FPDF
import urllib.parse

@st.cache
def load_data():
    df = pd.read_csv("D:/Prayag Files/TIET/Extras/hackathons/Hack4change/Round 2/Fraud.csv")
    df = df[df['amount'] <= 80000000]
    df = df[df['oldbalanceOrg'] <= 50000000]
    df = df[df['newbalanceOrig'] <= 40000000]
    df = df[df['oldbalanceDest'] <= 350000000]
    df = df[df['newbalanceDest'] <= 350000000]
    df['balanceOrigDiff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df = df[['balanceOrigDiff', 'amount', 'isFraud']]
    return df

@st.cache_data
def train_model(df):
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return model, accuracy, y_val, y_pred

model, accuracy, y_val, y_pred = train_model(load_data())

def generate_pdf_report(accuracy, report_content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fraud Detection Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Model Accuracy: {accuracy * 100:.2f}%", ln=True, align='L')
    pdf.cell(200, 10, txt="Report Content:", ln=True, align='L')
    for line in report_content:
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    pdf.output("fraud_detection_report.pdf")
    with open("fraud_detection_report.pdf", "rb") as f:
        b64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return b64_pdf

def main():
    st.markdown("""
    <style>
        .title {
            color: #007bff;
            font-size: 2em;
            text-align: center;
        }
        .header {
            color: #343a40;
            font-size: 1.5em;
            font-weight: bold;
        }
        .section {
            margin: 20px 0;
        }
        .button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
        .button:hover {
            background-color: #218838;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">Fraud Detection System</div>', unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ["Login", "Data Explorer", "Report Generator", "Predict Fraud"])

    if options == "Login":
        st.sidebar.subheader("User Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username == "admin" and password == "admin":
                st.session_state['authenticated'] = True
                st.sidebar.success("Logged in successfully")
            else:
                st.sidebar.error("Invalid username or password")
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            st.success("You are logged in")

    elif options == "Data Explorer":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            st.header("Data Explorer")
            if st.checkbox("Show Data"):
                st.write(load_data().head())

            st.subheader("Data Visualization")
            st.write("Correlation Heatmap")
            df = load_data()
            corr = df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, mask=mask, cmap=plt.cm.YlGnBu)
            st.pyplot()

            st.write("Balance Origin Difference by Fraud")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x="isFraud", y="balanceOrigDiff", ax=ax)
            plt.title('Balance Origin Difference by Fraud Status')
            st.pyplot()

        else:
            st.warning("Please login to access this section")

    elif options == "Report Generator":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            st.header("Report Generator")
            st.subheader("Logistic Regression Model")
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            cm = confusion_matrix(y_val, y_pred)
            st.write("Confusion Matrix")
            sns.heatmap(cm, annot=True, fmt='g')
            st.pyplot()

            st.write("Classification Report")
            st.text(classification_report(y_val, y_pred))

            report = classification_report(y_val, y_pred, target_names=['Not Fraud', 'Fraud'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            csv_buffer = io.StringIO()
            report_df.to_csv(csv_buffer)
            csv_data = csv_buffer.getvalue()
            st.download_button(label="Download Report", data=csv_data, file_name='classification_report.csv', mime='text/csv')

            if st.button("Generate PDF Report"):
                report_content = ["This is a sample report content."]
                b64_pdf = generate_pdf_report(accuracy, report_content)
                pdf_link = f'<a class="button" href="data:application/octet-stream;base64,{b64_pdf}" download="fraud_detection_report.pdf">Download PDF Report</a>'
                st.markdown(pdf_link, unsafe_allow_html=True)

        else:
            st.warning("Please login to access this section")

    elif options == "Predict Fraud":
        if 'authenticated' in st.session_state and st.session_state['authenticated']:
            st.header("Predict Fraud")
            st.subheader("Input Transaction Details")
            balanceOrigDiff = st.number_input("Balance Origin Difference", min_value=0.0, value=0.0)
            amount = 0  
            if st.button("Predict"):
                input_data = pd.DataFrame({'balanceOrigDiff': [balanceOrigDiff], 'amount': [amount]})
                prediction = model.predict(input_data)[0]
                prediction_prob = model.predict_proba(input_data)[0][prediction]
                if prediction == 1:
                    st.error(f"The transaction is predicted as Fraud with a probability of {prediction_prob:.2f}")
                    if st.button("Call Helpline"):
                        st.write("[Call Helpline](tel:+919654603250)")
                else:
                    st.success(f"The transaction is predicted as Not Fraud with a probability of {prediction_prob:.2f}")

            example_value = 5000000
            example_amount = 0  
            example_input_data = pd.DataFrame({'balanceOrigDiff': [example_value], 'amount': [example_amount]})
            example_prediction = model.predict(example_input_data)[0]
            if example_prediction == 1:
                st.write(f"Example: A balance origin difference of {example_value} is predicted as Fraud.")
            else:
                st.write(f"Example: A balance origin difference of {example_value} is predicted as Not Fraud.")
        else:
            st.warning("Please login to access this section")

    st.subheader("Feedback Form")
    feedback = st.text_area("Provide your feedback here:")
    if st.button("Send Feedback"):
        encoded_feedback = urllib.parse.quote(feedback)
        feedback_email_link = f"mailto:chawlapc.619@gmail.com?subject=Feedback from Fraud Detection App&body={encoded_feedback}"
        st.markdown(f'<a class="button" href="{feedback_email_link}">Send Feedback</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
