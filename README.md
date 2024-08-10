# Fraud Detection System
This is a web application built with Streamlit that allows users to explore, analyze, and predict fraudulent transactions. The app includes features like data exploration, model training, fraud prediction, report generation, and a feedback system. It's designed to be user-friendly and provides a secure login mechanism.

# Features
1. User Authentication
Login Page: Users must log in with a username and password to access the features of the app.
Admin Access: The default credentials are set to admin for both username and password.
2. Data Explorer
Data Preview: Users can view the first few rows of the dataset.
Correlation Heatmap: Visualizes the correlation between different features using a heatmap.
Balance Origin Difference Analysis: Boxplot comparing the balanceOrigDiff (difference in the balance of the origin account before and after the transaction) for fraudulent and non-fraudulent transactions.
3. Report Generator
Model Training: Trains a Logistic Regression model on the dataset to predict fraud.
Model Evaluation: Displays the accuracy of the model, confusion matrix, and classification report.
Download Reports: Users can download the classification report as a CSV file or generate a PDF report summarizing the model's performance.
4. Predict Fraud
Transaction Input: Users can input transaction details like balanceOrigDiff to predict whether a transaction is fraudulent or not.
Prediction Output: The app predicts if the transaction is fraudulent and displays the probability of the prediction.
Helpline Support: If a transaction is predicted as fraud, users can directly call a helpline number provided.
5. Feedback System
Submit Feedback: Users can provide feedback through the app, which opens their default email client with pre-filled feedback.
Installation
1. Clone the Repository
bash
Copy code
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
2. Install Dependencies
Make sure you have Python 3.8 or higher installed. Install the required Python packages using:


# Usage
## Login
Use the username admin and password admin to log in.
## Data Explorer
After logging in, navigate to the "Data Explorer" section using the sidebar.
View and analyze the dataset with visualizations.
## Report Generator
Generate and download model performance reports, including a detailed PDF.
## Predict Fraud
Enter transaction details and get instant predictions on whether the transaction is fraudulent.
If the transaction is predicted as fraud, you can call the helpline directly from the app.
## Feedback
Provide feedback using the text area provided. This will open your email client with the feedback message ready to send.
Customization
## Dataset
Replace the Fraud.csv file in the root directory with your dataset. Ensure that the structure of the dataset matches the columns used in the app (balanceOrigDiff, amount, and isFraud).
Authentication
Modify the authentication logic to integrate with a more secure system if needed. The current setup is a simple hardcoded username and password, which is admin & admin.
Model
The Logistic Regression model is used by default. You can experiment with other models by modifying the train_model() function.
Code Overview
fraud_detection.py: The main script that runs the Streamlit app.
Fraud.csv: The dataset used for training the model. Replace this with your dataset.
requirements.txt: Lists all the dependencies required to run the app.
## Dependencies
Python 3.8+
Streamlit
Pandas
Numpy
Matplotlib
Seaborn
Scikit-learn
FPDF (for PDF generation)
Contribution
Feel free to fork this repository and submit pull requests. Any enhancements or bug fixes are welcome!


## Contact
For any inquiries or support, contact the project maintainer at chawlapc.619@gmail.com.

