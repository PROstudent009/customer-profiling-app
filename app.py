import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Title of the web app
st.title('Visa Eligibility Classifier Using XGBoost')

# Sidebar for user input
st.sidebar.header('Applicant Data Input')


# Function to get user input from sidebar
def user_input_features():
    continent = st.sidebar.selectbox('Continent',
                                     ['Asia', 'Europe', 'Africa', 'North America', 'South America', 'Oceania'])
    education_of_employee = st.sidebar.selectbox('Education of Employee',
                                                 ['High School', 'Bachelors', 'Masters', 'PhD'])
    has_job_experience = st.sidebar.selectbox('Job Experience', ['Y', 'N'])
    requires_job_training = st.sidebar.selectbox('Requires Job Training', ['Y', 'N'])
    no_of_employees = st.sidebar.slider('Number of Employees in Company', 1, 10000, 50)
    yr_of_estab = st.sidebar.slider('Year of Establishment', 1800, 2024, 2000)
    region_of_employment = st.sidebar.selectbox('Region of Employment in US', ['Northeast', 'Midwest', 'South', 'West'])
    prevailing_wage = st.sidebar.number_input('Prevailing Wage', 0, 1000000, 50000)
    unit_of_wage = st.sidebar.selectbox('Unit of Wage', ['Hourly', 'Weekly', 'Monthly', 'Yearly'])
    full_time_position = st.sidebar.selectbox('Full-time Position', ['Y', 'N'])

    data = {
        'continent': continent,
        'education_of_employee': education_of_employee,
        'has_job_experience': has_job_experience,
        'requires_job_training': requires_job_training,
        'no_of_employees': no_of_employees,
        'yr_of_estab': yr_of_estab,
        'region_of_employment': region_of_employment,
        'prevailing_wage': prevailing_wage,
        'unit_of_wage': unit_of_wage,
        'full_time_position': full_time_position
    }

    features = pd.DataFrame(data, index=[0])
    return features


# Input data from user
input_data = user_input_features()

# Display input data
st.subheader('Applicant Data')
st.write(input_data)

data = pd.read_csv('EasyVisa.csv')
data.drop(['case_id'],axis=1,inplace=True)
df = pd.DataFrame(data)

# Splitting into features and target
X = df.drop(columns=['case_status'])
y = df['case_status'].apply(lambda x: 1 if x == 'Certified' else 0)

# Preprocess data using OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

# Fit on training data and apply to both train and user input
X_encoded = encoder.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# XGBoost classifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Test model accuracy
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy:.2f}')

st.subheader('Model Recall Score')
st.write(f'Recall: {recall:.2f}')


# Encode user input
input_data_encoded = encoder.transform(input_data)

# Make predictions with user input
prediction = xgb_model.predict(input_data_encoded)

# Display prediction result
st.subheader('Prediction: Is the applicant eligible for visa?')
result = 'Certified' if prediction == 1 else 'Denied'
st.write(result)


st.subheader('Visa Classification: Continent-wise')

# Continent-wise distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='continent', hue='case_status', multiple='stack', kde=False, discrete=True)
plt.title('Continent-wise Visa Classification')
plt.xlabel('Continent')
plt.ylabel('Number of Applications')

# Display the histogram in Streamlit
st.pyplot(plt)
plt.close()

st.subheader('Visa Classification: Education-wise')

# Education-wise distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='education_of_employee', hue='case_status', multiple='stack', kde=False, discrete=True)
plt.title('Education-wise Visa Classification')
plt.xlabel('Education Level')
plt.ylabel('Number of Applications')

# Display the histogram in Streamlit
st.pyplot(plt)
plt.close()
