import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time

# Load your dataframe here
# try:
#     df = pd.read_csv(r'C:\Users\ASUS\Downloads\diabetes\diabetes.csv')
# except FileNotFoundError:
#     st.error("Dataset file not found. Please check the file path.")
#     df = pd.DataFrame()  # Initialize an empty DataFrame


def show_prediction_page(df):
    st.markdown('<h1 style="color: Green;"> Check your diabetes ❤</h1>', unsafe_allow_html=True)
    st.write(
        '<p style="text-align: justify;">'
        'Welcome to the Diabetes Prediction Application. This tool uses a machine learning model to predict the likelihood of diabetes '
        'based on various health metrics. Please enter your health details in the sidebar to get your prediction. '
        'The visualizations below will help you understand your health metrics compared to others.'
        '</p>',
        unsafe_allow_html=True
    )

    # Sidebar Inputs
    st.sidebar.header('Patient Data')
    age = st.sidebar.number_input('Age of the Person:', min_value=0)
    gender = st.sidebar.radio('Gender:', ['Male', 'Female'])
    if gender == 'Female':
        pregnancies = st.sidebar.number_input('Number of Pregnancies:', min_value=0)
    else:
        pregnancies = 0
    glucose = st.sidebar.slider('Glucose Level:', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure Value:', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness Value:', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin Level:', 0, 846, 79)
    bmi = st.sidebar.slider('BMI Value:', 0.0, 67.0, 20.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function Value:', 0.0, 2.4, 0.47)
    st.sidebar.file_uploader('If you like Upload patient document')
    st.sidebar.text_area('Description')

    # Ensure all inputs are provided
    input_values = [glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
    all_inputs_provided = all(value != 0 for value in input_values)

    # Custom CSS for button
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50; 
            color: black; 
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }
        
                

        </style>
    """, unsafe_allow_html=True)

    # Button to Check Result
    if st.sidebar.button('Submit') and all_inputs_provided:
        # Preparing Data
        x = df.drop(['Outcome'], axis=1)
        y = df['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Model Training
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        # User Data
        user_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree_function],
            'Age': [age]
        })

        # st.subheader("wait the execution")
        with st.spinner('Wait for it...'):
            time.sleep(5)

        # Prediction
        user_result = rf.predict(user_data)

        # Output
        st.markdown('<h2 style="color: Red;"> Your Report: </h2>', unsafe_allow_html=True)
        # st.subheader('Your Report:')
        output = 'You are not Diabetic 😍' if user_result[0] == 0 else 'You are Diabetic 😓'
        st.write(output)

        # Accuracy
        # st.subheader('Accuracy:')
        st.markdown('<h2 style="color: Red;"> Accuracy: </h2>', unsafe_allow_html=True)
        st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

        # Visualizations
        st.markdown('<h1 style="color: pink;">Visualised Patient Report (Others vs Yours)</h1>', unsafe_allow_html=True)

        # Color Function
        color = 'blue' if user_result[0] == 0 else 'red'

        # Visualization Functions
        def plot_graph(x, y, user_x, user_y, title, x_label, y_label, hue, palette, ticks_x, ticks_y):
            fig = plt.figure()
            sns.scatterplot(x=x, y=y, data=df, hue=hue, palette=palette)
            sns.scatterplot(x=user_x, y=user_y, s=150, color=color)
            plt.xticks(np.arange(*ticks_x))
            plt.yticks(np.arange(*ticks_y))
            plt.title(title)
            st.pyplot(fig)

        # Age vs DPF
        # st.header('DPF Value Graph (Others vs Yours)')
        st.markdown('<h1 style="color: white;"> Diabetes Pedigree Function Graph </h1>', unsafe_allow_html=True)
        plot_graph('Age', 'DiabetesPedigreeFunction', user_data['Age'], user_data['DiabetesPedigreeFunction'],
                   '0 - Healthy & 1 - Unhealthy', 'Age', 'DiabetesPedigreeFunction', 'Outcome', 'YlOrBr',
                   (10, 100, 5), (0, 3, 0.2))

    else:
        st.warning("Please provide all input values.")


# show_prediction_page(df)
