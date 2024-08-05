import streamlit as st
from main2 import show_prediction_page
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Custom CSS to style the sidebar and the main page
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: green;
    }
    .stApp {
        background-color: black;
        color: white;
    }
    h1, h2, h3, h4, p, div, span, li, {
        color: white !important;
    }
    .number-input-label {
        color: white !important;
    }

   
    </style>
    """,
    unsafe_allow_html=True
)

st.progress(10)

# Load dataset
#try:
   # df = pd.read_csv('./diabetes/diabetes.csv')
#except FileNotFoundError:
    #st.error("Dataset file not found. Please check the file path.")
    #df = pd.DataFrame()  # Initialize an empty DataFrame
df = pd.read_csv(r'C:\Users\ASUS\Downloads\diabetes\diabetes.csv')

def greet():
    st.write("******************************************")

# WELCOME MESSAGE
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Prediction'])  # Two pages
if app_mode == 'Home':
    #st.title("👋Hello...")
    st.markdown('<h1 style="color: white;">👋Hello...</h1>', unsafe_allow_html=True)
    st.markdown('<h1 style="color: pink; text-align: center;">Welcome To The Diabetes Prediction Application!</h1>', unsafe_allow_html=True)
    st.image("gif2.gif")
    st.write(
        '<p style="text-align: justify;">'
        'Diabetes is a long-term disease where the body has too much blood sugar. This can harm the heart, blood vessels, eyes, kidneys, and nerves over time.'
        'The most common type is type 2 diabetes usually in adults. It happens when the body cant use insulin properly or does not make enough insulin.'
        'Over the past 30 years type 2 diabetes has increased a lot in all countries. Type 1 diabetes also called juvenile or insulin-dependent diabetes is a condition where the pancreas makes little or no insulin.'
        'For people with diabetes getting affordable treatment and insulin is very important. There is a global goal to stop the rise of diabetes and obesity by 2025.'
        '</p>',
        unsafe_allow_html=True
    )
    
    st.sidebar.image("image1.jpg")
    st.sidebar.image("image1.jpg")
    
    

    
    st.markdown('<h1 style="color: pink;">Protect yourself from diabetes</h1>', unsafe_allow_html=True)
    st.video("vedio.mp4")

    st.image("diabetic.jpg")

    
    
    # Plotting age distribution
    if not df.empty:
        # Age Distribution Bar Graph
        st.markdown('<h1 style="color: white;"> Distribution of Ages </h1>', unsafe_allow_html=True)
        fig = plt.figure()
        sns.histplot(df['Age'], bins=30, kde=True)
        plt.title('Distribution of Ages in Dataset')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        st.pyplot(fig)

            # Glucose Distribution Bar Graph
        st.markdown('<h1 style="color: white;"> Distribution of Glucose Levels </h1>', unsafe_allow_html=True)
        fig = plt.figure()
        sns.histplot(df['Glucose'], bins=30, kde=True)
        plt.title('Distribution of Glucose Levels in Dataset')
        plt.xlabel('Glucose Level')
        plt.ylabel('Frequency')
        st.pyplot(fig)

    else:
        st.warning("Dataset is empty. Please upload a valid dataset.")

    

    greet()
elif app_mode == 'Prediction':
    show_prediction_page(df)

