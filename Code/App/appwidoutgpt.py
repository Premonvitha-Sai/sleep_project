import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sqlite3
import hashlib
from modeling import load_model



# Create or connect to a SQLite database
conn = sqlite3.connect('user_credentials.db')
c = conn.cursor()

# Create table to store user credentials if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (
             username TEXT PRIMARY KEY,
             password TEXT)''')
conn.commit()

def register_user(username, password):
    # Hash the password to enhance security
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    # Check if the username already exists
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    if c.fetchone():
        #st.error("Username already exists. Please choose a different username.")
        return False  # Registration failed
    
    # Insert the new user into the database
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    st.success("Registration successful! You can now login.")
    return True  # Registration succeeded


def authenticate_user(username, password):
    # Hash the password to compare with the stored hash
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    # Retrieve user credentials from the database
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    if user:
        return True
    else:
        return False

def home_page():
    st.title("Diagnostic System for Sleep Disorder")
    st.write("""
        This system aims to diagnose sleep disorders using machine learning models. 
        Upload your data and get insights into your sleep health!
    """)
    main_page_image = mpimg.imread('main_page_image.jpg')
    st.image(main_page_image, use_column_width=True)

def register_page():
    st.title("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_password == confirm_password:
            register_user(new_username, new_password)
            st.success("Registration successful! You can now login.")
        else:
            st.error("Passwords do not match. Please try again.")

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Login successful!")
            # Set session variable for logged-in user
            st.session_state.logged_in = True
            st.session_state.username = username
            predictor_page()
        else:
            st.error("Incorrect username or password.")

def sleep_issue_image(issue, TITLE):
    fig, ax = plt.subplots(figsize=(2,2))
    fig.patch.set_alpha(0)

    img_path = 'images/sleep_apnea.jpeg' if (issue=='Sleep Apnea') else 'images/insomnia.jpeg'

    # Load image
    img = mpimg.imread(img_path)

    # Display the modified image
    ax.imshow(img, interpolation='none')

    # Remove axis
    ax.axis('off')

    # Comment on the issue for the user
    ax.set_title(TITLE, fontsize=8, color="#FFFFFF")
    
    return fig

def plot_filled_gender(male, percentage):
    fig, ax = plt.subplots(figsize=(1,2))
    fig.patch.set_alpha(0)

    img_path = 'images/male_silhouette.png' if (male==1) else 'images/female_silhouette.png'

    # Load image
    img = mpimg.imread(img_path)

    # Convert the entire image to gray while preserving the alpha (transparency) channel
    grayscale_img_with_alpha = np.zeros_like(img)
    grayscale_img_with_alpha[:, :, :3] = [0.05098039, 0.07058824, 0.50980392] # issue color
    grayscale_img_with_alpha[:, :, 3] = img[:, :, 3]

    # Calculate the y limit based on percentage
    ylim = int(img.shape[0] * (1 - (percentage / 100)))

    # Replace the grayscale values with the original values up to the y-limit
    grayscale_img_with_alpha[:ylim, :, :3] = [0,0,0]

    # Display the modified image
    ax.imshow(grayscale_img_with_alpha, interpolation='none')

    # Add the percentage text
    ax.text(0.5, 0.5, f'{percentage}%', ha='center', va='center', color='white', 
            fontweight='bold', fontsize=10, transform=ax.transAxes)

    # Remove axis
    ax.axis('off')
    
    return fig

def predictor_page():
    st.title(f"Sleep Issue Predictor💤 - Welcome, {st.session_state.username}!")

    # Loading issue model with scaling
    issue_model = load_model('models/NoSystolic_ScaledModelSVC.pkl')

    # Loading type of issue model with scaling
    issue_type_model = load_model('models/SleepIssueType_ModelScaled.pkl')

    def proba_predict(model, input_data):
        """
        Extracts the probability of having any sleep issue
        """
        prediction = model.predict_proba(input_data)
        return prediction[0][1]

    st.title("Set readings")
    # Creating column components for the plot
    _col1, _col2 = st.columns([.5,.5])

    with _col1:
        # User inputs
        age = st.number_input("Enter your age", value=31, min_value=18, max_value=60)
        sleep_duration = st.number_input("Enter sleep duration", value=8.0)
        heart_rate = st.number_input("Enter heart rate", value=70, min_value=60)
        daily_steps = st.number_input("Enter daily steps", value=8000)
    
    with _col2:
        is_male = st.selectbox("Select your gender", ["Male", "Female"])
        wf_technical = st.selectbox("Do you work in a technical or numeric field?", ["Yes", "No"])
        
        # BMI Calculation
        weight = st.number_input("What's your weight in Kg (kilograms)?", value=70.0)
        height = st.number_input("What's your height in M (meters)?", value=1.60)

    BMI = weight / np.square(height)
    elevated_bmi = 1 if BMI >= 25 else 0

    # Convert categorical data to numeric
    is_male = 1 if is_male == "Male" else 0
    wf_technical = 1 if wf_technical == "Yes" else 0

    # Predict button
    if st.button("Predict"):
        # Getting input data
        input_data = np.array([[int(age), sleep_duration, heart_rate, int(daily_steps), is_male, elevated_bmi, wf_technical]])
        columns = ['age', 'sleep_duration', 'heart_rate', 'daily_steps', 'is_male', 'elevated_bmi', 'wf_technical']
        input_df = pd.DataFrame(input_data, columns=columns)

        # Predicting using the model
        issue_prob = proba_predict(issue_model, input_df) * 100

        # Determine the issue type
        issue_prediction = issue_type_model.predict(input_df)[0]
        issue_prediction_text = "Sleep Apnea" if (issue_prediction == 1) else "Insomnia"

        # Store results and issue prediction in session state
        st.session_state.result = {
            'input_df': input_df,
            'issue_prob': issue_prob,
            'is_male': is_male,
            'issue_prediction_text': issue_prediction_text
        }

        # Redirect to result page
        result_page()

def result_page():
    st.title("Results")
    # Retrieve results from session state
    input_df = st.session_state.result['input_df']
    issue_prob = st.session_state.result['issue_prob']
    is_male = st.session_state.result['is_male']
    issue_prediction_text = st.session_state.result['issue_prediction_text']

    st.write(f"### Probability of having a sleep issue: {issue_prob}%")
    fig = plot_filled_gender(is_male, np.round(issue_prob, 2))
    st.pyplot(fig, use_container_width=False)

    if issue_prob >= 50:
        

        title = f"You may have {issue_prediction_text}"
        st.pyplot(sleep_issue_image(issue_prediction_text, title), use_container_width=False)
        if st.button("Therapy"):
            st.session_state.therapy_requested = True
    else:
        st.title("Your Sleep Health is Good👍")

    if st.button("⬅️ Go Back"):
        st.session_state.pop('result', None)  # Clear result from session state

def therapy_page():
    st.title("Therapy Recommendations")
    st.markdown('### Diagnosis and Therapy Recommendation')

    # Retrieve the predicted sleep issue from session state
    issue_prediction_text = st.session_state.result['issue_prediction_text']

    if issue_prediction_text == "Insomnia":
        st.markdown("""
            ##### Therapy for Insomnia:
            1. Maintain a consistent sleep schedule.
            2. Create a bedtime routine.
            3. Limit exposure to bright light in the evenings.
            4. Avoid consuming caffeine and alcohol late in the day.
            5. Manage stress.
            6. Exercise regularly, but not close to bedtime.
        """)
    elif issue_prediction_text == "Sleep Apnea":
        st.markdown("""
            ##### Therapy for Sleep Apnea:
            1. Maintain a healthy weight.
            2. Optimize sleep position.
            3. Avoid alcohol and sedatives before bed.
            4. Use continuous positive airway pressure (CPAP) therapy.
            5. Elevate the head of the bed.
            6. Stay mindful of caffeine intake.
        """)

    if st.button("⬅️ Go Back"):
        st.session_state.therapy_requested = False

# Logout button implementation
if "logged_in" in st.session_state and st.session_state.logged_in:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Sidebar Options for not logged-in users
    sidebar_option = st.sidebar.radio("Navigation", ["Home", "Register", "Login"])

    # Display corresponding page based on sidebar option for not logged-in users
    if sidebar_option == "Home":
        home_page()
    elif sidebar_option == "Register":
        register_page()
    elif sidebar_option == "Login":
        login_page()
else:
    if "therapy_requested" in st.session_state and st.session_state.therapy_requested:
        therapy_page()
    elif "result" in st.session_state:
        result_page()
    else:
        predictor_page()
