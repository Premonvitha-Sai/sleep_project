import streamlit as st
import numpy as np
import pandas as pd
import time
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
    # Hash the password before storing
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    # Insert user credentials into the database
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()

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
        else:
            st.error("Incorrect username or password.")

def predictor_page():
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Set readings"
    
    # Initialize issue_prediction
    if "issue_prediction" not in st.session_state:
        st.session_state.issue_prediction = None
        
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

    # Rest of the code for the predictor page goes here
    data_tab, result_tab, instr_tab = st.tabs(["Set readings", "Results", "Instructions"])

    with data_tab:
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
						 # Set the active tab to "Results"
            st.session_state.active_tab = "Results"
            with result_tab:
                with st.spinner('Wait for it...'):
                    time.sleep(5)
                    st.write(f"### Probability of having a sleep issue:")
                    fig = plot_filled_gender(is_male, np.round(issue_prob, 2))

                    # This value will control the layout and will give the user extra information 
                    # on the sleep condition it suffers using the logistic model
                    cut_off = 50

                    # Show plot in the middle of the app
                    col1, col2, col3 = st.columns([.2,.8,.1])
                    with col2:
                        st.pyplot(fig, use_container_width=False)
                        issue_prediction = issue_type_model.predict(input_df)[0]
                        issue_prediction = "Sleep Apnea" if (issue_prediction == 1) else "Insomnia"

                        if (issue_prob >= cut_off):
                            title = f"You may have {issue_prediction}"
                            st.pyplot(sleep_issue_image(issue_prediction, title), use_container_width=False)
                            if issue_prediction == "Sleep Apnea":
                                therapy_text = """
                                    ## Therapy for Sleep Apnea:
                                    1. Maintain a healthy weight.
                                    2. Optimize sleep position.
                                    3. Avoid alcohol and sedatives before bed.
                                    4. Use continuous positive airway pressure (CPAP) therapy.
                                    5. Elevate the head of the bed.
                                    6. Stay mindful of caffeine intake.
                                """
                            elif issue_prediction == "Insomnia":  # Add this condition for insomnia
                                 therapy_text = """
                                    ## Therapy for Sleep Apnea:
                                    1. Maintain a healthy weight.
                                    2. Optimize sleep position.
                                    3. Avoid alcohol and sedatives before bed.
                                    4. Use continuous positive airway pressure (CPAP) therapy.
                                    5. Elevate the head of the bed.
                                    6. Stay mindful of caffeine intake."""
                                 st.markdown(therapy_text)

        
        else:
             with result_tab:
                  if st.session_state.active_tab == "Results":
                        st.markdown('Set your readings in the "Set readings" tab and hit predict to get your prediction')
                   

    # Display therapy recommendations
    with instr_tab:
        st.title("Home Level Therapy for Sleep Disorders")
        insomnia_therapy_text = """
            ## Therapy for Insomnia:
            1. Establish a consistent sleep schedule.
            2. Create a relaxing bedtime routine.
            3. Create a comfortable sleep environment.
            4. Limit exposure to screens before bedtime.
            5. Watch your diet and avoid stimulants.
            6. Engage in light exercise.
            7. Stay mindful of caffeine intake.
        """
        sleep_apnea_therapy_text = """
            ## Therapy for Sleep Apnea:
            1. Maintain a healthy weight.
            2. Optimize sleep position.
            3. Avoid alcohol and sedatives before bed.
            4. Use continuous positive airway pressure (CPAP) therapy.
            5. Elevate the head of the bed.
            6. Stay mindful of caffeine intake.
        """
        healthy_sleep_therapy_text = """
            ## Therapy for Healthy Sleep:
            - Be consistent.
            - Make sure your bedroom is quiet, dark, relaxing, and at a comfortable temperature.
            - Remove electronic devices, such as TVs, computers, and smartphones, from the bedroom.
            - Avoid large meals, caffeine, and alcohol before bedtime.
            - Get some exercise.
        """
        st.markdown("### Insomnia Therapy")
        st.markdown(insomnia_therapy_text)
        st.markdown("### Sleep Apnea Therapy")
        st.markdown(sleep_apnea_therapy_text)
        st.markdown("### Healthy Sleep Therapy")
        st.markdown(healthy_sleep_therapy_text)

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
    # Logged-in users can access predictor page
    predictor_page()
