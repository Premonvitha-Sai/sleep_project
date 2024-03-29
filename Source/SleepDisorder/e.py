import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sqlite3
import hashlib
from modeling import load_model
import openai
import base64

def initialize_gpt_session():
    """Initializes or resets the GPT model in the session state."""
    st.session_state["openai_model"] = "gpt-3.5-turbo"
    st.session_state.messages = []

def load_gpt3_model():
    st.title(f":rainbow[This is a GPT-3 Bot]🤖")
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sidebar logic for displaying the Go Back button conditionally
    if "logged_in" in st.session_state and st.session_state.logged_in:
      if "ask_bot" in st.session_state and st.session_state.ask_bot:
        
        if st.sidebar.button("⬅️ Go Back"):
            # Correct the condition to reset the bot interaction state
             # Correctly indicate exiting the bot page
            st.session_state.ask_bot = False  # Assuming 'ask_bot' is also used for navigation control

            # Optionally trigger a rerun to refresh the page layout based on the updated session state
            st.experimental_rerun()
            
def load_gpt3_models():
    st.title(f":rainbow[This is a GPT-3 Bot]🤖")
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Sidebar logic for displaying the Go Back button conditionally
    if "logged_in" in st.session_state and st.session_state.logged_in:
      if "ask_bot" in st.session_state and st.session_state.ask_bot:
          if st.sidebar.button("⬅️ Go Back"):
            # Correct the condition to reset the bot interaction state
             # Correctly indicate exiting the bot page
            st.session_state.ask_bot = False  # Assuming 'ask_bot' is also used for navigation control

            # Optionally trigger a rerun to refresh the page layout based on the updated session state
            st.experimental_rerun()




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
    st.title(":rainbow[Diagnostic System for Sleep Disorder]")
    st.write("""
        This system aims to diagnose sleep disorders using machine learning models. 
        Upload your data and get insights into your sleep health!
    """)

   
    file_path = "main_page_image.gif"  # Update the path to your local GIF file
    file_ = open(file_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    # Modify the img tag to include width and height attributes
    # This example sets the width to 100% to make it responsive
    # Adjust the width and height values as needed
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="main page image" style="width:100%;">',
        unsafe_allow_html=True,
    )


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

            # Reset GPT model and messages for the new session
            initialize_gpt_session()

            # Rest of the login logic
            st.session_state.logged_in = True
            st.session_state.username = username

            # Now, proceed to the predictor page or any other intended destination
            predictor_page()
        else:
            st.error("Incorrect username or password.")


def sleep_issue_image(issue):
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
    # ax.set_title(TITLE, fontsize=8, color="#FFFFFF")
    
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

    # st.title("Set readings")
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
    
    st.title(":rainbow[Results]")
    # Retrieve results from session state
    input_df = st.session_state.result['input_df']
    issue_prob = st.session_state.result['issue_prob']
    is_male = st.session_state.result['is_male']
    issue_prediction_text = st.session_state.result['issue_prediction_text']

    st.write(f"### Probability of having a sleep issue: {issue_prob}%")
    fig = plot_filled_gender(is_male, np.round(issue_prob, 2))
    st.pyplot(fig, use_container_width=False)
    
    # Display risk level warning based on probability
    if 45 < issue_prob < 50:
        st.warning("There are chances of getting exposed to a sleep disorder. Please take care.")
        
        
    elif 50 < issue_prob < 70:
        st.warning(f"You have a :blue[**Moderate risk level.**] Please check our home-level therapy for :red[**{issue_prediction_text}**], :violet[**{st.session_state.username}**]🥺")
    elif issue_prob >= 70:
        st.error(f"You have a :orange[**High risk level.**] Please consult the Doctor and check our home-level therapy for :orange[**{issue_prediction_text}**] temporarily😊")
        
    if issue_prob >= 50:
        #  title = f"You may have :red[ {issue_prediction_text}"
         st.title(f"You may have :red[ {issue_prediction_text}]")
         st.markdown('### Check 👇 for Therapy Recommendations')
         if st.button("Therapy"):
            st.session_state.therapy_requested = True
         st.pyplot(sleep_issue_image(issue_prediction_text), use_container_width=False)
        
    else:
        st.title(f":green[Your Sleep Health is Good],:violet[{st.session_state.username}]👍")

    
    if "logged_in" in st.session_state and st.session_state.logged_in:
        # Place the back button beside the logout button
        _, _,col2 = st.columns([1, 3, 1])
        with col2:
            if st.button("⬅️ Go Back"):
                st.session_state.pop('result', None)  # Clear result from session state



def therapy_page():
    st.title("Therapy Recommendations")
    # st.markdown('### Diagnosis and Therapy Recommendation')

    # Retrieve the predicted sleep issue and risk level from session state
    issue_prediction_text = st.session_state.result['issue_prediction_text']
    issue_prob = st.session_state.result['issue_prob']

    # Define therapy recommendations based on issue and risk level
    if issue_prediction_text == "Sleep Apnea":
        if issue_prob >= 70:
             st.error(f"**:orange[It's crucial for you to consult with a healthcare provider for a comprehensive treatment plan tailored to your specific needs]**, :violet[{st.session_state.username}]🥺")
             st.markdown(f"""
                ##### High Risk Level Remedies for {issue_prediction_text}
                1. **Positional Therapy:** Sleeping on your side can prevent the tongue and soft tissues in the throat from obstructing the airway, a common issue in sleep apnea.
                2. **Weight Management:** Excess weight can contribute to throat constriction and worsen sleep apnea symptoms. A healthy diet and exercise plan can help reduce these risks.
                3. **Avoid Alcohol and Sedatives:** These substances relax the muscles in the throat, increasing the risk of airway obstruction during sleep.
                4. **Allergy Management:** Reduce allergens in your bedroom since nasal congestion can worsen sleep apnea symptoms.
                5. **CPAP Adherence:** For severe sleep apnea, continuous positive airway pressure (CPAP) therapy is essential. Ensuring proper mask fit and gradually acclimating to the machine can improve comfort and compliance.
            """)
        elif 50 < issue_prob < 70:
            st.warning(f"**:blue[Moderate levels may not always require the same intensity of medical intervention as severe cases, a healthcare professional can provide valuable guidance, rule out underlying conditions, and recommend tailored treatment strategies,]** :violet[**{st.session_state.username}**]🥺")
            st.markdown(f"""
    ##### Moderate Risk Level Remedies for {issue_prediction_text}
    1. **Elevate Head of Bed:** Elevating the head during sleep can reduce snoring and mild sleep apnea by keeping airways open.
    2. **Nasal Sprays or Breathing Strips:** These can help keep your nasal passages open and reduce sleep apnea symptoms.
    3. **Regular Sleep Schedule:** Consistency in sleep patterns can help improve the quality of sleep and reduce apnea episodes.
    4. **Limit Caffeine and Heavy Meals Before Bed:** This can improve sleep quality and reduce the likelihood of sleep disturbances.
    5. **Quit Smoking:** Smoking can increase inflammation and fluid retention in the throat, worsening sleep apnea.
""")


    elif issue_prediction_text == "Insomnia":
        if issue_prob >= 70:
            st.error(f"**:orange[It's crucial for youto consult with a healthcare provider for a comprehensive treatment plan tailored to your specific needs]**, :violet[{st.session_state.username}]🥺")
            st.markdown(f"""
                ##### High Risk Level Remedies for :red[{issue_prediction_text}]
                1. **Cognitive Behavioral Therapy (CBT) for Insomnia:** This is a structured program that helps you identify and replace thoughts and behaviors that cause or worsen sleep problems with habits that promote sound sleep.
                2. **Prescription Medication:** In some cases, medication may be necessary under the guidance of a healthcare provider.
                3. **Relaxation Techniques:** Techniques such as deep breathing exercises, meditation, and progressive muscle relaxation can help reduce anxiety and promote sleep.
                4. **Sleep Restriction Therapy:** Limiting the time spent in bed can help consolidate sleep and increase sleep efficiency.
                5. **Avoid Stimulants:** Eliminate or reduce consumption of caffeine and nicotine, especially in the hours leading up to bedtime.
            """)
        elif 50 < issue_prob < 70:
            st.warning(f"**:blue[Moderate levels may not always require the same intensity of medical intervention as severe cases, a healthcare professional can provide valuable guidance, rule out underlying conditions, and recommend tailored treatment strategies,]** :violet[**{st.session_state.username}**]🥺")
            st.markdown(f"""
                ##### Moderate Risk Level Remedies for :red[{issue_prediction_text}]
                1. :green[**Regular Exercise:**] Engaging in regular, moderate exercise can help improve sleep patterns, though it's best not to exercise right before bedtime.
                2. **:green[Sleep Environment Improvement:]** Ensure your bedroom is comfortable, dark, quiet, and cool to facilitate easier sleep.
                3. **:green[Limit Screen Time Before Bed:]** Reducing exposure to screens before bedtime can help cue your brain that it's time to wind down.
                4. **:green[Mindfulness and Meditation:]** Regular practice can help reduce stress and anxiety, making it easier to fall asleep.
                5. **:green[Consistent Sleep Schedule:]** Going to bed and waking up at the same time every day can help regulate your body's internal clock and improve sleep quality.
            """)
        # if st.button("⬅️ Go Back to Results"):
        #   st.session_state.therapy_requested = False
        #   st.experimental_rerun()

    if "logged_in" in st.session_state and st.session_state.logged_in:
    # Place the back button beside the logout button
     _, _, col2 = st.columns([1, 3, 1])
     with col2:
        if st.button("⬅️ Go Back"):
            st.session_state.therapy_requested = False
            st.experimental_rerun()
            
def chatbot_page():
    st.title(":rainbow[This Bot🤖 tries to answer any of your questions, **:violet[{st.session_state.username}]**🥺]")
    load_gpt3_model()


# Logout button implementation
# Logout button implementation
if "logged_in" in st.session_state and st.session_state.logged_in:
    col1,_,col2 = st.columns([1,3,1])
    with col1:
        if st.button("Logout"):
            # Logout logic
            initialize_gpt_session()
            st.session_state.logged_in = False
            

        
            # Cleanup
            st.session_state.pop('ask_bot', None)  # Remove the bot interaction flag if it exists
            st.experimental_rerun()
    with col2:
        if st.button("Ask our Bot 🤖"):
            # Direct the user to interact with the bot
            st.session_state.ask_bot = True
            st.experimental_rerun()



# Check if user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Sidebar Options for not logged-in users
    sidebar_option = st.sidebar.radio("Navigation", ["Home", "Register", "Login", "Ask our Bot🤖"])

    # Display corresponding page based on sidebar option for not logged-in users
    if sidebar_option == "Home":
     home_page()
    elif sidebar_option == "Register":
      register_page()
    elif sidebar_option == "Login":
      login_page()
    elif sidebar_option == "Ask our Bot🤖":
      load_gpt3_models()
      
else:
	   # Check if the user has requested to interact with the bot
    if "ask_bot" in st.session_state and st.session_state.ask_bot:
        load_gpt3_model()  # This will display the chat interface
    elif "therapy_requested" in st.session_state and st.session_state.therapy_requested:
        therapy_page()
    elif "result" in st.session_state:
        result_page()
    else:
        predictor_page()
