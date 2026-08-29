#from twilio.rest import Client
import os
import streamlit as st
import pandas as pd
import joblib
import shap
import speech_recognition as sr
import requests
import json

from io import BytesIO
from gtts import gTTS
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

#twilio_client = Client(
#    os.getenv("TWILIO_ACCOUNT_SID"),
#       os.getenv("TWILIO_AUTH_TOKEN")
#)


# ============================================================
# FFMPEG
# ============================================================

AudioSegment.converter = "ffmpeg"

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("irrigation_model.pkl")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

# ============================================================
# GET TELEGRAM CHAT ID
# ============================================================

def get_telegram_chat_id():

    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]

    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/getUpdates"
    )

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()

    if data["ok"] and data["result"]:

        latest_update = data["result"][-1]

        message = latest_update.get("message")

        if message:

            chat = message.get("chat")

            if chat:

                return str(chat["id"])

    return None

# ============================================================
# SAVE TELEGRAM USER
# ============================================================

def save_telegram_user(chat_id):

    file_path = "users.json"

    try:

        with open(file_path, "r") as file:
            users = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):

        users = {}

    users[str(chat_id)] = {
        "chat_id": str(chat_id)
    }

    with open(file_path, "w") as file:

        json.dump(
            users,
            file,
            indent=4
        )

# ============================================================
# SEND TELEGRAM ALERT TO CURRENT USER ONLY
# ============================================================

def send_telegram_alert(
    crop_type,
    soil_moisture,
    temperature,
    humidity,
    rainfall,
    probability,
    chat_id
):

    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]

    if not chat_id:
        raise Exception(
            "Telegram is not connected."
        )

    message = (
        "🚨 IRRIGATION ALERT\n\n"
        f"Crop: {crop_type}\n"
        f"Soil Moisture: {soil_moisture}%\n"
        f"Temperature: {temperature}°C\n"
        f"Humidity: {humidity}%\n"
        f"Rainfall: {rainfall} mm\n\n"
        f"Irrigation Probability: "
        f"{probability * 100:.1f}%\n\n"
        "The system recommends irrigation "
        "based on the current field conditions."
    )

    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": message
        }
    )

    response.raise_for_status()

    return response.json()
# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Crop Monitoring",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Smart Crop Monitoring System")

st.write(
    "AI-powered irrigation prediction with Explainable AI."
)

st.divider()


# ============================================================
# GET TELEGRAM CHAT ID
# ============================================================

def get_telegram_chat_id():

    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]

    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/getUpdates"
    )

    response = requests.get(url)

    response.raise_for_status()

    data = response.json()

    if data["ok"] and data["result"]:

        latest_update = data["result"][-1]

        message = latest_update.get("message")

        if message:

            chat = message.get("chat")

            if chat:

                return str(chat["id"])

    return None


# ============================================================
# SAVE TELEGRAM USER
# ============================================================

def save_telegram_user(chat_id):

    file_path = "users.json"

    try:

        with open(file_path, "r") as file:
            users = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):

        users = {}

    users[str(chat_id)] = {
        "chat_id": str(chat_id)
    }

    with open(file_path, "w") as file:

        json.dump(
            users,
            file,
            indent=4
        )


# ============================================================
# TELEGRAM CONNECTION
# ============================================================

st.subheader("📱 Connect Telegram Alerts")

st.write(
    "Connect your Telegram account to receive personal irrigation alerts."
)

if st.button(
    "🔗 Connect Telegram",
    use_container_width=True
):

    chat_id = get_telegram_chat_id()

    if chat_id:

        st.session_state["telegram_chat_id"] = chat_id

        save_telegram_user(chat_id)

        st.success(
            "✅ Telegram connected successfully!"
        )

        st.info(
            "Your irrigation alerts will be sent only to this Telegram account."
        )

    else:

        st.warning(
            "Please open the Telegram bot and send /start first."
        )
# ============================================================
# CROP INFORMATION
# ============================================================

st.subheader("🌾 Crop Information")

crop_type = st.selectbox(
    "Select Crop",
    [
        "Wheat",
        "Rice",
        "Maize",
        "Cotton"
    ]
)

growth_stage = st.selectbox(
    "Select Growth Stage",
    [
        "Seedling",
        "Vegetative",
        "Flowering",
        "Maturity"
    ]
)


# ============================================================
# ENVIRONMENTAL CONDITIONS
# ============================================================

st.divider()

st.subheader("🌦️ Environmental Conditions")

soil_moisture = st.number_input(
    "Soil Moisture (%)",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=30.0,
    step=1.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    step=1.0
)


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "Soil_Moisture": [soil_moisture],
    "Temperature": [temperature],
    "Humidity": [humidity],
    "Rainfall": [rainfall],
    "Crop_Type": [crop_type],
    "Growth_Stage": [growth_stage]
})
# ============================================================
# IRRIGATION PREDICTION
# ============================================================

st.divider()

st.subheader("💧 Irrigation Decision")

if st.button("Check Irrigation", use_container_width=True):

    input_data = pd.DataFrame({
        "Soil_Moisture": [soil_moisture],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "Crop_Type": [crop_type],
        "Growth_Stage": [growth_stage]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get irrigation probability
    probability = model.predict_proba(input_data)[0][1]

    # Display prediction
    if prediction == 1:

        st.error(
            "🚨 IRRIGATION REQUIRED"
        )

        st.write(
            "The current field conditions indicate "
            "that the crop may require irrigation."
        )

        # Telegram alert
        try:

            chat_id = st.session_state.get(
                "telegram_chat_id"
            )

            if not chat_id:
                raise Exception(
                    "Telegram is not connected. "
                    "Please connect Telegram first."
                )

            send_telegram_alert(
                crop_type,
                soil_moisture,
                temperature,
                humidity,
                rainfall,
                probability,
                chat_id
            )

            st.success(
                "📱 Telegram irrigation alert sent to "
                "your connected Telegram account!"
            )

        except Exception as error:

            st.warning(
                f"Telegram alert could not be sent: {error}"
            )

    else:

        st.success(
            "✅ IRRIGATION NOT REQUIRED"
        )

        st.write(
            "The current field conditions do not indicate "
            "an immediate need for irrigation."
        )

    st.metric(
        "Irrigation Probability",
        f"{probability * 100:.1f}%"
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Why did the model make this prediction?"
    )

    try:

        # Transform input
        transformed_input = preprocessor.transform(
            input_data
        )

        # Get feature names
        feature_names = (
            preprocessor.get_feature_names_out()
        )

        # Create SHAP explainer
        explainer = shap.TreeExplainer(
            classifier
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(
            transformed_input
        )


        # ----------------------------------------------------
        # HANDLE SHAP OUTPUT
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            shap_array = shap_values[prediction][0]

        elif len(shap_values.shape) == 3:

            shap_array = shap_values[
                0,
                :,
                prediction
            ]

        else:

            shap_array = shap_values[0]


        # ----------------------------------------------------
        # CREATE EXPLANATION
        # ----------------------------------------------------

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": shap_array
        })

        explanation["Absolute_Impact"] = (
            explanation["SHAP_Value"].abs()
        )

        explanation = explanation.sort_values(
            "Absolute_Impact",
            ascending=False
        )

        top_features = explanation.head(5)


        # ----------------------------------------------------
        # TOP FACTORS
        # ----------------------------------------------------

        st.write(
            "### Top Factors Influencing the Prediction"
        )

        for _, row in top_features.iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP_Value"]

            feature = feature.replace(
                "categorical__",
                ""
            )

            feature = feature.replace(
                "remainder__",
                ""
            )


            if "Soil_Moisture" in feature:

                friendly_name = "Soil Moisture"

            elif "Temperature" in feature:

                friendly_name = "Temperature"

            elif "Humidity" in feature:

                friendly_name = "Humidity"

            elif "Rainfall" in feature:

                friendly_name = "Rainfall"

            elif "Crop_Type" in feature:

                friendly_name = "Crop Type"

            elif "Growth_Stage" in feature:

                friendly_name = "Growth Stage"

            else:

                friendly_name = feature


            if shap_value > 0:

                explanation_text = (
                    f"{friendly_name} increased "
                    "the irrigation requirement."
                )

            else:

                explanation_text = (
                    f"{friendly_name} reduced "
                    "the irrigation requirement."
                )


            st.write(
                f"🔹 **{explanation_text}**"
            )


        # ----------------------------------------------------
        # SHAP CHART
        # ----------------------------------------------------

        st.write(
            "### 📊 Feature Impact"
        )

        chart_data = top_features.set_index(
            "Feature"
        )["SHAP_Value"]

        st.bar_chart(
            chart_data
        )


    except Exception as error:

        st.warning(
            "The model prediction was successful, "
            "but the SHAP explanation could not be generated."
        )

        st.write(
            f"Technical details: {error}"
        )


# ============================================================
# VOICE ASSISTANT
# ============================================================

st.divider()

st.subheader("🎙️ Voice Assistant")

st.write(
    "Click the microphone button and ask your irrigation question."
)

audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="voice_recorder"
)


# ============================================================
# VOICE PROCESSING
# ============================================================

if audio:

    st.success(
        "Voice recorded successfully."
    )

    try:

        audio_bytes = audio["bytes"]

        # Convert WebM to WAV
        audio_segment = AudioSegment.from_file(
            BytesIO(audio_bytes),
            format="webm"
        )

        wav_buffer = BytesIO()

        audio_segment.export(
            wav_buffer,
            format="wav"
        )

        wav_buffer.seek(0)


        # Speech recognition
        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_buffer) as source:

            recorded_audio = recognizer.record(
                source
            )

        text = recognizer.recognize_google(
            recorded_audio,
            language="en-IN"
        )


        st.subheader(
            "📝 Your Question"
        )

        st.write(text)


        # ----------------------------------------------------
        # QUESTION ANALYSIS
        # ----------------------------------------------------

        question = text.lower()

        irrigation_words = [
            "irrigation",
            "water",
            "watering",
            "irrigate",
            "paani",
            "moisture",
            "rainfall",
            "crop"
        ]


        if any(
            word in question
            for word in irrigation_words
        ):

            # Voice prediction uses the same
            # current field conditions

            voice_prediction = model.predict(
                input_data
            )[0]

            voice_probability = model.predict_proba(
                input_data
            )[0][1]


            st.subheader(
                "🤖 Assistant Response"
            )


            if voice_prediction == 1:

                response_text = (
                    f"Irrigation is required for your "
                    f"{crop_type} crop. "
                    f"The irrigation probability is "
                    f"{voice_probability * 100:.1f} percent. "
                    f"The current field conditions indicate "
                    f"that the crop may need water."
                )

                st.error(
                    "🚨 IRRIGATION REQUIRED"
                )

            else:

                response_text = (
                    f"Irrigation is not required for your "
                    f"{crop_type} crop. "
                    f"The irrigation probability is "
                    f"{voice_probability * 100:.1f} percent. "
                    f"The current field conditions do not "
                    f"indicate an immediate need for irrigation."
                )

                st.success(
                    "✅ IRRIGATION NOT REQUIRED"
                )


            st.write(
                response_text
            )

            st.metric(
                "Irrigation Probability",
                f"{voice_probability * 100:.1f}%"
            )


            # Text to speech
            tts = gTTS(
                text=response_text,
                lang="en",
                slow=False
            )

            audio_output = BytesIO()

            tts.write_to_fp(
                audio_output
            )

            audio_output.seek(0)

            st.audio(
                audio_output,
                format="audio/mp3"
            )


        else:

            response_text = (
                "I can help you with irrigation monitoring. "
                "Please ask a question related to irrigation, "
                "soil moisture, rainfall, crop conditions "
                "or watering."
            )

            st.subheader(
                "🤖 Assistant Response"
            )

            st.write(
                response_text
            )


            tts = gTTS(
                text=response_text,
                lang="en",
                slow=False
            )

            audio_output = BytesIO()

            tts.write_to_fp(
                audio_output
            )

            audio_output.seek(0)

            st.audio(
                audio_output,
                format="audio/mp3"
            )


    except sr.UnknownValueError:

        st.warning(
            "Sorry, I could not understand the audio. "
            "Please speak clearly and try again."
        )


    except sr.RequestError:

        st.error(
            "Speech recognition service is unavailable. "
            "Please check your internet connection."
        )


    except Exception as error:

        st.error(
            f"An error occurred while processing the voice: {error}"
        )


# ============================================================
# CURRENT FIELD CONDITIONS
# ============================================================

st.divider()

st.subheader(
    "🌾 Current Field Conditions"
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Soil Moisture",
        f"{soil_moisture}%"
    )

    st.metric(
        "Temperature",
        f"{temperature}°C"
    )


with col2:

    st.metric(
        "Humidity",
        f"{humidity}%"
    )

    st.metric(
        "Rainfall",
        f"{rainfall} mm"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    "This system is a machine-learning prototype. "
    "Irrigation decisions should be validated using "
    "actual field conditions and agronomic recommendations."
)