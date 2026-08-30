#from twilio.rest import Client
import os
import streamlit as st
import pandas as pd
import joblib
import shap
import speech_recognition as sr
import requests
import json
import uuid

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
# PAGE CONFIG  (wide layout instead of centered)
# ============================================================

st.set_page_config(
    page_title="Smart Crop Monitoring",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# GLOBAL CSS POLISH
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    div[data-testid="stMetric"] {
        background-color: #F1F8E9;
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 14px 16px;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stLinkButton"] > a {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.25);
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    .hero {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 28px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 4px;
    }

    .hero p {
        color: #E8F5E9;
        margin-bottom: 0;
        font-size: 1.05rem;
    }

    footer.app-footer {
        text-align: center;
        color: #6b6b6b;
        font-size: 0.85rem;
        padding-top: 2rem;
    }

    footer.app-footer a {
        color: #2E7D32;
        text-decoration: none;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌱 Smart Crop Monitoring System</h1>
        <p>AI-powered irrigation prediction with Explainable AI, Telegram alerts and a voice assistant.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GET TELEGRAM CHAT ID USING UNIQUE CONNECTION CODE
# ============================================================

def get_telegram_chat_id(connection_code):

    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]

    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/getUpdates"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    if not data["ok"]:
        return None

    for update in reversed(data["result"]):

        message = update.get("message")

        if not message:
            continue

        text = message.get("text", "")

        expected_text = f"/start {connection_code}"

        if text.strip() == expected_text:

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
# TABS — main navigation instead of one long scroll
# ============================================================

tab_predict, tab_voice, tab_telegram, tab_about = st.tabs(
    ["💧 Predict", "🎙️ Voice Assistant", "📱 Telegram Alerts", "ℹ️ About"]
)


# ============================================================
# SHARED INPUT STATE
# Rendered once inside the Predict tab, but referenced by
# the Voice Assistant tab too — so collect it before the tabs
# using session_state to avoid duplicate widgets.
# ============================================================

with tab_predict:

    left, right = st.columns([1, 1], gap="large")

    with left:

        st.subheader("🌾 Crop Information")

        crop_type = st.selectbox(
            "Select Crop",
            ["Wheat", "Rice", "Maize", "Cotton"],
            key="crop_type"
        )

        growth_stage = st.selectbox(
            "Select Growth Stage",
            ["Seedling", "Vegetative", "Flowering", "Maturity"],
            key="growth_stage"
        )

    with right:

        st.subheader("🌦️ Environmental Conditions")

        soil_moisture = st.number_input(
            "Soil Moisture (%)",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            key="soil_moisture"
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
            key="temperature"
        )

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="humidity"
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=500.0,
            value=0.0,
            step=1.0,
            key="rainfall"
        )

    input_data = pd.DataFrame({
        "Soil_Moisture": [soil_moisture],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "Rainfall": [rainfall],
        "Crop_Type": [crop_type],
        "Growth_Stage": [growth_stage]
    })

    st.divider()

    st.subheader("💧 Irrigation Decision")

    if st.button("Check Irrigation", use_container_width=True, type="primary"):

        with st.spinner("Analyzing field conditions..."):

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        result_col, prob_col = st.columns([2, 1])

        with result_col:

            if prediction == 1:

                st.error("🚨 IRRIGATION REQUIRED")

                st.write(
                    "The current field conditions indicate "
                    "that the crop may require irrigation."
                )

                try:

                    chat_id = st.session_state.get("telegram_chat_id")

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

                st.success("✅ IRRIGATION NOT REQUIRED")

                st.write(
                    "The current field conditions do not indicate "
                    "an immediate need for irrigation."
                )

        with prob_col:

            st.metric(
                "Irrigation Probability",
                f"{probability * 100:.1f}%"
            )

            st.progress(min(max(probability, 0.0), 1.0))

        # ========================================================
        # SHAP EXPLANATION
        # ========================================================

        st.divider()

        st.subheader("🧠 Why did the model make this prediction?")

        try:

            transformed_input = preprocessor.transform(input_data)
            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_input)

            if isinstance(shap_values, list):

                shap_array = shap_values[prediction][0]

            elif len(shap_values.shape) == 3:

                shap_array = shap_values[0, :, prediction]

            else:

                shap_array = shap_values[0]

            explanation = pd.DataFrame({
                "Feature": feature_names,
                "SHAP_Value": shap_array
            })

            explanation["Absolute_Impact"] = explanation["SHAP_Value"].abs()
            explanation = explanation.sort_values("Absolute_Impact", ascending=False)
            top_features = explanation.head(5)

            st.write("### Top Factors Influencing the Prediction")

            for _, row in top_features.iterrows():

                feature = row["Feature"]
                shap_value = row["SHAP_Value"]

                feature = feature.replace("categorical__", "")
                feature = feature.replace("remainder__", "")

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
                    explanation_text = f"{friendly_name} increased the irrigation requirement."
                else:
                    explanation_text = f"{friendly_name} reduced the irrigation requirement."

                st.write(f"🔹 **{explanation_text}**")

            st.write("### 📊 Feature Impact")

            chart_data = top_features.set_index("Feature")["SHAP_Value"]
            st.bar_chart(chart_data)

        except Exception as error:

            st.warning(
                "The model prediction was successful, "
                "but the SHAP explanation could not be generated."
            )

            st.write(f"Technical details: {error}")

    st.divider()

    st.subheader("🌾 Current Field Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Soil Moisture", f"{soil_moisture}%")

    with col2:
        st.metric("Temperature", f"{temperature}°C")

    with col3:
        st.metric("Humidity", f"{humidity}%")

    with col4:
        st.metric("Rainfall", f"{rainfall} mm")


# ============================================================
# TELEGRAM TAB
# ============================================================

with tab_telegram:

    st.subheader("📱 Connect Telegram Alerts")

    st.write(
        "Connect your Telegram account to receive personal irrigation alerts."
    )

    if "telegram_connection_code" not in st.session_state:

        st.session_state["telegram_connection_code"] = uuid.uuid4().hex[:12]

    connection_code = st.session_state["telegram_connection_code"]

    telegram_bot_username = "SmartCropMonitoringBot"

    telegram_url = (
        f"https://t.me/{telegram_bot_username}?start={connection_code}"
    )

    st.link_button(
        "🤖 Open Telegram Bot",
        telegram_url,
        use_container_width=True
    )

    st.info(
        "1. Open the Telegram bot using the button above.\n\n"
        "2. Press Start in Telegram.\n\n"
        "3. Return here and click Connect Telegram."
    )

    if st.button("🔗 Connect Telegram", use_container_width=True, type="primary"):

        with st.spinner("Checking Telegram connection..."):
            chat_id = get_telegram_chat_id(connection_code)

        if chat_id:

            st.session_state["telegram_chat_id"] = chat_id

            st.success("✅ Telegram connected successfully!")

            st.info(
                "Your irrigation alerts will be sent only "
                "to your connected Telegram account."
            )

        else:

            st.warning(
                "Telegram connection not found. "
                "Please open the bot, press Start, "
                "and then click Connect Telegram again."
            )

    if st.session_state.get("telegram_chat_id"):
        st.success("Status: Telegram is connected for this session ✅")
    else:
        st.warning("Status: Telegram is not connected yet ⚠️")


# ============================================================
# VOICE ASSISTANT TAB
# ============================================================

with tab_voice:

    st.subheader("🎙️ Voice Assistant")

    st.write(
        "Click the microphone button and ask your irrigation question."
    )

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        key="voice_recorder"
    )

    if audio:

        st.success("Voice recorded successfully.")

        try:

            with st.spinner("Transcribing your question..."):

                audio_bytes = audio["bytes"]

                audio_segment = AudioSegment.from_file(
                    BytesIO(audio_bytes),
                    format="webm"
                )

                wav_buffer = BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)

                recognizer = sr.Recognizer()

                with sr.AudioFile(wav_buffer) as source:
                    recorded_audio = recognizer.record(source)

                text = recognizer.recognize_google(
                    recorded_audio,
                    language="en-IN"
                )

            st.subheader("📝 Your Question")
            st.write(text)

            question = text.lower()

            irrigation_words = [
                "irrigation", "water", "watering", "irrigate",
                "paani", "moisture", "rainfall", "crop"
            ]

            if any(word in question for word in irrigation_words):

                with st.spinner("Thinking..."):

                    voice_prediction = model.predict(input_data)[0]
                    voice_probability = model.predict_proba(input_data)[0][1]

                st.subheader("🤖 Assistant Response")

                if voice_prediction == 1:

                    response_text = (
                        f"Irrigation is required for your {crop_type} crop. "
                        f"The irrigation probability is {voice_probability * 100:.1f} percent. "
                        f"The current field conditions indicate that the crop may need water."
                    )

                    st.error("🚨 IRRIGATION REQUIRED")

                else:

                    response_text = (
                        f"Irrigation is not required for your {crop_type} crop. "
                        f"The irrigation probability is {voice_probability * 100:.1f} percent. "
                        f"The current field conditions do not indicate an immediate need for irrigation."
                    )

                    st.success("✅ IRRIGATION NOT REQUIRED")

                st.write(response_text)

                st.metric("Irrigation Probability", f"{voice_probability * 100:.1f}%")

                with st.spinner("Generating voice response..."):

                    tts = gTTS(text=response_text, lang="en", slow=False)
                    audio_output = BytesIO()
                    tts.write_to_fp(audio_output)
                    audio_output.seek(0)

                st.audio(audio_output, format="audio/mp3")

            else:

                response_text = (
                    "I can help you with irrigation monitoring. "
                    "Please ask a question related to irrigation, "
                    "soil moisture, rainfall, crop conditions or watering."
                )

                st.subheader("🤖 Assistant Response")
                st.write(response_text)

                tts = gTTS(text=response_text, lang="en", slow=False)
                audio_output = BytesIO()
                tts.write_to_fp(audio_output)
                audio_output.seek(0)

                st.audio(audio_output, format="audio/mp3")

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

            st.error(f"An error occurred while processing the voice: {error}")


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader("ℹ️ About this project")

    st.write(
        """
        **Smart Crop Monitoring System** predicts whether irrigation is
        required using soil moisture, temperature, humidity, rainfall,
        crop type and growth stage.

        - 🤖 Machine Learning model for prediction
        - 🧠 SHAP-based Explainable AI
        - 📱 Multi-user Telegram alerts
        - 🎙️ Voice assistant (speech-to-text + text-to-speech)
        """
    )

    st.info(
        "This system is a machine-learning prototype. "
        "Irrigation decisions should be validated using "
        "actual field conditions and agronomic recommendations."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <footer class="app-footer">
        Built with Python &amp; Streamlit · by Kartik ·
        <a href="https://github.com/chkartik16" target="_blank">GitHub</a>
    </footer>
    """,
    unsafe_allow_html=True
)