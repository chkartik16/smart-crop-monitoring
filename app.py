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
import hashlib
import hmac
import streamlit.components.v1 as components

#twilio_client = Client(
#    os.getenv("TWILIO_ACCOUNT_SID"),
#       os.getenv("TWILIO_AUTH_TOKEN")
#)


# ============================================================
# LANGUAGE SUPPORT (English / Hindi)
# ============================================================

TEXT = {
    "en": {
        "hero_title": "🌱 Smart Crop Monitoring System",
        "hero_subtitle": "AI-powered irrigation prediction with Explainable AI, Telegram alerts and a voice assistant.",
        "tab_predict": "💧 Predict",
        "tab_voice": "🎙️ Voice Assistant",
        "tab_telegram": "📱 Telegram Alerts",
        "tab_about": "ℹ️ About",
        "crop_info": "🌾 Crop Information",
        "select_crop": "Select Crop",
        "select_growth_stage": "Select Growth Stage",
        "env_conditions": "🌦️ Environmental Conditions",
        "soil_moisture": "Soil Moisture (%)",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",
        "irrigation_decision": "💧 Irrigation Decision",
        "check_irrigation": "Check Irrigation",
        "analyzing": "Analyzing field conditions...",
        "irrigation_required": "🚨 IRRIGATION REQUIRED",
        "irrigation_required_msg": "The current field conditions indicate that the crop may require irrigation.",
        "irrigation_not_required": "✅ IRRIGATION NOT REQUIRED",
        "irrigation_not_required_msg": "The current field conditions do not indicate an immediate need for irrigation.",
        "irrigation_probability": "Irrigation Probability",
        "telegram_sent": "📱 Telegram irrigation alert sent to your connected Telegram account!",
        "telegram_not_sent": "Telegram alert could not be sent: {error}",
        "why_prediction": "🧠 Why did the model make this prediction?",
        "top_factors": "### Top Factors Influencing the Prediction",
        "feature_impact": "### 📊 Feature Impact",
        "shap_failed": "The model prediction was successful, but the SHAP explanation could not be generated.",
        "current_conditions": "🌾 Current Field Conditions",
        "lbl_soil_moisture": "Soil Moisture",
        "lbl_temperature": "Temperature",
        "lbl_humidity": "Humidity",
        "lbl_rainfall": "Rainfall",
        "connect_telegram_title": "📱 Connect Telegram Alerts",
        "connect_telegram_desc": "Connect your Telegram account to receive personal irrigation alerts.",
        "quick_connect": "⚡ Quick Connect (one click)",
        "quick_connect_desc": "Tap below and confirm in Telegram — no codes to copy.",
        "manual_connect": "🔧 Manual Connect (backup method)",
        "open_bot": "🤖 Open Telegram Bot",
        "manual_steps": "1. Open the Telegram bot using the button above.\n\n2. Press Start in Telegram.\n\n3. Return here and click Connect Telegram.",
        "connect_button": "🔗 Connect Telegram",
        "checking_connection": "Checking Telegram connection...",
        "connected_success": "✅ Telegram connected successfully!",
        "connected_info": "Your irrigation alerts will be sent only to your connected Telegram account.",
        "not_connected_warning": "Telegram connection not found. Please open the bot, press Start, and then click Connect Telegram again.",
        "status_connected": "Status: Telegram is connected for this session ✅",
        "status_not_connected": "Status: Telegram is not connected yet ⚠️",
        "voice_assistant_title": "🎙️ Voice Assistant",
        "voice_assistant_desc": "Click the microphone button and ask your irrigation question.",
        "voice_recorded": "Voice recorded successfully.",
        "transcribing": "Transcribing your question...",
        "your_question": "📝 Your Question",
        "thinking": "Thinking...",
        "assistant_response": "🤖 Assistant Response",
        "generating_voice": "Generating voice response...",
        "fallback_response": "I can help you with irrigation monitoring. Please ask a question related to irrigation, soil moisture, rainfall, crop conditions or watering.",
        "voice_not_understood": "Sorry, I could not understand the audio. Please speak clearly and try again.",
        "voice_service_down": "Speech recognition service is unavailable. Please check your internet connection.",
        "voice_error": "An error occurred while processing the voice: {error}",
        "about_title": "ℹ️ About this project",
        "about_body": """
        **Smart Crop Monitoring System** predicts whether irrigation is
        required using soil moisture, temperature, humidity, rainfall,
        crop type and growth stage.

        - 🤖 Machine Learning model for prediction
        - 🧠 SHAP-based Explainable AI
        - 📱 Multi-user Telegram alerts
        - 🎙️ Voice assistant (speech-to-text + text-to-speech)
        """,
        "disclaimer": "This system is a machine-learning prototype. Irrigation decisions should be validated using actual field conditions and agronomic recommendations.",
        "footer": 'Built with Python &amp; Streamlit · by Kartik · <a href="https://github.com/chkartik16" target="_blank">GitHub</a>',
    },
    "hi": {
        "hero_title": "🌱 स्मार्ट क्रॉप मॉनिटरिंग सिस्टम",
        "hero_subtitle": "एआई-आधारित सिंचाई पूर्वानुमान, व्याख्या योग्य एआई, टेलीग्राम अलर्ट और वॉइस असिस्टेंट के साथ।",
        "tab_predict": "💧 पूर्वानुमान",
        "tab_voice": "🎙️ वॉइस असिस्टेंट",
        "tab_telegram": "📱 टेलीग्राम अलर्ट",
        "tab_about": "ℹ️ जानकारी",
        "crop_info": "🌾 फसल जानकारी",
        "select_crop": "फसल चुनें",
        "select_growth_stage": "वृद्धि अवस्था चुनें",
        "env_conditions": "🌦️ पर्यावरणीय स्थितियाँ",
        "soil_moisture": "मिट्टी की नमी (%)",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "rainfall": "वर्षा (मिमी)",
        "irrigation_decision": "💧 सिंचाई निर्णय",
        "check_irrigation": "सिंचाई जांचें",
        "analyzing": "खेत की स्थिति का विश्लेषण हो रहा है...",
        "irrigation_required": "🚨 सिंचाई आवश्यक है",
        "irrigation_required_msg": "वर्तमान खेत की स्थिति दर्शाती है कि फसल को सिंचाई की आवश्यकता हो सकती है।",
        "irrigation_not_required": "✅ सिंचाई आवश्यक नहीं है",
        "irrigation_not_required_msg": "वर्तमान खेत की स्थिति तत्काल सिंचाई की आवश्यकता नहीं दर्शाती है।",
        "irrigation_probability": "सिंचाई संभावना",
        "telegram_sent": "📱 आपके जुड़े हुए टेलीग्राम खाते पर सिंचाई अलर्ट भेज दिया गया है!",
        "telegram_not_sent": "टेलीग्राम अलर्ट नहीं भेजा जा सका: {error}",
        "why_prediction": "🧠 मॉडल ने यह पूर्वानुमान क्यों दिया?",
        "top_factors": "### पूर्वानुमान को प्रभावित करने वाले मुख्य कारक",
        "feature_impact": "### 📊 फीचर प्रभाव",
        "shap_failed": "मॉडल पूर्वानुमान सफल रहा, लेकिन SHAP व्याख्या तैयार नहीं की जा सकी।",
        "current_conditions": "🌾 वर्तमान खेत की स्थिति",
        "lbl_soil_moisture": "मिट्टी की नमी",
        "lbl_temperature": "तापमान",
        "lbl_humidity": "आर्द्रता",
        "lbl_rainfall": "वर्षा",
        "connect_telegram_title": "📱 टेलीग्राम अलर्ट कनेक्ट करें",
        "connect_telegram_desc": "व्यक्तिगत सिंचाई अलर्ट पाने के लिए अपना टेलीग्राम खाता कनेक्ट करें।",
        "quick_connect": "⚡ त्वरित कनेक्ट (एक क्लिक)",
        "quick_connect_desc": "नीचे टैप करें और टेलीग्राम में पुष्टि करें — कोई कोड कॉपी करने की जरूरत नहीं।",
        "manual_connect": "🔧 मैन्युअल कनेक्ट (बैकअप तरीका)",
        "open_bot": "🤖 टेलीग्राम बॉट खोलें",
        "manual_steps": "1. ऊपर दिए गए बटन से टेलीग्राम बॉट खोलें।\n\n2. टेलीग्राम में Start दबाएं।\n\n3. यहाँ वापस आएं और Connect Telegram पर क्लिक करें।",
        "connect_button": "🔗 टेलीग्राम कनेक्ट करें",
        "checking_connection": "टेलीग्राम कनेक्शन जांचा जा रहा है...",
        "connected_success": "✅ टेलीग्राम सफलतापूर्वक कनेक्ट हो गया!",
        "connected_info": "आपके सिंचाई अलर्ट केवल आपके जुड़े हुए टेलीग्राम खाते पर भेजे जाएंगे।",
        "not_connected_warning": "टेलीग्राम कनेक्शन नहीं मिला। कृपया बॉट खोलें, Start दबाएं, और फिर से Connect Telegram पर क्लिक करें।",
        "status_connected": "स्थिति: टेलीग्राम इस सत्र के लिए कनेक्ट है ✅",
        "status_not_connected": "स्थिति: टेलीग्राम अभी कनेक्ट नहीं है ⚠️",
        "voice_assistant_title": "🎙️ वॉइस असिस्टेंट",
        "voice_assistant_desc": "माइक्रोफोन बटन दबाएं और अपना सिंचाई संबंधी सवाल पूछें।",
        "voice_recorded": "आवाज़ सफलतापूर्वक रिकॉर्ड हो गई।",
        "transcribing": "आपके सवाल को टेक्स्ट में बदला जा रहा है...",
        "your_question": "📝 आपका सवाल",
        "thinking": "सोचा जा रहा है...",
        "assistant_response": "🤖 असिस्टेंट का जवाब",
        "generating_voice": "आवाज़ का जवाब तैयार किया जा रहा है...",
        "fallback_response": "मैं सिंचाई निगरानी में आपकी मदद कर सकता हूँ। कृपया सिंचाई, मिट्टी की नमी, वर्षा, फसल की स्थिति या पानी देने से जुड़ा सवाल पूछें।",
        "voice_not_understood": "क्षमा करें, आवाज़ समझ नहीं आई। कृपया स्पष्ट रूप से बोलें और फिर से प्रयास करें।",
        "voice_service_down": "स्पीच रिकग्निशन सेवा उपलब्ध नहीं है। कृपया अपना इंटरनेट कनेक्शन जांचें।",
        "voice_error": "आवाज़ प्रोसेस करते समय एक त्रुटि हुई: {error}",
        "about_title": "ℹ️ इस प्रोजेक्ट के बारे में",
        "about_body": """
        **स्मार्ट क्रॉप मॉनिटरिंग सिस्टम** मिट्टी की नमी, तापमान, आर्द्रता,
        वर्षा, फसल के प्रकार और वृद्धि अवस्था के आधार पर बताता है कि
        सिंचाई की आवश्यकता है या नहीं।

        - 🤖 पूर्वानुमान के लिए मशीन लर्निंग मॉडल
        - 🧠 SHAP आधारित व्याख्या योग्य एआई
        - 📱 मल्टी-यूज़र टेलीग्राम अलर्ट
        - 🎙️ वॉइस असिस्टेंट (स्पीच-टू-टेक्स्ट + टेक्स्ट-टू-स्पीच)
        """,
        "disclaimer": "यह सिस्टम एक मशीन-लर्निंग प्रोटोटाइप है। सिंचाई संबंधी निर्णय वास्तविक खेत की स्थिति और कृषि विशेषज्ञ की सलाह के आधार पर ही लें।",
        "footer": 'Python और Streamlit से बना · Kartik द्वारा · <a href="https://github.com/chkartik16" target="_blank">GitHub</a>',
    },
}


def t(key, **kwargs):
    lang = st.session_state.get("lang", "en")
    text = TEXT[lang].get(key, TEXT["en"].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text


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
# LANGUAGE TOGGLE + HERO SECTION
# ============================================================

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

lang_col1, lang_col2, lang_spacer = st.columns([1, 1, 6])

with lang_col1:
    if st.button("English", use_container_width=True,
                 type="primary" if st.session_state["lang"] == "en" else "secondary"):
        st.session_state["lang"] = "en"
        st.rerun()

with lang_col2:
    if st.button("हिंदी", use_container_width=True,
                 type="primary" if st.session_state["lang"] == "hi" else "secondary"):
        st.session_state["lang"] = "hi"
        st.rerun()

st.markdown(
    f"""
    <div class="hero">
        <h1>{t('hero_title')}</h1>
        <p>{t('hero_subtitle')}</p>
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
# TELEGRAM LOGIN WIDGET — ONE-CLICK CONNECT
# ============================================================
# Requires: bot domain must be linked once via BotFather
# (send /setdomain to @BotFather and give it this app's URL).
# On click, Telegram redirects back with signed user data in
# the URL — we verify the signature with the bot token and
# use the returned Telegram user id directly as the chat_id.
# ============================================================

def verify_telegram_login(auth_data, bot_token):

    data = dict(auth_data)
    received_hash = data.pop("hash", None)

    if not received_hash:
        return False

    check_string = "\n".join(
        f"{key}={data[key]}" for key in sorted(data.keys())
    )

    secret_key = hashlib.sha256(bot_token.encode()).digest()

    computed_hash = hmac.new(
        secret_key,
        check_string.encode(),
        hashlib.sha256
    ).hexdigest()

    return computed_hash == received_hash


def render_telegram_login_widget(bot_username, app_url):

    widget_html = f"""
    <script async src="https://telegram.org/js/telegram-widget.js?22"
        data-telegram-login="{bot_username}"
        data-size="large"
        data-radius="10"
        data-auth-url="{app_url}"
        data-request-access="write">
    </script>
    """

    components.html(widget_html, height=60)


# ============================================================
# TABS — main navigation instead of one long scroll
# ============================================================

tab_predict, tab_voice, tab_telegram, tab_about = st.tabs(
    [t("tab_predict"), t("tab_voice"), t("tab_telegram"), t("tab_about")]
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

        st.subheader(t("crop_info"))

        crop_type = st.selectbox(
            t("select_crop"),
            ["Wheat", "Rice", "Maize", "Cotton"],
            key="crop_type"
        )

        growth_stage = st.selectbox(
            t("select_growth_stage"),
            ["Seedling", "Vegetative", "Flowering", "Maturity"],
            key="growth_stage"
        )

    with right:

        st.subheader(t("env_conditions"))

        soil_moisture = st.number_input(
            t("soil_moisture"),
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
            key="soil_moisture"
        )

        temperature = st.number_input(
            t("temperature"),
            min_value=-10.0,
            max_value=60.0,
            value=30.0,
            step=1.0,
            key="temperature"
        )

        humidity = st.number_input(
            t("humidity"),
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="humidity"
        )

        rainfall = st.number_input(
            t("rainfall"),
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

    st.subheader(t("irrigation_decision"))

    if st.button(t("check_irrigation"), use_container_width=True, type="primary"):

        with st.spinner(t("analyzing")):

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        result_col, prob_col = st.columns([2, 1])

        with result_col:

            if prediction == 1:

                st.error(t("irrigation_required"))

                st.write(t("irrigation_required_msg"))

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

                    st.success(t("telegram_sent"))

                except Exception as error:

                    st.warning(t("telegram_not_sent", error=error))

            else:

                st.success(t("irrigation_not_required"))

                st.write(t("irrigation_not_required_msg"))

        with prob_col:

            st.metric(
                t("irrigation_probability"),
                f"{probability * 100:.1f}%"
            )

            st.progress(min(max(probability, 0.0), 1.0))

        # ========================================================
        # SHAP EXPLANATION
        # ========================================================

        st.divider()

        st.subheader(t("why_prediction"))

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

            st.write(t("top_factors"))

            friendly_names_en = {
                "Soil_Moisture": "Soil Moisture",
                "Temperature": "Temperature",
                "Humidity": "Humidity",
                "Rainfall": "Rainfall",
                "Crop_Type": "Crop Type",
                "Growth_Stage": "Growth Stage",
            }

            friendly_names_hi = {
                "Soil_Moisture": "मिट्टी की नमी",
                "Temperature": "तापमान",
                "Humidity": "आर्द्रता",
                "Rainfall": "वर्षा",
                "Crop_Type": "फसल का प्रकार",
                "Growth_Stage": "वृद्धि अवस्था",
            }

            friendly_names = (
                friendly_names_hi
                if st.session_state.get("lang") == "hi"
                else friendly_names_en
            )

            for _, row in top_features.iterrows():

                feature = row["Feature"]
                shap_value = row["SHAP_Value"]

                feature = feature.replace("categorical__", "")
                feature = feature.replace("remainder__", "")

                friendly_name = feature

                for key, label in friendly_names.items():
                    if key in feature:
                        friendly_name = label
                        break

                if st.session_state.get("lang") == "hi":

                    if shap_value > 0:
                        explanation_text = f"{friendly_name} ने सिंचाई की आवश्यकता को बढ़ाया।"
                    else:
                        explanation_text = f"{friendly_name} ने सिंचाई की आवश्यकता को घटाया।"

                else:

                    if shap_value > 0:
                        explanation_text = f"{friendly_name} increased the irrigation requirement."
                    else:
                        explanation_text = f"{friendly_name} reduced the irrigation requirement."

                st.write(f"🔹 **{explanation_text}**")

            st.write(t("feature_impact"))

            chart_data = top_features.set_index("Feature")["SHAP_Value"]
            st.bar_chart(chart_data)

        except Exception as error:

            st.warning(t("shap_failed"))

            st.write(f"Technical details: {error}")

    st.divider()

    st.subheader(t("current_conditions"))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(t("lbl_soil_moisture"), f"{soil_moisture}%")

    with col2:
        st.metric(t("lbl_temperature"), f"{temperature}°C")

    with col3:
        st.metric(t("lbl_humidity"), f"{humidity}%")

    with col4:
        st.metric(t("lbl_rainfall"), f"{rainfall} mm")


# ============================================================
# TELEGRAM TAB
# ============================================================

with tab_telegram:

    st.subheader(t("connect_telegram_title"))

    st.write(t("connect_telegram_desc"))

    # ------------------------------------------------------
    # Check for a Telegram Login Widget redirect on this load
    # ------------------------------------------------------

    query_params = st.query_params

    if "hash" in query_params and not st.session_state.get("telegram_chat_id"):

        bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
        auth_data = {k: v for k, v in query_params.items()}

        if verify_telegram_login(auth_data, bot_token):

            st.session_state["telegram_chat_id"] = auth_data["id"]
            save_telegram_user(auth_data["id"])
            st.query_params.clear()
            st.success(t("connected_success"))
            st.info(t("connected_info"))

        else:

            st.warning(t("not_connected_warning"))

    st.markdown(f"#### {t('quick_connect')}")
    st.caption(t("quick_connect_desc"))

    # bot_username must exactly match your bot's @username casing
    # (lowercase 'b' confirmed via BotFather). app_url must exactly
    # match the domain linked via BotFather /setdomain.
    render_telegram_login_widget(
        bot_username="SmartCropMonitoringbot",
        app_url="https://smart-crop-monitoring-mpbhpxpwniuto5kv2ygto4.streamlit.app"
    )

    st.divider()

    with st.expander(t("manual_connect")):

        if "telegram_connection_code" not in st.session_state:

            st.session_state["telegram_connection_code"] = uuid.uuid4().hex[:12]

        connection_code = st.session_state["telegram_connection_code"]

        telegram_bot_username = "SmartCropMonitoringbot"

        telegram_url = (
            f"https://t.me/{telegram_bot_username}?start={connection_code}"
        )

        st.link_button(
            t("open_bot"),
            telegram_url,
            use_container_width=True
        )

        st.info(t("manual_steps"))

        if st.button(t("connect_button"), use_container_width=True, type="primary"):

            with st.spinner(t("checking_connection")):
                chat_id = get_telegram_chat_id(connection_code)

            if chat_id:

                st.session_state["telegram_chat_id"] = chat_id
                save_telegram_user(chat_id)

                st.success(t("connected_success"))
                st.info(t("connected_info"))

            else:

                st.warning(t("not_connected_warning"))

    if st.session_state.get("telegram_chat_id"):
        st.success(t("status_connected"))
    else:
        st.warning(t("status_not_connected"))


# ============================================================
# VOICE ASSISTANT TAB
# ============================================================

with tab_voice:

    st.subheader(t("voice_assistant_title"))

    st.write(t("voice_assistant_desc"))

    audio = mic_recorder(
        start_prompt="🎙️ Start Recording" if st.session_state.get("lang") != "hi" else "🎙️ रिकॉर्डिंग शुरू करें",
        stop_prompt="⏹️ Stop Recording" if st.session_state.get("lang") != "hi" else "⏹️ रिकॉर्डिंग रोकें",
        key="voice_recorder"
    )

    if audio:

        st.success(t("voice_recorded"))

        try:

            with st.spinner(t("transcribing")):

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

                speech_lang = "hi-IN" if st.session_state.get("lang") == "hi" else "en-IN"

                text = recognizer.recognize_google(
                    recorded_audio,
                    language=speech_lang
                )

            st.subheader(t("your_question"))
            st.write(text)

            question = text.lower()

            irrigation_words = [
                "irrigation", "water", "watering", "irrigate",
                "paani", "moisture", "rainfall", "crop",
                "सिंचाई", "पानी", "बारिश", "फसल", "नमी"
            ]

            if any(word in question for word in irrigation_words):

                with st.spinner(t("thinking")):

                    voice_prediction = model.predict(input_data)[0]
                    voice_probability = model.predict_proba(input_data)[0][1]

                st.subheader(t("assistant_response"))

                is_hindi = st.session_state.get("lang") == "hi"

                if voice_prediction == 1:

                    if is_hindi:
                        response_text = (
                            f"आपकी {crop_type} फसल के लिए सिंचाई आवश्यक है। "
                            f"सिंचाई की संभावना {voice_probability * 100:.1f} प्रतिशत है। "
                            f"वर्तमान खेत की स्थिति दर्शाती है कि फसल को पानी की आवश्यकता हो सकती है।"
                        )
                    else:
                        response_text = (
                            f"Irrigation is required for your {crop_type} crop. "
                            f"The irrigation probability is {voice_probability * 100:.1f} percent. "
                            f"The current field conditions indicate that the crop may need water."
                        )

                    st.error(t("irrigation_required"))

                else:

                    if is_hindi:
                        response_text = (
                            f"आपकी {crop_type} फसल के लिए सिंचाई आवश्यक नहीं है। "
                            f"सिंचाई की संभावना {voice_probability * 100:.1f} प्रतिशत है। "
                            f"वर्तमान खेत की स्थिति तत्काल सिंचाई की आवश्यकता नहीं दर्शाती है।"
                        )
                    else:
                        response_text = (
                            f"Irrigation is not required for your {crop_type} crop. "
                            f"The irrigation probability is {voice_probability * 100:.1f} percent. "
                            f"The current field conditions do not indicate an immediate need for irrigation."
                        )

                    st.success(t("irrigation_not_required"))

                st.write(response_text)

                st.metric(t("irrigation_probability"), f"{voice_probability * 100:.1f}%")

                with st.spinner(t("generating_voice")):

                    tts_lang = "hi" if is_hindi else "en"
                    tts = gTTS(text=response_text, lang=tts_lang, slow=False)
                    audio_output = BytesIO()
                    tts.write_to_fp(audio_output)
                    audio_output.seek(0)

                st.audio(audio_output, format="audio/mp3")

            else:

                response_text = t("fallback_response")

                st.subheader(t("assistant_response"))
                st.write(response_text)

                tts_lang = "hi" if st.session_state.get("lang") == "hi" else "en"
                tts = gTTS(text=response_text, lang=tts_lang, slow=False)
                audio_output = BytesIO()
                tts.write_to_fp(audio_output)
                audio_output.seek(0)

                st.audio(audio_output, format="audio/mp3")

        except sr.UnknownValueError:

            st.warning(t("voice_not_understood"))

        except sr.RequestError:

            st.error(t("voice_service_down"))

        except Exception as error:

            st.error(t("voice_error", error=error))


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.subheader(t("about_title"))

    st.write(t("about_body"))

    st.info(t("disclaimer"))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <footer class="app-footer">
        {t('footer')}
    </footer>
    """,
    unsafe_allow_html=True
)