"""
weather.py
==========
Thin wrapper around the OpenWeatherMap "Current Weather" free-tier API.
Requires an API key stored in Streamlit secrets as OPENWEATHER_API_KEY.
Get a free key at https://openweathermap.org/api (no credit card needed
for the free tier, ~1000 calls/day).
"""

import requests


class WeatherFetchError(Exception):
    pass


def fetch_weather(city, api_key):
    """
    Fetches current weather for a city and maps it to the fields the
    irrigation model expects: temperature (°C), humidity (%),
    rainfall (mm, last 1 hour if available else 0.0).

    Returns a dict: {"temperature": float, "humidity": float, "rainfall": float}
    Raises WeatherFetchError with a human-readable message on failure.
    """

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
    except requests.RequestException as error:
        raise WeatherFetchError(f"Could not reach weather service: {error}")

    if response.status_code == 401:
        raise WeatherFetchError("Invalid OpenWeatherMap API key.")

    if response.status_code == 404:
        raise WeatherFetchError(f"City '{city}' not found.")

    if response.status_code != 200:
        raise WeatherFetchError(
            f"Weather service returned an error (status {response.status_code})."
        )

    data = response.json()

    temperature = data.get("main", {}).get("temp")
    humidity = data.get("main", {}).get("humidity")
    rainfall = data.get("rain", {}).get("1h", 0.0)

    if temperature is None or humidity is None:
        raise WeatherFetchError("Weather data was incomplete for this city.")

    return {
        "temperature": round(float(temperature), 1),
        "humidity": round(float(humidity), 1),
        "rainfall": round(float(rainfall), 1),
    }
