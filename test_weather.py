import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import weather


def _mock_response(status_code, json_data=None):
    response = MagicMock()
    response.status_code = status_code
    if json_data is not None:
        response.json.return_value = json_data
    return response


def test_fetch_weather_success():
    mock_json = {
        "main": {"temp": 29.4, "humidity": 55},
        "rain": {"1h": 1.2},
    }

    with patch("requests.get", return_value=_mock_response(200, mock_json)):
        result = weather.fetch_weather("Delhi", "fake-key")

    assert result == {"temperature": 29.4, "humidity": 55.0, "rainfall": 1.2}


def test_fetch_weather_no_rain_field_defaults_to_zero():
    mock_json = {"main": {"temp": 20.0, "humidity": 40}}

    with patch("requests.get", return_value=_mock_response(200, mock_json)):
        result = weather.fetch_weather("Pune", "fake-key")

    assert result["rainfall"] == 0.0


def test_fetch_weather_invalid_key():
    with patch("requests.get", return_value=_mock_response(401)):
        try:
            weather.fetch_weather("Delhi", "bad-key")
            assert False, "should have raised WeatherFetchError"
        except weather.WeatherFetchError as e:
            assert "API key" in str(e)


def test_fetch_weather_city_not_found():
    with patch("requests.get", return_value=_mock_response(404)):
        try:
            weather.fetch_weather("NotACity", "fake-key")
            assert False, "should have raised WeatherFetchError"
        except weather.WeatherFetchError as e:
            assert "not found" in str(e)
