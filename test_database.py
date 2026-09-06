"""
Tests for database.py. Uses a temporary SQLite file so it never
touches the real smart_crop.db used by the running app.
"""

import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Reload the database module pointed at a temp file for isolation."""
    import database as db_module

    test_db_path = tmp_path / "test_smart_crop.db"
    monkeypatch.setattr(db_module, "DB_PATH", str(test_db_path))

    db_module.init_db()
    yield db_module


def test_create_and_verify_user(db):
    assert db.create_user("alice", "secret123") is True
    assert db.verify_user("alice", "secret123") is True
    assert db.verify_user("alice", "wrongpassword") is False


def test_duplicate_signup_rejected(db):
    assert db.create_user("bob", "pass1") is True
    assert db.create_user("bob", "pass2") is False


def test_telegram_chat_id_roundtrip(db):
    db.create_user("carol", "pw")
    assert db.get_telegram_chat_id("carol") is None

    db.set_telegram_chat_id("carol", "987654321")
    assert db.get_telegram_chat_id("carol") == "987654321"


def test_prediction_history_logging(db):
    db.create_user("dave", "pw")

    db.log_prediction("dave", "Wheat", "Seedling", 30.0, 30.0, 50.0, 0.0, 1, 0.8)
    db.log_prediction("dave", "Rice", "Vegetative", 65.0, 28.0, 70.0, 10.0, 0, 0.15)

    history = db.get_user_history("dave")
    assert len(history) == 2
    assert history[0]["crop_type"] == "Rice"  # most recent first
    assert history[1]["crop_type"] == "Wheat"


def test_history_is_per_user(db):
    db.create_user("erin", "pw")
    db.create_user("frank", "pw")

    db.log_prediction("erin", "Maize", "Flowering", 40.0, 32.0, 60.0, 2.0, 1, 0.7)

    assert len(db.get_user_history("erin")) == 1
    assert len(db.get_user_history("frank")) == 0
