from fastapi.testclient import TestClient

from app import main
from app.main import app
from app.schemas import Prediction

client = TestClient(app)


def test_health_mode_mock():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["mode"] == "mock"


def test_detection_francais():
    r = client.post("/detect-language",
                    json={"text": "Je ne peux plus acceder a mon compte."})
    assert r.status_code == 200
    assert r.json()["language"] == "fr"


def test_detection_arabe():
    r = client.post("/detect-language",
                    json={"text": "لا أستطيع الدخول إلى حسابي"})
    assert r.status_code == 200
    assert r.json()["language"] == "ar"


def test_detection_anglais():
    r = client.post("/detect-language",
                    json={"text": "I cannot log into my account."})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "en"
    assert body["provider"] == "mock"
    assert body["latency_ms"] >= 0
    assert body["requires_review"] is False


def test_texte_trop_court_rejete():
    assert client.post("/detect-language", json={"text": "ab"}).status_code == 422


def test_champ_text_absent_rejete():
    assert client.post("/detect-language", json={}).status_code == 422


def test_fournisseur_en_panne_renvoie_503(monkeypatch):
    def boom(text):
        raise RuntimeError("secret interne: https://prive.example/hf_abc123")

    monkeypatch.setattr(main.provider, "detect_language", boom)
    r = client.post("/detect-language", json={"text": "Bonjour tout le monde"})
    assert r.status_code == 503
    assert "hf_" not in r.text and "prive" not in r.text  # aucun secret expose


def test_confiance_faible_demande_revue(monkeypatch):
    monkeypatch.setattr(
        main.provider, "detect_language",
        lambda text: [Prediction(label="fr", score=0.40),
                      Prediction(label="en", score=0.35)],
    )
    r = client.post("/detect-language", json={"text": "texte ambigu"})
    assert r.status_code == 200
    assert r.json()["requires_review"] is True
