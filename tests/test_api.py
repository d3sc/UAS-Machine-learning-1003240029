"""Pengujian API Estimasi Harga Kendaraan Bekas."""

VALID_PAYLOAD = {
    "make": "Ford",
    "model": "Fiesta",
    "year": 2018,
    "transmission": "Manual",
    "fuel_type": "Petrol",
    "mileage": 65000,
    "tax": 145,
    "mpg": 55.4,
    "engine_size": 1.5,
}


def test_root_returns_service_info(client):
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["project"] == "Estimasi Harga Kendaraan Bekas"
    assert body["endpoint"] == "POST /predict-harga"


def test_health_returns_200_and_model_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_valid_prediction_returns_expected_schema(client):
    response = client.post(
        "/predict-harga",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["estimasi_harga"] >= 0
    assert body["mata_uang"] == "GBP"

    assert (
        body["rentang_perkiraan"]["minimum"]
        <= body["estimasi_harga"]
    )

    assert (
        body["rentang_perkiraan"]["maximum"]
        >= body["estimasi_harga"]
    )

    assert body["keyakinan"] in {
        "tinggi",
        "sedang",
        "rendah",
    }

    assert "model_version" in body


def test_missing_required_field_returns_422(client):
    invalid = VALID_PAYLOAD.copy()

    invalid.pop("year")

    response = client.post(
        "/predict-harga",
        json=invalid,
    )

    assert response.status_code == 422


def test_unknown_enum_returns_422(client):
    invalid = {
        **VALID_PAYLOAD,
        "transmission": "Semi Otomatis",
    }

    response = client.post(
        "/predict-harga",
        json=invalid,
    )

    assert response.status_code == 422


def test_out_of_range_value_returns_422(client):
    invalid = {
        **VALID_PAYLOAD,
        "mileage": -1,
    }

    response = client.post(
        "/predict-harga",
        json=invalid,
    )

    assert response.status_code == 422


def test_older_vehicle_is_predicted_cheaper(client):
    newer = {
        **VALID_PAYLOAD,
        "year": 2020,
    }

    older = {
        **VALID_PAYLOAD,
        "year": 2010,
    }

    newer_price = client.post(
        "/predict-harga",
        json=newer,
    ).json()["estimasi_harga"]

    older_price = client.post(
        "/predict-harga",
        json=older,
    ).json()["estimasi_harga"]

    assert older_price < newer_price


def test_higher_mileage_is_not_predicted_more_expensive(client):
    low_mileage = {
        **VALID_PAYLOAD,
        "mileage": 25000,
    }

    high_mileage = {
        **VALID_PAYLOAD,
        "mileage": 175000,
    }

    low_price = client.post(
        "/predict-harga",
        json=low_mileage,
    ).json()["estimasi_harga"]

    high_price = client.post(
        "/predict-harga",
        json=high_mileage,
    ).json()["estimasi_harga"]

    assert high_price <= low_price