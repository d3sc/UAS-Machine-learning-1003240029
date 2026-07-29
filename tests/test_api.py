VALID_PAYLOAD = {
    "brand": "Toyota",
    "car_model": "Corolla",
    "model_year": 2018,
    "transmission": "Automatic",
    "body_type": "Sedan",
    "fuel_type": "Petrol",
    "engine_capacity": 1500,
    "kilometers_run": 65000,
}


def test_root_returns_service_info(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["endpoint_prediksi"] == "POST /predict-harga"


def test_health_returns_200_and_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_valid_prediction_returns_expected_schema(client):
    response = client.post("/predict-harga", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["estimasi_harga"] >= 0
    assert body["mata_uang"] == "IDR"
    assert body["kurs_bdt_ke_idr"] > 0
    assert body["rentang_perkiraan"]["minimum"] <= body["estimasi_harga"]
    assert body["rentang_perkiraan"]["maximum"] >= body["estimasi_harga"]
    assert body["keyakinan"] in {"tinggi", "sedang", "rendah"}


def test_missing_required_field_returns_422(client):
    invalid = VALID_PAYLOAD.copy()
    invalid.pop("model_year")
    response = client.post("/predict-harga", json=invalid)
    assert response.status_code == 422


def test_unknown_enum_returns_422(client):
    invalid = {**VALID_PAYLOAD, "transmission": "Semi Otomatis"}
    response = client.post("/predict-harga", json=invalid)
    assert response.status_code == 422


def test_out_of_range_value_returns_422(client):
    invalid = {**VALID_PAYLOAD, "kilometers_run": -1}
    response = client.post("/predict-harga", json=invalid)
    assert response.status_code == 422


def test_older_vehicle_is_predicted_cheaper(client):
    newer = {**VALID_PAYLOAD, "model_year": 2020}
    older = {**VALID_PAYLOAD, "model_year": 2010}
    newer_price = client.post("/predict-harga", json=newer).json()["estimasi_harga"]
    older_price = client.post("/predict-harga", json=older).json()["estimasi_harga"]
    assert older_price < newer_price


def test_higher_mileage_is_not_predicted_more_expensive(client):
    low_mileage = {**VALID_PAYLOAD, "kilometers_run": 25000}
    high_mileage = {**VALID_PAYLOAD, "kilometers_run": 175000}
    low_price = client.post("/predict-harga", json=low_mileage).json()["estimasi_harga"]
    high_price = client.post("/predict-harga", json=high_mileage).json()["estimasi_harga"]
    assert high_price <= low_price

