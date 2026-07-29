# Artefak model

Folder ini sengaja tidak menyimpan `model.joblib` di Git. Buat artefaknya dengan:

```bash
python -m src.load_data
python -m src.train
python -m src.evaluate
```

Perintah tersebut menghasilkan pipeline lengkap `model.joblib` dan `metadata.json` yang dibutuhkan API.

