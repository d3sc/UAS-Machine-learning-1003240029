# Ringkasan EDA

## 1. Distribusi Harga
Distribusi harga kendaraan bersifat right-skewed. Nilai persentil ke-99 sekitar 50,715. Oleh karena itu transformasi log1p layak dipertimbangkan saat proses training.

## 2. Missing Value
Dataset memiliki sedikit atau bahkan tidak memiliki missing value. Jika terdapat missing value pada tahap preprocessing, penanganannya dapat menggunakan SimpleImputer.

## 3. Hubungan Umur Kendaraan dan Harga
Semakin tua umur kendaraan, secara umum harga semakin rendah. Hubungan ini tidak sepenuhnya linear sehingga model berbasis pohon kemungkinan memiliki performa lebih baik.

## 4. Fuel Type
Harga kendaraan bervariasi menurut jenis bahan bakar. Boxplot menunjukkan adanya perbedaan distribusi harga antar fuel type.

## 5. Korelasi
Heatmap Spearman digunakan untuk melihat hubungan antar fitur numerik seperti year, mileage, engine_size, tax, mpg, vehicle_age, dan price.

## Prakiraan Sebelum Training

- Vehicle age diperkirakan memiliki korelasi negatif terhadap price.
- Random Forest diperkirakan memberikan performa lebih baik dibanding Linear Regression karena mampu menangkap hubungan non-linear.
- Transformasi log terhadap target dapat membantu mengurangi pengaruh kendaraan dengan harga yang sangat tinggi.
