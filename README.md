# E-Commerce Sales Data Cleaning Project

## Overview
Project ini bertujuan untuk membersihkan dan mempersiapkan dataset penjualan e-commerce yang memiliki data tidak konsisten, missing value, duplicate data, format angka dan tanggal yang tidak rapi, serta outlier sebelum digunakan untuk analisis data atau machine learning.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Data Cleaning Process
Proses data cleaning yang dilakukan meliputi:
- Normalisasi nama kolom
- Pembersihan data kategorikal
- Konversi data numerik dan tanggal
- Penanganan missing value
- Deteksi dan penanganan outlier menggunakan metode IQR
- Validasi transaksi dengan menghitung ulang nilai total berdasarkan `price` dan `quantity`
- Penghapusan duplicate data

## Feature Engineering
Feature engineering yang ditambahkan pada dataset:
- Year
- Month
- Day
- Quarter
- Day Name
- Is Weekend
- Year Month
- Transaction Value
- Revenue Per Unit
- Price Level
- Quantity Bucket
- High Value Order Flag

## Output
Setelah seluruh proses selesai, dataset akhir disimpan dalam file:

```bash
ecommerce_clean.csv
```

Dataset hasil akhir sudah bersih, konsisten, dan siap digunakan untuk:
- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning
```
