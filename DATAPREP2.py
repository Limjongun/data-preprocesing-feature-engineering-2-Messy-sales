import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("messy_ecommerce_sales_data.csv")

# Rapikan nama kolom agar tidak ada spasi depan/belakang
df.columns = df.columns.str.strip()

# =========================
# 2. CEK DATA AWAL
# =========================
print(df.head())
print(df.tail())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe(include="all"))

# =========================
# 3. GANTI TOKEN MISSING MENJADI NaN
# =========================
# Dataset ini punya beberapa token teks yang mewakili data hilang
missing_tokens = ['UNKNOWN', 'ERROR', 'N/A', '', 'nan', 'NaN', 'NULL', 'null']

# Untuk semua kolom bertipe object/string, rapikan spasi dulu
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()

# Ganti token missing jadi NaN
df = df.replace(missing_tokens, np.nan)

print("\nMissing value setelah replace token:")
print(df.isna().sum())

# =========================
# 4. HAPUS DUPLIKAT
# =========================
print("\nJumlah duplikat sebelum hapus:", df.duplicated().sum())
df = df.drop_duplicates()
print("Jumlah duplikat setelah hapus:", df.duplicated().sum())

# =========================
# 5. BERSIHKAN KOLOM TEKS
# =========================
# Customer_Name
df["Customer_Name"] = df["Customer_Name"].astype("string").str.strip()

# Product
df["Product"] = df["Product"].astype("string").str.strip().str.title()

# Category: samakan variasi penulisan
df["Category"] = df["Category"].astype("string").str.strip().str.lower()

category_map = {
    "electronic": "Electronics",
    "electronics": "Electronics",
    "home": "Home",
    "sports": "Sports",
    "clothing": "Clothing",
    "books": "Books"
}
df["Category"] = df["Category"].replace(category_map)

# Payment_Method: diseragamkan
df["Payment_Method"] = df["Payment_Method"].astype("string").str.strip().str.lower()

payment_map = {
    "cash on delivery": "Cash on Delivery",
    "paypal": "PayPal",
    "bank transfer": "Bank Transfer",
    "credit card": "Credit Card"
}
df["Payment_Method"] = df["Payment_Method"].replace(payment_map)

# Status: diseragamkan
df["Status"] = df["Status"].astype("string").str.strip().str.lower()

status_map = {
    "shipped": "Shipped",
    "processing": "Processing",
    "delivered": "Delivered",
    "cancelled": "Cancelled",
    "returned": "Returned"
}
df["Status"] = df["Status"].replace(status_map)

# =========================
# 6. KONVERSI KOLOM NUMERIK
# =========================
def extract_number(x):
    """
    Ambil angka dari isi sel yang tidak rapi.
    Contoh:
    - '300$' -> 300
    - '4a' -> 4
    - 'abd' -> NaN
    - 'four hundred' -> NaN
    """
    if pd.isna(x):
        return np.nan
    text = str(x).strip().replace(",", "")
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return np.nan

num_col = ["Price", "Quantity", "Total"]

# Quantity dan Price dibersihkan dari teks aneh
df["Quantity"] = df["Quantity"].apply(extract_number)
df["Price"] = df["Price"].apply(extract_number)

# Total juga boleh tetap dibaca sebagai numerik
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

print("\nTipe data setelah konversi numerik:")
print(df.dtypes)

# =========================
# 7. KONVERSI TANGGAL
# =========================
# Dataset ini punya format campuran, misalnya:
# 11/22/2024 dan Jan 5 2023
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce", infer_datetime_format=True)

print("\nTipe data setelah konversi tanggal:")
print(df.dtypes)

# =========================
# 8. ISI MISSING VALUE
# =========================
# Quantity: isi dengan median, lalu bulatkan ke integer
if df["Quantity"].notna().any():
    df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Quantity"] = df["Quantity"].round().astype("Int64")

# Price: isi dengan median
if df["Price"].notna().any():
    df["Price"] = df["Price"].fillna(df["Price"].median())

# Category: isi dengan modus, kalau tidak ada isi "Unknown"
if df["Category"].notna().any():
    category_mode = df["Category"].mode(dropna=True)
    if len(category_mode) > 0:
        df["Category"] = df["Category"].fillna(category_mode.iloc[0])
    else:
        df["Category"] = df["Category"].fillna("Unknown")
else:
    df["Category"] = df["Category"].fillna("Unknown")

# Payment_Method dan Status juga bisa diisi modus bila ada missing
for col in ["Payment_Method", "Status", "Product", "Customer_Name"]:
    if df[col].notna().any():
        mode_val = df[col].mode(dropna=True)
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val.iloc[0])

# Order_Date yang gagal parsing jadi NaT akan dibuang
df = df.dropna(subset=["Order_Date"])

# =========================
# 9. VALIDASI TRANSAKSI
# =========================
# Karena Total di dataset kadang kosong/tidak konsisten,
# lebih aman dihitung ulang dari Quantity * Price
df["Calculated_Total"] = (df["Price"] * df["Quantity"]).round(2)

# Simpan selisih antara Total asli dan hasil hitung ulang
df["Diff"] = df["Total"] - df["Calculated_Total"]

print("\nRingkasan selisih Total:")
print(df["Diff"].describe())

# Ganti Total dengan hasil perhitungan ulang agar konsisten
df["Total"] = df["Calculated_Total"]

# =========================
# 10. OUTLIER DETECTION DAN CLIPPING
# =========================
print("\nOutlier detect: Total")

Q1 = df["Total"].quantile(0.25)
Q3 = df["Total"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier = df[(df["Total"] < lower) | (df["Total"] > upper)]
print("Jumlah outlier:", outlier.shape[0])

# Clip ekstrem agar tidak terlalu mengganggu analisis
lower_cap = df["Total"].quantile(0.01)
upper_cap = df["Total"].quantile(0.99)
df["Total"] = df["Total"].clip(lower=lower_cap, upper=upper_cap)

# =========================
# 11. FEATURE ENGINEERING WAKTU
# =========================
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Day"] = df["Order_Date"].dt.day
df["Day_Name"] = df["Order_Date"].dt.day_name()
df["is_weekend"] = df["Day_Name"].isin(["Saturday", "Sunday"]).astype(int)

# Lebih aman pakai Year-Month supaya tidak tercampur antar tahun
df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# =========================
# 12. ANALISIS SEDERHANA
# =========================
monthly_sales = df.groupby("Year_Month")["Total"].sum().sort_index()

plt.figure(figsize=(10, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Year-Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df["Payment_Method"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Payment Method")
plt.ylabel("")
plt.tight_layout()
plt.show()

# =========================
# 13. SIMPAN HASIL
# =========================
df.to_csv("ecommerce_clean.csv", index=False)

print("\nData sudah dibersihkan dan disimpan ke: ecommerce_clean.csv")
print("\nFinal shape:", df.shape)
print("\nMissing values terakhir:")
print(df.isna().sum())