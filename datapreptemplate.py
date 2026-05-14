import pandas as pd
import numpy as np
import re

# =========================
# CONFIG
# =========================
INPUT_FILE = "messy_ecommerce_sales_data.csv"
OUTPUT_FILE = "ecommerce_clean.csv"

# =========================
# HELPER FUNCTIONS
# =========================
def clean_col_name(col):
    """
    Ubah nama kolom jadi format bersih:
    - lowercase
    - spasi / simbol jadi underscore
    - underscore berlebih dihapus
    """
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

def is_missing_like(x):
    """
    Deteksi nilai kosong yang ditulis dalam bentuk teks.
    """
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"", "unknown", "error", "n/a", "na", "nan", "null", "none", "-"}

def parse_amount(x):
    """
    Parsing angka/uang yang berantakan.
    Contoh:
    - '300$' -> 300.0
    - '1,200' -> 1200.0
    - '15.000' -> 15000.0
    - '1,200.50' -> 1200.5
    - '1.200,50' -> 1200.5
    - 'abd' -> NaN
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    # Ambil hanya angka, koma, titik, minus
    s = re.sub(r"[^\d,.\-]", "", s)

    if s in {"", "-", ".", ",", "-.", "-,"}:
        return np.nan

    # Jika ada koma dan titik sekaligus
    if "," in s and "." in s:
        # Format Eropa: 1.200,50
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        # Format US: 1,200.50
        else:
            s = s.replace(",", "")

    # Jika hanya koma
    elif "," in s:
        parts = s.split(",")
        # Kalau bagian belakang 1-2 digit, anggap desimal
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    # Jika hanya titik
    elif "." in s:
        parts = s.split(".")
        # Kalau titik terakhir diikuti 1-2 digit, anggap desimal
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            pass
        else:
            s = s.replace(".", "")

    try:
        return float(s)
    except:
        return np.nan

def fill_mode(series, default="Unknown"):
    """
    Isi missing value dengan modus.
    Kalau modus tidak ada, pakai default.
    """
    mode_vals = series.mode(dropna=True)
    if len(mode_vals) > 0:
        return series.fillna(mode_vals.iloc[0])
    return series.fillna(default)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# =========================
# NORMALIZE COLUMN NAMES
# =========================
df.columns = [clean_col_name(c) for c in df.columns]

# Hapus kolom unnamed atau kolom nama kosong
df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False, regex=True)]
df = df.loc[:, df.columns != ""]

# Hapus baris yang benar-benar kosong
df = df.dropna(how="all")

# =========================
# STANDARDIZE EXPECTED COLUMN NAMES
# =========================
# Kalau ada nama kolom yang mirip, kita seragamkan ke nama utama
rename_map = {
    "customername": "customer_name",
    "customer_name": "customer_name",
    "orderid": "order_id",
    "order_id": "order_id",
    "orderdate": "order_date",
    "order_date": "order_date",
    "paymentmethod": "payment_method",
    "payment_method": "payment_method",
    "dayname": "day_name",
    "day_name": "day_name",
    "isweekend": "is_weekend",
    "is_weekend": "is_weekend",
}
df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

# =========================
# DROP FULLY EMPTY COLUMNS
# =========================
# Ubah string kosong / spasi menjadi NaN dulu
for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", np.nan)

# Hapus kolom yang seluruh isinya kosong
df = df.dropna(axis=1, how="all")

# =========================
# CEK KOLUM YANG DIHARAPKAN
# =========================
expected_cols = [
    "id",
    "customer_name",
    "order_id",
    "order_date",
    "product",
    "category",
    "quantity",
    "price",
    "payment_method",
    "status",
    "total",
]

# Kalau ada kolom yang belum ada, buat kosong supaya kode tidak error
for col in expected_cols:
    if col not in df.columns:
        df[col] = np.nan

# =========================
# CLEAN TEXT COLUMNS
# =========================
text_cols = ["customer_name", "product", "category", "payment_method", "status", "order_id"]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", np.nan)
    df[col] = df[col].replace(["unknown", "error", "n/a", "na", "nan", "null", "none", "-"], np.nan)

# Customer name dan product dirapikan
df["customer_name"] = fill_mode(df["customer_name"], default="Unknown")
df["product"] = fill_mode(df["product"], default="Unknown").astype("string").str.title()

# Category diseragamkan
df["category"] = df["category"].astype("string").str.strip().str.lower()
df["category"] = df["category"].replace({
    "electronic": "Electronics",
    "electronics": "Electronics",
    "home": "Home",
    "sports": "Sports",
    "clothing": "Clothing",
    "books": "Books"
})
df["category"] = fill_mode(df["category"], default="Unknown")

# Payment method diseragamkan
df["payment_method"] = df["payment_method"].astype("string").str.strip().str.lower()
df["payment_method"] = df["payment_method"].replace({
    "cash on delivery": "Cash on Delivery",
    "paypal": "PayPal",
    "bank transfer": "Bank Transfer",
    "credit card": "Credit Card"
})
df["payment_method"] = fill_mode(df["payment_method"], default="Unknown")

# Status diseragamkan
df["status"] = df["status"].astype("string").str.strip().str.lower()
df["status"] = df["status"].replace({
    "shipped": "Shipped",
    "processing": "Processing",
    "delivered": "Delivered",
    "cancelled": "Cancelled",
    "returned": "Returned"
})
df["status"] = fill_mode(df["status"], default="Unknown")

# =========================
# PARSE NUMERIC COLUMNS
# =========================
df["quantity"] = df["quantity"].apply(parse_amount)
df["price"] = df["price"].apply(parse_amount)
df["total"] = df["total"].apply(parse_amount)

# Angka tidak valid dianggap missing
df.loc[df["quantity"] <= 0, "quantity"] = np.nan
df.loc[df["price"] <= 0, "price"] = np.nan
df.loc[df["total"] <= 0, "total"] = np.nan

# =========================
# PARSE DATE
# =========================
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", infer_datetime_format=True)

# Baris dengan tanggal invalid dibuang
df = df.dropna(subset=["order_date"])

# =========================
# HANDLE MISSING NUMERIC VALUES
# =========================
# Quantity -> median lalu integer nullable
if df["quantity"].notna().any():
    df["quantity"] = df["quantity"].fillna(df["quantity"].median())
df["quantity"] = df["quantity"].round().astype("Int64")

# Price -> median
if df["price"].notna().any():
    df["price"] = df["price"].fillna(df["price"].median())
df["price"] = df["price"].round(2)

# Total -> nanti dihitung ulang, jadi fill sementara tidak perlu terlalu penting
if df["total"].notna().any():
    df["total"] = df["total"].fillna(df["total"].median())
df["total"] = df["total"].round(2)

# =========================
# HAPUS DUPLIKAT
# =========================
# Jika order_id ada dan tidak kosong, pakai itu sebagai acuan duplikat
if df["order_id"].notna().any():
    df = df.drop_duplicates(subset=["order_id"], keep="first")
else:
    df = df.drop_duplicates()

# =========================
# VALIDASI TRANSAKSI
# =========================
# Hitung total bersih dari price * quantity
df["calculated_total"] = (df["price"] * df["quantity"]).round(2)

# Selisih antara total asli dan total hitung ulang
df["diff"] = (df["total"] - df["calculated_total"]).round(2)

# Flag mismatch jika selisihnya lebih dari 1
df["mismatch_flag"] = (df["diff"].abs() > 1).astype(int)

# Untuk data final, pakai total hasil hitung ulang agar konsisten
df["total"] = df["calculated_total"]

# =========================
# FEATURE ENGINEERING
# =========================
df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day"] = df["order_date"].dt.day
df["quarter"] = df["order_date"].dt.quarter
df["day_name"] = df["order_date"].dt.day_name()
df["is_weekend"] = df["day_name"].isin(["Saturday", "Sunday"]).astype(int)
df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

df["transaction_value"] = (df["price"] * df["quantity"]).round(2)
df["revenue_per_unit"] = df["price"]
df["product_name_length"] = df["product"].astype(str).str.len()

df["price_level"] = pd.cut(
    df["price"],
    bins=[-np.inf, 100, 300, 600, np.inf],
    labels=["Low", "Medium", "High", "Premium"]
)

df["quantity_bucket"] = pd.cut(
    df["quantity"].astype(float),
    bins=[0, 1, 3, 5, np.inf],
    labels=["1", "2-3", "4-5", "6+"],
    include_lowest=True
)

# High value order berdasarkan kuartil 75%
threshold = df["total"].quantile(0.75)
df["high_value_order"] = (df["total"] >= threshold).astype(int)

# =========================
# FINAL CLEAN-UP: HAPUS KOLOM Aneh/Kosong
# =========================
# Hapus kolom yang seluruh isinya kosong
df = df.dropna(axis=1, how="all")

# Pastikan tidak ada kolom nama aneh
df.columns = [clean_col_name(c) for c in df.columns]
df = df.loc[:, ~df.columns.str.contains(r"^unnamed", case=False, regex=True)]
df = df.loc[:, df.columns != ""]

# Buang kolom yang tidak dipakai jika kamu ingin output lebih rapi
# Tapi tetap simpan kolom penting untuk audit
keep_cols = [
    "id",
    "customer_name",
    "order_id",
    "order_date",
    "product",
    "category",
    "quantity",
    "price",
    "payment_method",
    "status",
    "total",
    "calculated_total",
    "diff",
    "mismatch_flag",
    "year",
    "month",
    "day",
    "quarter",
    "day_name",
    "is_weekend",
    "year_month",
    "transaction_value",
    "revenue_per_unit",
    "product_name_length",
    "price_level",
    "quantity_bucket",
    "high_value_order"
]

# Simpan hanya kolom yang benar-benar ada
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols]

# =========================
# FINAL MISSING VALUE CHECK
# =========================
# Isi sisa missing di kolom teks dengan Unknown
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype("string").fillna("Unknown")
    df[col] = df[col].replace("", "Unknown")

# Isi sisa missing di numerik dengan median
for col in df.select_dtypes(include=["number"]).columns:
    if df[col].isna().any():
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(0)

# =========================
# VALIDASI AKHIR
# =========================
print("Sisa missing value per kolom:")
print(df.isna().sum())

print("\nNama kolom final:")
print(df.columns.tolist())

print("\nShape final:")
print(df.shape)

# Pastikan tidak ada kolom unnamed / kosong
assert not any(c.startswith("unnamed") or c.strip() == "" for c in df.columns), "Masih ada kolom aneh!"

# =========================
# SAVE OUTPUT
# =========================
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nData bersih berhasil disimpan ke: {OUTPUT_FILE}")