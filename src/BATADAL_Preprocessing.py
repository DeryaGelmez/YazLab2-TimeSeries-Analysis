import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# 1. Dosya yolunu tanımla
# =========================

base_path = Path(r"D:\Users\51001564\OneDrive - ARÇELİK A.Ş\Desktop\Dataset\BATADAL")
file_path = base_path / "batadal_prepared.csv"


# =========================
# 2. Veriyi oku
# =========================

batadal_df = pd.read_csv(file_path)

batadal_df.columns = batadal_df.columns.str.strip()

print("BATADAL verisi okundu.")
print("Toplam satır:", batadal_df.shape[0])
print("Toplam sütun:", batadal_df.shape[1])


# =========================
# 3. Feature / target ayrımı
# =========================

target_column = "anomaly"

excluded_columns = [
    "DATETIME",
    "ATT_FLAG",
    target_column
]

feature_columns = [
    col for col in batadal_df.columns
    if col not in excluded_columns
]

X = batadal_df[feature_columns]
y = batadal_df[target_column]
datetime_col = batadal_df["DATETIME"]


# =========================
# 4. Sayısal dönüşüm
# =========================

X = X.apply(pd.to_numeric, errors="coerce")

print("\nFeature sütunlarında sayısala çevrilemeyen/eksik değer sayıları:")
print(X.isnull().sum())

X = X.fillna(X.mean())

print("\nSayısal dönüşüm ve eksik değer doldurma tamamlandı.")

print("\nFeature sayısı:", len(feature_columns))
print("Target sütunu:", target_column)

print("\nTarget dağılımı:")
print(y.value_counts())


# =========================
# 5. Zaman sıralı split
# %60 train, %20 validation, %20 test
# =========================

total_len = len(batadal_df)

train_end = int(total_len * 0.60)
val_end = int(total_len * 0.80)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
datetime_train = datetime_col.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
datetime_val = datetime_col.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
datetime_test = datetime_col.iloc[val_end:]

print("\nZaman sıralı split tamamlandı.")

print("Train satır sayısı:", X_train.shape[0])
print("Validation satır sayısı:", X_val.shape[0])
print("Test satır sayısı:", X_test.shape[0])

print("\nTrain target dağılımı:")
print(y_train.value_counts())

print("\nValidation target dağılımı:")
print(y_val.value_counts())

print("\nTest target dağılımı:")
print(y_test.value_counts())

print("\nTrain tarih aralığı:")
print(datetime_train.iloc[0], "→", datetime_train.iloc[-1])

print("\nValidation tarih aralığı:")
print(datetime_val.iloc[0], "→", datetime_val.iloc[-1])

print("\nTest tarih aralığı:")
print(datetime_test.iloc[0], "→", datetime_test.iloc[-1])


# =========================
# 6. Normalization
# Sadece train üzerinde fit edilir
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nNormalization tamamlandı.")

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_val_scaled shape:", X_val_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)


# =========================
# 7. PCA - Automata için PC1
# Sadece train üzerinde fit edilir
# =========================

pca = PCA(n_components=1)

X_train_pc1 = pca.fit_transform(X_train_scaled)
X_val_pc1 = pca.transform(X_val_scaled)
X_test_pc1 = pca.transform(X_test_scaled)

print("\nPCA tamamlandı.")

print("X_train_pc1 shape:", X_train_pc1.shape)
print("X_val_pc1 shape:", X_val_pc1.shape)
print("X_test_pc1 shape:", X_test_pc1.shape)

print("PC1 explained variance ratio:", pca.explained_variance_ratio_[0])