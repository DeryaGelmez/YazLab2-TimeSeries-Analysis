import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# 1. Dosya yolunu tanımla
# =========================

base_path = Path(r"D:\Users\51001564\OneDrive - ARÇELİK A.Ş\Desktop\Dataset\SKAB")
file_path = base_path / "skab_combined.csv"


# =========================
# 2. Veriyi oku
# =========================

skab_df = pd.read_csv(
    file_path,
    sep=";",
    encoding="cp1254"
)

print("SKAB verisi okundu.")
print("Toplam satır:", skab_df.shape[0])
print("Toplam sütun:", skab_df.shape[1])


# =========================
# 3. Feature / target / group ayrımı
# =========================

target_column = "anomaly"
group_column = "source_file"

excluded_columns = [
    "datetime",
    "changepoint",
    "source_group",
    "source_file",
    target_column
]

feature_columns = [
    col for col in skab_df.columns
    if col not in excluded_columns
]

X = skab_df[feature_columns]
y = skab_df[target_column]
groups = skab_df[group_column]


# =========================
# 4. Sayısal dönüşüm
# =========================

# Sayısala çevrilemeyen değerleri NaN yap
X = X.apply(pd.to_numeric, errors="coerce")

print("\nFeature sütunlarında sayısala çevrilemeyen/eksik değer sayıları:")
print(X.isnull().sum())

# Eksik/bozuk değerleri sütun ortalaması ile doldur
X = X.fillna(X.mean())

print("\nSayısal dönüşüm ve eksik değer doldurma tamamlandı.")

print("\nFeature veri tipleri:")
print(X.dtypes)


# =========================
# 5. Genel bilgi çıktıları
# =========================

print("\nFeature sütunları:")
print(feature_columns)

print("\nFeature sayısı:", len(feature_columns))
print("Target sütunu:", target_column)
print("Group sütunu:", group_column)

print("\nTarget dağılımı:")
print(y.value_counts())

print("\nGroup sayısı:")
print(groups.nunique())


# =========================
# 6. GroupKFold tanımla
# =========================

n_splits = 5

gkf = GroupKFold(n_splits=n_splits)

print(f"\nGroupKFold {n_splits} fold ile oluşturuldu.")


# =========================
# 7. İlk fold preprocessing
# =========================

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

    print(f"\n========== Fold {fold + 1} ==========")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    print("Train satır sayısı:", X_train.shape[0])
    print("Test satır sayısı:", X_test.shape[0])

    print("Train dosya sayısı:", train_groups.nunique())
    print("Test dosya sayısı:", test_groups.nunique())

    # Leakage kontrolü
    common_files = set(train_groups.unique()).intersection(
        set(test_groups.unique())
    )

    print("Train-test ortak dosya sayısı:", len(common_files))


    # =========================
    # 8. Normalization
    # =========================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nNormalization tamamlandı.")

    print("X_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)


    # =========================
    # 9. PCA - Automata için PC1
    # =========================

    pca = PCA(n_components=1)

    X_train_pc1 = pca.fit_transform(X_train_scaled)
    X_test_pc1 = pca.transform(X_test_scaled)

    print("\nPCA tamamlandı.")

    print("X_train_pc1 shape:", X_train_pc1.shape)
    print("X_test_pc1 shape:", X_test_pc1.shape)

    print(
        "PC1 explained variance ratio:",
        pca.explained_variance_ratio_[0]
    )

    # Şimdilik sadece ilk fold'u test ediyoruz
    break