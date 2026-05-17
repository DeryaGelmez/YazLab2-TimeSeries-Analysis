import pandas as pd
from pathlib import Path

# BATADAL klasör yolu
base_path = Path(r"D:\Users\51001564\OneDrive - ARÇELİK A.Ş\Desktop\Dataset\BATADAL")

# Dosya yolu
file_path = base_path / "BATADAL_dataset04.csv"

# CSV dosyasını oku
batadal_df = pd.read_csv(file_path)

# Sütun isimlerindeki baştaki/sondaki boşlukları temizle
batadal_df.columns = batadal_df.columns.str.strip()

print("BATADAL dosyası okundu.")
print("Toplam satır sayısı:", batadal_df.shape[0])
print("Toplam sütun sayısı:", batadal_df.shape[1])

# İlk 5 satırı göster
print("\nİlk 5 satır:")
print(batadal_df.head())

# Sütun isimlerini göster
print("\nSütun isimleri:")
print(batadal_df.columns.tolist())

# Eksik veri kontrolü
print("\nEksik veri sayıları:")
print(batadal_df.isnull().sum())

# ATT_FLAG dağılımı
print("\nATT_FLAG dağılımı:")
print(batadal_df["ATT_FLAG"].value_counts())

# Yeni anomaly sütunu oluştur
# -999 = normal (0)
# 1 = anomaly (1)
batadal_df["anomaly"] = batadal_df["ATT_FLAG"].replace({
    -999: 0,
    1: 1
})

# Yeni anomaly dağılımını kontrol et
print("\nYeni anomaly dağılımı:")
print(batadal_df["anomaly"].value_counts())

# Hazırlanmış veriyi kaydet
output_path = base_path / "batadal_prepared.csv"

batadal_df.to_csv(output_path, index=False)

print(f"\nBATADAL hazırlanmış veri kaydedildi: {output_path}")