import pandas as pd
from pathlib import Path

base_path = Path("data/raw/BATADAL")

file_path = base_path / "BATADAL_dataset04.csv"

batadal_df = pd.read_csv(file_path)

batadal_df.columns = batadal_df.columns.str.strip()

print("BATADAL dosyası okundu.")
print("Toplam satır sayısı:", batadal_df.shape[0])
print("Toplam sütun sayısı:", batadal_df.shape[1])

print("\nİlk 5 satır:")
print(batadal_df.head())

print("\nSütun isimleri:")
print(batadal_df.columns.tolist())

print("\nEksik veri sayıları:")
print(batadal_df.isnull().sum())

print("\nATT_FLAG dağılımı:")
print(batadal_df["ATT_FLAG"].value_counts())

batadal_df["anomaly"] = batadal_df["ATT_FLAG"].replace({
    -999: 0,
    1: 1
})

print("\nYeni anomaly dağılımı:")
print(batadal_df["anomaly"].value_counts())

output_path = base_path / "batadal_prepared.csv"

batadal_df.to_csv(output_path, index=False)

print(f"\nBATADAL hazırlanmış veri kaydedildi: {output_path}")