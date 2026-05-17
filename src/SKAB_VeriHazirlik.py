import pandas as pd
from pathlib import Path

base_path = Path(r"D:\Users\51001564\OneDrive - ARÇELİK A.Ş\Desktop\Dataset\SKAB")

groups = ["valve1", "valve2"]

all_dfs = []

for group in groups:
    folder_path = base_path / group
    
    csv_files = list(folder_path.glob("*.csv"))
    print(f"{group} klasöründe {len(csv_files)} csv bulundu.")
    
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=";")
        
        df["source_group"] = group
        
        df["source_file"] = f"{group}_{file_path.name}"
        
        all_dfs.append(df)

skab_df = pd.concat(all_dfs, ignore_index=True)

print("Birleştirme tamamlandı.")
print("Toplam satır sayısı:", skab_df.shape[0])
print("Toplam sütun sayısı:", skab_df.shape[1])

print(skab_df.head())

print(skab_df.columns.tolist())

print("Anomaly değer dağılımı:")
print(skab_df["anomaly"].value_counts())

print("\nSource group dağılımı:")
print(skab_df["source_group"].value_counts())

print("\nDosya bazlı satır sayıları:")
print(skab_df["source_file"].value_counts().sort_index())

print("\nEksik veri sayıları:")
print(skab_df.isnull().sum())

output_path = base_path / "skab_combined.csv"

skab_df.to_csv(output_path, index=False, sep=";")

print(f"SKAB birleştirilmiş veri kaydedildi: {output_path}")