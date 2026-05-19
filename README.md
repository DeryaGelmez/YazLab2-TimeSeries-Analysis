# YazLab2 - Zaman Serisi Anomali Tespiti

Bu proje, YazLab-II dersi kapsamında geliştirilmiştir. Projede çok değişkenli zaman serisi verileri üzerinde:

- Derin Öğrenme Modelleri
- Olasılıksal Otomata Tabanlı Modeller

kullanılarak anomali tespiti gerçekleştirilmektedir.

---

# Kullanılan Veri Setleri

## SKAB
- Endüstriyel sensör verileri içeren anomali tespit veri seti
- source_file bazlı Group Split uygulanmıştır
- Veri sızıntısını önlemek amacıyla aynı dosya hem train hem test kümesinde kullanılmamıştır

## BATADAL
- Su dağıtım sistemi saldırı tespit veri seti
- Zaman sıralı train/validation/test ayrımı uygulanmıştır

---

# Proje Klasör Yapısı

```text
YazLab2/
│
├── config/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── src/
│   ├── preprocessing/
│   ├── automata/
│   ├── deep_learning/
│   └── utils/
│
├── results/
│   ├── figures/
│   ├── logs/
│   └── metrics/
│
├── reports/
│   └── Process.docx
│
└── README.md
```

---

# Veri Ön İşleme

Veri ön işleme aşamasında aşağıdaki işlemler uygulanmıştır:

- Eksik veri kontrolleri
- Normalizasyon
- PCA boyut indirgeme
- PC1 çıkarımı
- Veri sızıntısı önleme (Data Leakage Prevention)
- Train/Test ayrımı

## SKAB Veri Seti

- GroupKFold mantığına uygun dosya bazlı ayrım uygulanmıştır
- source_file sütunu grup değişkeni olarak kullanılmıştır

## BATADAL Veri Seti

- Zaman sırası korunmuştur
- Veri:
  - %60 Train
  - %20 Validation
  - %20 Test
  olarak ayrılmıştır

---

# Olasılıksal Otomata Pipeline

Projede geliştirilen automata tabanlı model aşağıdaki adımlardan oluşmaktadır:

1. PAA (Piecewise Aggregate Approximation)
2. SAX (Symbolic Aggregate approXimation)
3. Sliding Window ile pattern çıkarımı
4. State oluşturma
5. Transition Probability hesaplama
6. Path Probability analizi
7. Levenshtein Distance ile unseen pattern yönetimi
8. Açıklanabilirlik modülü

---

# Açıklanabilirlik Modülü

Model her karar için aşağıdaki bilgileri üretmektedir:

- mevcut state
- pattern bilgisi
- seen/unseen durumu
- eşlenen pattern
- geçiş olasılığı
- path probability
- anomaly kararı

Örnek çıktı:

```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```

---

# Oluşturulan Görseller

Proje kapsamında otomatik olarak aşağıdaki görseller oluşturulmaktadır:

- Confusion Matrix
- Transition Probability Heatmap
- Automata State Diagram

Çıktılar aşağıdaki klasörlerde saklanmaktadır:

```text
results/figures/
results/logs/
results/metrics/
```

---

# Parametre Analizi

Aşağıdaki parametre kombinasyonları test edilmiştir:

- window_size = [3, 4, 5, 6]
- alphabet_size = [3, 4, 5, 6]

Analizlerde aşağıdaki değerler incelenmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- state sayısı
- transition density

---

# Kullanılan Değerlendirme Metrikleri

Projede aşağıdaki değerlendirme metrikleri kullanılmaktadır:

- Accuracy
- Precision
- Recall
- F1-Score

Ek olarak:

- state count
- transition density
- path probability

analizleri gerçekleştirilmektedir.

---

# Derin Öğrenme Modelleri

Projede aşağıdaki modeller uygulanacaktır:

- 1D-CNN
- LSTM
- GRU

Model eğitimleri Google Colab ortamında GPU kullanılarak gerçekleştirilecektir.

---

# Deneysel Senaryolar

Deneyler aşağıdaki senaryolar altında gerçekleştirilecektir:

- Orijinal veri
- Gaussian Noise eklenmiş veri
- Unseen pattern verisi

---

# Geliştiriciler

- Derya Gelmez
- Melih İyigören

Kocaeli Üniversitesi  
YazLab-II Projesi