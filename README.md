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
- Veri sızıntısı önleme
- Train/Test ayrımı

## SKAB Veri Seti

- GroupKFold mantığına uygun dosya bazlı ayrım uygulanmıştır
- source_file sütunu grup değişkeni olarak kullanılmıştır

## BATADAL Veri Seti

- Zaman sırası korunmuştur
- Veri %60 Train, %20 Validation ve %20 Test olarak ayrılmıştır

---

# Olasılıksal Otomata Pipeline

Projede geliştirilen automata tabanlı model aşağıdaki adımlardan oluşmaktadır:

1. PAA
2. SAX
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
- Parameter Sensitivity Graphs
- F1-score vs Alphabet Size
- Recall vs Alphabet Size
- State Count vs Alphabet Size
- Transition Density vs Alphabet Size

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

Analiz sonuçları göstermiştir ki:

- alphabet size arttıkça state sayısı artmaktadır
- transition density azalmaktadır
- model daha agresif anomaly detection davranışı göstermektedir
- recall artarken false positive oranı da artmaktadır

Bu durum anomaly detection problemlerindeki klasik trade-off yapısını göstermektedir.

## SKAB En İyi Sonuç

```text
window_size = 6
alphabet_size = 6

F1-score = 0.394
Recall = 0.759
```

## BATADAL En İyi Sonuç

```text
window_size = 4
alphabet_size = 6

F1-score = 0.090
Recall = 0.714
```

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

Derin öğrenme modelleri ile automata modeli final aşamada karşılaştırılacaktır.

Karşılaştırmalarda aşağıdaki metrikler kullanılacaktır:

- Accuracy
- Precision
- Recall
- F1-score
- Explainability
- Noise Robustness
- Unseen Pattern Robustness

## Derin Öğrenme Pipeline

Bağımlılıkları kurduktan sonra proje kök dizininden aşağıdaki komutlarla deneyleri çalıştırabilirsiniz:

```bash
pip install -r requirements.txt
python -m scripts.run_deep_learning
python -m scripts.run_deep_learning --smoke
```

Örnek filtreli koşum:

```bash
python -m scripts.run_deep_learning --smoke --datasets batadal --models lstm --scenarios original
```

---

# Deneysel Senaryolar

Deneyler aşağıdaki senaryolar altında gerçekleştirilmiştir:

- Orijinal veri
- Gaussian Noise eklenmiş veri
- Controlled Unseen Pattern verisi

## Gaussian Noise Deneyi

Modelin gürültülü veriler altındaki dayanıklılığı ölçülmüştür.

Bu deneyde test verisine düşük seviyeli Gaussian Noise eklenmiş ve model performansı tekrar değerlendirilmiştir.

Sonuçlar modelin düşük seviyeli gürültü altında görece stabil kaldığını göstermiştir.

## Controlled Unseen Experiment

Modelin eğitim sırasında görülmeyen pattern’lar karşısındaki davranışı analiz edilmiştir.

Bu deneyde kontrollü şekilde unseen pattern’lar oluşturulmuş ve Levenshtein Distance algoritması ile en yakın pattern eşleştirmesi gerçekleştirilmiştir.

Örnek mapping:

```text
ccdcbb → ccdcbe
distance = 1
```

Bu yapı sayesinde model unseen pattern’lar altında çalışmaya devam edebilmiştir.

---

# Mevcut Durum

Tamamlanan çalışmalar:

- SKAB preprocessing
- BATADAL preprocessing
- Probabilistic Automata pipeline
- Explainability sistemi
- Transition analysis
- Heatmap üretimi
- State diagram üretimi
- Parameter sensitivity analysis
- Gaussian noise experiments
- Controlled unseen experiments
- Summary metric tabloları

Devam eden çalışmalar:

- Deep Learning modelleri
- LSTM
- GRU
- 1D-CNN
- Final model comparison
- Final rapor yazımı

---

# Geliştiriciler

- Derya Gelmez
- Melih İyigören

Kocaeli Üniversitesi  
YazLab-II Projesi