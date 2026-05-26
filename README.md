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
│   │   ├── data_loader.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── scenarios.py
│   │   ├── experiment_runner.py
│   │   ├── results_aggregator.py
│   │   └── models/
│   └── utils/
│
├── scripts/
│   └── run_deep_learning.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── logs/
│
├── results/
│   ├── figures/
│   ├── logs/
│   └── metrics/
│
├── requirements.txt
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

Projede aşağıdaki PyTorch tabanlı modeller uygulanmıştır:

- LSTM
- GRU
- 1D-CNN

Modeller çok değişkenli (multivariate) scaled veri üzerinde sliding-window ile sequence oluşturarak eğitilir. Derin öğrenme modelleri ile automata modeli final aşamada karşılaştırılacaktır.

Karşılaştırmalarda aşağıdaki metrikler kullanılmaktadır:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC / PR-AUC
- Noise Robustness
- Unseen Pattern Robustness

> **Not:** Kök dizinde `models/lstm.py`, `models/gru.py`, `models/cnn1d.py` dosyaları varsa bunlar **deprecated** olup kullanılmamaktadır. Aktif model implementasyonları `src/deep_learning/models/` altındadır.

---

# Derin Öğrenme Pipeline

PyTorch tabanlı DL pipeline'ı tek komutla tüm model × veri seti × senaryo × seed (× SKAB fold) kombinasyonlarını koşturur ve rapor tablolarını üretir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Komut Örnekleri

Tam koşum (3 model × 2 dataset × 3 senaryo × 5 seed × SKAB 5-fold):

```bash
python -m scripts.run_deep_learning
```

Hızlı smoke test (2 epoch, tek seed, tek SKAB fold):

```bash
python -m scripts.run_deep_learning --smoke
```

Filtreli smoke test:

```bash
python -m scripts.run_deep_learning --smoke --datasets batadal --models lstm --scenarios original
```

Parametreli örnek:

```bash
python -m scripts.run_deep_learning --models lstm gru --datasets skab --scenarios original noise --seeds 42 123
```

270 run gibi uzun grid koşularında matplotlib görsellerini atlayarak hızlandırma (metrikler ve `model.pt` kaydedilir):

```bash
python -m scripts.run_deep_learning --fast
```

Kayıtlı `history.json` ve `model.pt` dosyalarından görselleri **yeniden eğitmeden** üretme:

```bash
# Raporda kullanılacak 11 temsilci run (önerilen liste)
python -m scripts.regenerate_figures --manifest docs/dl_figures_curated_run_ids.txt

# Tek run
python -m scripts.regenerate_figures --run-ids batadal_lstm_original_seed123_foldNA

# Tüm 270 run (1–3 saat, eğitim yok)
python -m scripts.regenerate_figures --all
```

Çıktı: `outputs/figures/<run_id>/` altında `loss.png`, `acc.png`, `cm.png`, `roc.png`, `pr.png`.

## Grid 2: Weighted BCE (imbalance-aware)

İkinci deney grid'i; protokol aynı (batch 32, seed'ler, split'ler), kayıp fonksiyonu train pencere pozitif oranından `pos_weight` alır.

```bash
python -m scripts.run_deep_learning --weighted-bce
```

- Run klasörleri: `outputs/<run_id>_wbce/` (baseline ile karışmaz)
- Metrikler: `outputs/metrics/raw_results_wbce.csv`, `table1_wbce.md`, `table2_wbce.md`
- `metrics.json` içinde `pos_weight` ve `loss_type` alanları

Google Colab'de de aynı komutlar proje kök dizininden çalıştırılabilir.

## Performans (proje isterlerine uygun)

Aşağıdaki optimizasyonlar **batch size=32** ve diğer sabit hiperparametreleri değiştirmeden uygulanır (`config/config.py`):

- SKAB/BATADAL verisi CLI başında bir kez yüklenir (fold başına CSV tekrarı yok)
- `DataLoader`: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor` (CUDA yoksa otomatik kapatılır)
- GPU transfer: `non_blocking=True`
- Mixed precision (`DL_USE_AMP`) CUDA üzerinde eğitim/validation hızlandırması
- `--fast`: run başına 5 PNG üretimini atlar; tablolar için `metrics.json` yeterlidir

## `outputs/` Klasör Yapısı

```text
outputs/
├── <run_id>/                  # Her deney için artifact dizini
│   ├── history.json           # Epoch bazlı train/val loss ve accuracy
│   ├── metrics.json           # Test metrikleri (accuracy, f1, roc_auc, ...)
│   └── model.pt               # En iyi model ağırlıkları
├── figures/
│   └── <run_id>/              # Run bazlı görseller
│       ├── loss.png
│       ├── acc.png
│       ├── cm.png
│       ├── roc.png
│       └── pr.png
├── metrics/
│   ├── raw_results.csv        # Tüm run'ların ham metrikleri
│   ├── table1.md / table1.csv
│   └── table2.md / table2.csv
├── models/                    # Rezerve dizin (create_required_dirs)
└── logs/                      # Rezerve dizin
```

`<run_id>` formatı: `{dataset}_{model}_{scenario}_seed{seed}_fold{fold|NA}`  
Örnek: `batadal_lstm_original_seed42_foldNA`

## Senaryo Tanımları

| Senaryo | Açıklama |
| :--- | :--- |
| **original** | İşlenmiş test verisi doğrudan kullanılır |
| **noise** | Yalnızca test sekanslarına `DL_GAUSSIAN_NOISE_STD` ile Gaussian gürültü eklenir; eğitim aynı kalır |
| **unseen** | Test feature'larına drift uygulanır (`X * DL_UNSEEN_DRIFT_SCALE + DL_UNSEEN_DRIFT_MEAN`); SKAB'da GroupKFold ile source_file leave-out zaten sağlanır |

## Üretilen Tablolar ve Görseller

**Tablolar** (`outputs/metrics/`):

- **Tablo 1:** Model performansı ve stabilitesi — SKAB / BATADAL sütunlarında F1 ± std (original senaryo)
- **Tablo 2:** Gürültü etkisi (Orijinal / Gürültülü F1) ve Unseen senaryo analizi
- **raw_results.csv:** Her `(dataset, model, scenario, seed, fold)` kombinasyonu için ham metrikler

**Görseller** (`outputs/figures/<run_id>/`):

- Loss eğrisi (train + val)
- Accuracy eğrisi (train + val)
- Confusion Matrix
- ROC eğrisi + AUC
- Precision-Recall eğrisi + AUC

## Sabit Hiperparametreler

Tüm hiperparametreler `config/config.py` dosyasından okunur:

- `MAX_EPOCHS=50`, `BATCH_SIZE=32`, `EARLY_STOPPING_PATIENCE=5`
- `RANDOM_SEEDS=[42, 123, 2026, 7, 999]`
- `DL_SEQUENCE_LENGTH=30`, `DL_STRIDE=1`, `DL_LEARNING_RATE=1e-3`
- `DL_NUM_WORKERS=4`, `DL_PIN_MEMORY=True`, `DL_USE_AMP=True` (CPU ortamında otomatik devre dışı)

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
- Deep Learning pipeline (LSTM, GRU, 1D-CNN)
- DL senaryo deneyleri (original / noise / unseen)
- DL otomatik tablo ve görsel üretimi

Devam eden çalışmalar:

- Final model comparison (DL vs Automata)
- Final rapor yazımı

---

# Geliştiriciler

- Derya Gelmez
- Melih İyigören

Kocaeli Üniversitesi  
YazLab-II Projesi