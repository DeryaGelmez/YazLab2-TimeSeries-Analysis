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

- Su dağıtım sistemi saldırı tespit veri seti (Training Dataset 2 kullanılmıştır)
- **Etiket sütunu:** `ATT_FLAG` → `anomaly` olarak yeniden adlandırılmıştır (0 = normal, 1 = saldırı)
- Zaman bilgisi içeren `DATETIME` sütunu model girdisine dahil edilmemiş, yalnızca sıralama için kullanılmıştır
- Zaman sıralı bölünme: %60 train / %20 validation / %20 test

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

PDF örneğiyle uyumlu çıktı (unseen senaryosu):

```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "nearest_distance": 1,
  "transitions": [
    {"transition": "aab -> abc", "probability": 0.720000},
    {"transition": "abc -> bcc", "probability": 0.150000}
  ],
  "probability": 0.108,
  "confidence_score": 0.108,
  "decision": "anomaly"
}
```

Gerçek sistem çıktısından örnek (seen senaryosu, anomali kararı):

```json
{
  "time_step": 12,
  "state": "baaa",
  "pattern": "aaaa",
  "status": "seen",
  "mapped_to": null,
  "nearest_distance": null,
  "transitions": [
    {"transition": "baaa -> aaaa", "probability": 0.708029},
    {"transition": "aaaa -> aaab", "probability": 0.130560}
  ],
  "probability": 0.092440,
  "confidence_score": 0.092440,
  "decision": "anomaly"
}
```

**Karar gerekçesi:** P(baaa→aaaa) × P(aaaa→aaab) = 0.708 × 0.131 = **0.092 < 0.10 (eşik)** → ANOMALİ

---

# Oluşturulan Görseller

Proje kapsamında otomatik olarak aşağıdaki görseller oluşturulmaktadır:

**Otomata görselleri** (`results/figures/`):

| Görsel | Dosya |
| :--- | :--- |
| Confusion Matrix | `skab_automata_confusion_matrix.png` |
| ROC Eğrisi (AUC=0.40) | `skab_automata_roc_curve.png` |
| Precision-Recall Eğrisi | `skab_automata_pr_curve.png` |
| Transition Probability Heatmap | `skab_automata_transition_heatmap.png` |
| Automata State Diagram | `skab_automata_state_diagram.png` |
| F1 vs Alphabet (sabit fold) | `skab_f1_score_vs_alphabet.png` |
| F1 vs Alphabet (GroupKFold, mean±std) | `skab_f1_score_vs_alphabet_gkf.png` |
| Recall vs Alphabet | `skab_recall_vs_alphabet.png` |
| State Count vs Alphabet | `skab_state_count_vs_alphabet.png` |
| Transition Density vs Alphabet | `skab_transition_density_vs_alphabet.png` |

BATADAL için de eşdeğer görseller (`batadal_*` önekiyle) üretilmektedir.

**Derin öğrenme görselleri** (`outputs/figures/<run_id>/`): loss, accuracy, confusion matrix, ROC, PR eğrileri.

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

## SKAB En İyi Parametre (GroupKFold, 5 fold)

```text
window_size   = 5
alphabet_size = 6

F1-score = 0.494 ± 0.069   (fold ortalaması ± std)
State sayısı ≈ 1420
Geçiş yoğunluğu = 0.00112
```

> Değerlendirme: 5-fold GroupKFold (source_file bazlı). Path probability = P(prev→cur) × P(cur→next).

## BATADAL En İyi Parametre (zaman sıralı test)

```text
window_size   = 4
alphabet_size = 6

F1-score = 0.074
Recall   = 0.857
State sayısı ≈ 239
```

> Not: BATADAL'da yüksek recall (%85.7) düşük precision (%4) ile birlikte gelir.
> Pozitif sınıf oranı ~%4 olduğundan F1 değeri yapısal olarak düşük kalmaktadır.

---

# Kullanılan Değerlendirme Metrikleri

Projede aşağıdaki değerlendirme metrikleri kullanılmaktadır:

- Accuracy
- Precision
- Recall
- F1-Score

Ek olarak: state count, transition density, path probability, ROC-AUC, Average Precision analizleri gerçekleştirilmektedir.

## İstatistiksel Güvenilirlik

### Deney Tekrarı

Her rastgele işlem içeren deney **5 farklı seed** [42, 123, 2026, 7, 999] ile çalıştırılmış, sonuçlar **ortalama ± standart sapma** olarak raporlanmıştır.

**SKAB** — fold bazlı sonuçlar (window=4, alphabet=3):

| Fold | F1 | Recall | Precision |
| :---: | :---: | :---: | :---: |
| 1 | 0.171 | 0.138 | 0.227 |
| 2 | 0.336 | 0.278 | 0.424 |
| 3 | 0.224 | 0.172 | 0.324 |
| 4 | 0.316 | 0.217 | 0.580 |
| 5 | 0.203 | 0.170 | 0.253 |
| **mean±std** | **0.250 ± 0.072** | 0.195 ± 0.055 | 0.362 ± 0.144 |

**BATADAL** — zaman sıralı test kümesi sonuçları (window=4, alphabet=3):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| :---: | :---: | :---: | :---: | :---: |
| 0.569 | 0.024 | 0.286 | 0.043 | 0.362 |

### Wilcoxon İstatistiksel Testleri

Otomata senaryo karşılaştırması için Wilcoxon signed-rank testi (`scipy.stats.wilcoxon`) uygulanmıştır.
Modül: `src/utils/statistical_tests.py` → `run_wilcoxon_pairwise()`

| Karşılaştırma | p-değeri | Anlamlı (α=0.05)? |
| :--- | :---: | :---: |
| SKAB: Original vs Gaussian Noise | 0.0625 | Hayır |
| SKAB: Noise vs Controlled Unseen | 0.0625 | Hayır |
| SKAB: Original vs Controlled Unseen | 0.0625 | Hayır |

**Yorum:** p=0.0625, n=5 için Wilcoxon testinin üretebileceği **minimum p değeridir**. Yani senaryo farklılıkları istatistiksel olarak kanıtlanamayacak kadar küçük bir örneklem üzerinde test edilmektedir. Bu, küçük örneklem boyutunun (n=5 seed/fold) istatistiksel gücü doğrudan sınırladığını göstermekte; etki büyüklükleri (özellikle unseen F1=0.289 vs original F1=0.171) pratik anlamlılık taşısa da istatistiksel anlamlılık eşiğine ulaşılamamaktadır.

---

# Derin Öğrenme Modelleri

Projede aşağıdaki PyTorch tabanlı modeller uygulanmıştır:

- LSTM
- GRU
- 1D-CNN

Modeller çok değişkenli (multivariate) scaled veri üzerinde sliding-window ile sequence oluşturarak eğitilir. Derin öğrenme modelleri ile otomata modelinin karşılaştırması **DL vs Otomata Karşılaştırması** bölümünde sunulmaktadır.

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
│   ├── raw_results.csv        # Grid 1 — tüm run'ların ham metrikleri
│   ├── raw_results_wbce.csv   # Grid 2 — weighted BCE ham metrikleri
│   ├── table1.md / table1.csv
│   ├── table1_wbce.md / table1_wbce.csv
│   ├── table2.md / table2.csv
│   └── table2_wbce.md / table2_wbce.csv
├── models/                    # Rezerve dizin (create_required_dirs)
└── logs/                      # Rezerve dizin
```

`<run_id>` formatı: `{dataset}_{model}_{scenario}_seed{seed}_fold{fold|NA}`  
Örnek: `batadal_lstm_original_seed42_foldNA`  
Grid 2 (weighted BCE) soneki: `..._foldNA_wbce`

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
- `DL_NUM_WORKERS=0`, `DL_PIN_MEMORY=True`, `DL_USE_AMP=True` (CPU ortamında otomatik devre dışı; Windows DataLoader benchmark sonrası `num_workers=0` seçildi)

Grid 2 ek sabitleri:

- `DL_POS_WEIGHT_MAX=25.0` — train pencere pozitif oranından türetilen `pos_weight` tavanı
- `DL_WEIGHTED_BCE_SUFFIX="_wbce"` — weighted run klasör/metrik ayrımı

---

# Derin Öğrenme — Deney Sonuçları ve Analiz (Taslak)

> Tablolardaki değerler, her biri 270 deney içeren iki grid'in ham çıktılarından (`raw_results.csv`, `raw_results_wbce.csv`) türetilmiştir. DL–Otomata karşılaştırması son bölümde sunulmaktadır.

## Deney Tasarımı Özeti

Derin öğrenme deneyleri, proje tanımında belirtilen deney protokolü esas alınarak yürütülmüştür: üç model (LSTM, GRU, 1D-CNN), iki veri seti (SKAB, BATADAL), üç senaryo (original, noise, unseen) ve beş bağımsız rastgele tohum (seed). SKAB için dosya bazlı GroupKFold (5 fold), BATADAL için zaman sıralı %60/%20/%20 bölünme uygulanmıştır.

| Parametre | Değer |
| :--- | :--- |
| Modeller | LSTM, GRU, 1D-CNN |
| Veri setleri | SKAB (GroupKFold, 5 fold), BATADAL (zaman sıralı %60/%20/%20) |
| Senaryolar | original, noise (Gaussian), unseen (distribution drift) |
| Seed'ler | 42, 123, 2026, 7, 999 |
| Toplam run / grid | **270** (SKAB: 225 fold-run, BATADAL: 45 run) |
| Kayıp (Grid 1) | `BCEWithLogitsLoss()` — pos_weight yok |
| Kayıp (Grid 2) | `BCEWithLogitsLoss(pos_weight=(1−p)/p)` — train pencere pozitif oranı; tavan 25 |

**Sınıf dağılımı gözlemi:** SKAB'da ortalama pos_weight ≈ **1.85** (görece dengeli sınıf dağılımı). BATADAL'da ≈ **23.28** (pozitif sınıf oranı ~%4) — belirgin sınıf dengesizliği.

## Grid 1 — Baseline BCE

**Komut:** `python -m scripts.run_deep_learning` (`--fast` ile görselsiz hızlandırma kullanılabilir)

**Amaç:** Standart ikili çapraz entropi (BCE) kayıp fonksiyonu ile sliding-window tabanlı DL modellerinin SKAB ve BATADAL üzerindeki temel performansının ölçülmesi.

**Özet bulgular:**

- **SKAB:** En yüksek F1 değeri 1D-CNN modelinde elde edilmiştir (F1 ≈ 0.58); GRU ve LSTM sırasıyla ~0.49 ve ~0.46 düzeyindedir. Tüm modeller, beş seed üzerinde anlamlı F1 skorları üretmiştir.
- **BATADAL:** Deneylerin büyük bölümünde **majority-class collapse** (çoğunluk sınıfı çökmesi) gözlenmiştir (accuracy ≈ 0.90, F1 ≈ 0). Belirgin istisna: **LSTM / seed123 / F1 = 0.780**. GRU modelinde seed42 ile F1 = 0.407 gibi sınırlı başarılar kaydedilmiştir.
- **Collapse oranı (BATADAL, tüm senaryolar):** 25/45 run (%55.6) — tanım: recall < 0.05 ve accuracy > 0.85.

### Tablo 1a — Grid 1, Original Senaryo (F1 ± std)

| Model | SKAB | BATADAL |
| :--- | :---: | :---: |
| **LSTM** | 0.463 ± 0.216 | 0.156 ± 0.349 |
| **GRU** | 0.492 ± 0.256 | 0.143 ± 0.167 |
| **1D-CNN** | **0.577 ± 0.091** | 0.065 ± 0.078 |

**Model sıralaması (SKAB):** 1D-CNN > GRU > LSTM. **BATADAL:** Tüm modeller düşük ortalama F1 değerleri sergilemiştir; ortalamalar büyük ölçüde seed123 LSTM aykırı değerine (outlier) bağlıdır.

### Grid 1 — SKAB Detay (Original)

| Model | Ort. F1 | Ort. Acc | Ort. Recall | Ort. Prec | Ort. ROC-AUC | En iyi run | En düşük run |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| LSTM | 0.463 | 0.684 | 0.440 | 0.551 | 0.660 | seed7/fold4, F1=0.727 | seed2026/fold1, F1=0.038 |
| GRU | 0.492 | 0.715 | 0.473 | 0.588 | 0.706 | seed999/fold4, F1=0.738 | seed42/fold3, F1=0.005 |
| 1D-CNN | **0.577** | 0.729 | 0.535 | 0.651 | 0.739 | seed999/fold4, F1=0.762 | seed2026/fold3, F1=0.391 |

**Seed bazlı F1 (GRU, fold ortalaması):** 7→0.543, 42→0.516, 123→0.522, 999→0.513, 2026→0.366

### Grid 1 — SKAB Senaryo Karşılaştırması

| Model | Original F1 | Noise F1 | Unseen F1 | Gürültü kaybı |
| :--- | :---: | :---: | :---: | :---: |
| LSTM | 0.463 | 0.453 | 0.270 | ~%2.1 |
| GRU | 0.492 | 0.481 | 0.220 | ~%2.3 |
| 1D-CNN | 0.577 | 0.548 | 0.235 | ~%5.0 |

SKAB veri setinde gürültü etkisi sınırlıdır (%2–5 F1 düşüşü). Unseen (drift) senaryosunda F1 değerleri belirgin biçimde azalmaktadır (~0.22–0.27).

### Grid 1 — BATADAL Detay (Original)

| Model | Ort. F1 | Ort. Acc | Ort. Recall | Ort. Prec | Ort. ROC-AUC | Medyan F1 | F1>0.05 seed sayısı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM | 0.156 | 0.913 | 0.138 | 0.180 | 0.980 | 0.000 | 1/5 |
| GRU | 0.143 | 0.906 | 0.090 | 0.414 | 0.984 | 0.135 | 3/5 |
| 1D-CNN | 0.065 | 0.904 | 0.035 | 0.575 | 0.622 | 0.025 | 2/5 |

**Seed bazlı F1 (LSTM):** 123→**0.780**, diğer tüm seed'ler→0.000

**Seed bazlı F1 (GRU):** 42→0.407, 2026→0.172, 123→0.135, 7→0.000, 999→0.000

**En yüksek performanslı BATADAL run'ı (Grid 1):** LSTM / seed123 / original — F1=0.780, Acc=0.962, Recall=0.688, Precision=0.797, ROC-AUC=0.989

**Majority-class collapse profili (LSTM / seed42):** F1=0.000, Acc=0.900, Recall=0.000, ROC-AUC=0.983 — model örnekleri sıralayabilmekte (yüksek ROC-AUC), ancak varsayılan eşik değerinde tüm örnekleri negatif sınıfa atamaktadır.

### Grid 1 — BATADAL Senaryo Karşılaştırması

| Model | Original F1 | Noise F1 | Unseen F1 |
| :--- | :---: | :---: | :---: |
| LSTM | 0.156 | 0.163 | 0.189 |
| GRU | 0.143 | 0.146 | **0.241** |
| 1D-CNN | 0.065 | 0.069 | 0.027 |

Unseen senaryosunda GRU (F1=0.241) ve LSTM seed123 (unseen F1=0.816) belirli düzeyde sinyal üretmiştir; original senaryoda genel tablo majority-class collapse ağırlıklıdır.

---

## Grid 2 — Weighted BCE (Imbalance-Aware)

**Komut:** `python -m scripts.run_deep_learning --weighted-bce`

**Amaç:** Eğitim kümesindeki sliding-window pozitif oranından türetilen `pos_weight` katsayısı ile sınıf dengesizliğinin etkisini azaltmak ve pozitif sınıf recall değerini artırmak.

**Özet bulgular:**

- **SKAB:** **GRU modeli en yüksek performansı** göstermiştir (F1 ≈ 0.62 ± 0.10). Grid 1'e kıyasla **+0.126 F1** artışı ve belirgin ölçüde düşük standart sapma (0.256→0.095) gözlenmiştir. 1D-CNN değerleri neredeyse değişmemiş; LSTM hafif artış göstermiştir.
- **BATADAL:** Collapse oranı Grid 1 ile **aynı** kalmıştır (%55.6). Ortalama F1 değerleri Grid 1 ile eşdeğer veya daha düşüktür. seed123 LSTM: 0.780→0.708; seed42 GRU: 0.407→0.041 (**performans düşüşü**).
- Weighted BCE, SKAB'da GRU performansını sistematik biçimde iyileştirirken BATADAL'da yapısal sorunu gidermemiştir; sınıf dengesizliğine yönelik bu müdahalenin veri setine özgü etkileri ayrıca değerlendirilmelidir.

### Tablo 1b — Grid 2, Original Senaryo (F1 ± std)

| Model | SKAB | BATADAL |
| :--- | :---: | :---: |
| **LSTM** | 0.474 ± 0.114 | 0.152 ± 0.312 |
| **GRU** | **0.618 ± 0.095** | 0.025 ± 0.038 |
| **1D-CNN** | 0.576 ± 0.087 | 0.012 ± 0.028 |

**Model sıralaması (SKAB):** **GRU > 1D-CNN > LSTM** (Grid 1'deki 1D-CNN > GRU sıralamasının tersine dönmüştür).

### Grid 2 — SKAB Detay (Original)

| Model | Ort. F1 | Ort. Acc | Ort. Recall | Ort. Prec | Ort. ROC-AUC | En iyi run | En düşük run |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| LSTM | 0.474 | 0.622 | 0.499 | 0.482 | 0.610 | seed42/fold4, F1=0.602 | seed999/fold2, F1=0.229 |
| GRU | **0.618** | 0.728 | 0.642 | 0.622 | 0.775 | seed123/fold4, F1=0.768 | seed42/fold1, F1=0.290 |
| 1D-CNN | 0.576 | 0.702 | 0.578 | 0.591 | 0.724 | seed999/fold0, F1=0.756 | seed42/fold0, F1=0.432 |

**Seed bazlı F1 (GRU, fold ortalaması):** 7→0.616, 42→0.572, 123→0.643, 999→0.640, 2026→**0.620** (Grid 1'de 2026→0.366; **+0.254** iyileşme)

### Grid 2 — SKAB Senaryo Karşılaştırması

| Model | Original F1 | Noise F1 | Unseen F1 | Gürültü kaybı |
| :--- | :---: | :---: | :---: | :---: |
| LSTM | 0.474 | 0.467 | 0.293 | ~%1.6 |
| GRU | **0.618** | 0.606 | **0.321** | ~%2.0 |
| 1D-CNN | 0.576 | 0.557 | 0.288 | ~%3.4 |

Weighted BCE, unseen senaryosunda da SKAB veri setinde tutarlı iyileşme sağlamıştır (GRU: 0.220→0.321).

### Grid 2 — BATADAL Detay (Original)

| Model | Ort. F1 | Ort. Acc | Ort. Recall | Ort. Prec | Ort. ROC-AUC | Medyan F1 | F1>0.05 seed sayısı |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM | 0.152 | 0.895 | 0.135 | 0.174 | 0.964 | 0.000 | 1/5 |
| GRU | 0.025 | 0.895 | 0.015 | 0.085 | 0.962 | 0.000 | 1/5 |
| 1D-CNN | 0.012 | 0.884 | 0.008 | 0.035 | 0.353 | 0.000 | 1/5 |

**Seed bazlı F1 (LSTM):** 123→**0.708**, 999→0.050, diğerleri→0.000

**Karşılaştırmalı örnek — BATADAL / GRU / seed42 / original:**

| Metrik | Grid 1 (Baseline) | Grid 2 (Weighted) |
| :--- | :---: | :---: |
| F1 | **0.407** | 0.041 |
| Recall | 0.275 | 0.025 |
| Precision | 0.786 | 0.118 |
| ROC-AUC | 0.983 | 0.965 |

### Grid 2 — BATADAL Senaryo Karşılaştırması

| Model | Original F1 | Noise F1 | Unseen F1 |
| :--- | :---: | :---: | :---: |
| LSTM | 0.152 | 0.152 | 0.172 |
| GRU | 0.025 | 0.056 | 0.170 |
| 1D-CNN | 0.012 | 0.013 | 0.032 |

---

## Grid 1 vs Grid 2 — Karşılaştırmalı Analiz

### Tablo 1c — F1 Değişimi (Grid 2 − Grid 1, Original Senaryo)

| Model | Δ SKAB | Δ BATADAL | Değerlendirme |
| :--- | :---: | :---: | :--- |
| **LSTM** | +0.012 | −0.004 | SKAB'da sınırlı artış; BATADAL'da anlamlı değişim yok |
| **GRU** | **+0.126** | −0.117 | SKAB'da belirgin iyileşme; BATADAL'da belirgin düşüş |
| **1D-CNN** | ~0 | −0.052 | SKAB'da stabil; BATADAL'da düşüş |

### Ana Bulgular

| Bulgu | Detay |
| :--- | :--- |
| SKAB / GRU | +0.126 F1, seed std 0.256→0.095 — weighted BCE SKAB'da olumlu etki |
| SKAB / 1D-CNN | Değişim yok — model zaten dengeli sınıf dağılımında yeterli performans göstermekteydi |
| BATADAL genel | Collapse oranı %55.6 — weighted BCE yapısal sorunu gidermedi |
| BATADAL seed42 / GRU | 0.407 → 0.041 — kısmen başarılı konfigürasyonda performans kaybı |
| BATADAL seed123 / LSTM | 0.780 → 0.708 — en yüksek F1 değerinde hafif düşüş |
| F1>0.05 run sayısı (BATADAL original) | Grid 1 GRU 3/5 → Grid 2 GRU 1/5 |

**Yorum:** `pos_weight ≈ 23.3` formülasyonu, SKAB'da (pos_weight ≈ 1.85) GRU optimizasyonunu sistematik biçimde iyileştirirken, BATADAL'da seed42/GRU gibi kısmen başarılı konfigürasyonlarda performans düşüşüne yol açmıştır. Sınıf dengesizliği düzeltmesinin etkisinin veri seti özelliklerine ve başlatma (seed) koşullarına bağlı olduğu görülmektedir.

---

## SKAB — Birleşik Değerlendirme

SKAB veri setinde derin öğrenme modelleri, belirlenen deney protokolü kapsamında **tutarlı anomali tespiti** performansı sergilemiştir.

**Referans DL sonucu (otomata karşılaştırması için):** Grid 2 / GRU / original — **F1 = 0.618 ± 0.095** (5 seed, 5 fold ortalaması).

**Gerekçe:** Grid 2 (weighted BCE), güncel pipeline'ı temsil etmektedir; SKAB'da en yüksek ve en stabil performans GRU modeli ile elde edilmiştir. Grid 1 ile karşılaştırma, DL içi ablation analizi olarak sunulacaktır.

**Gürültü dayanıklılığı:** Her iki gridde SKAB F1 düşüşü ~%2–5 aralığındadır; düşük seviyeli Gaussian gürültü altında modeller görece stabil kalmıştır.

**Unseen (drift):** Dağılım kayması (distribution drift) uygulandığında F1 değeri ~0.29–0.32'ye (Grid 2 GRU) düşmektedir.

**En yüksek 5 run (Grid 2, original):** Tümü SKAB veri setine aittir — GRU/1D-CNN, F1 0.71–0.77 aralığında.

---

## BATADAL — Performans Analizi

BATADAL veri setinde derin öğrenme modelleri **çoğunlukla majority-class collapse** sergilemiştir. Bu bulgu, veri seti karakteristiği ile model mimarisi arasındaki uyumsuzluğun nicel olarak ortaya konması açısından değerlendirilmelidir.

### Düşük Performansın Olası Nedenleri

1. **Belirgin sınıf dengesizliği:** Pozitif sınıf oranı ~%4 (pos_weight ≈ 23.3). Model, tüm örnekleri negatif sınıfa atayarak accuracy ≈ %90 elde edebilmekte; F1 skoru ise sıfıra inmektedir.
2. **Accuracy metriğinin yanıltıcı etkisi:** Yüksek accuracy değerleri, dengesiz anomali tespiti problemlerinde tek başına yeterli değildir. F1, precision ve recall birlikte raporlanmalıdır.
3. **ROC-AUC ile F1 arasındaki ayrışma:** Collapse gösteren run'larda ROC-AUC 0.73–0.98 iken F1=0 gözlenmektedir; model ayırım gücüne sahip olmakla birlikte varsayılan sınıflandırma eşiği (0.5) uygun değildir.
4. **Seed kaynaklı instabilite:** Aynı protokol altında seed123 LSTM F1≈0.71–0.78 iken diğer dört seed F1≈0 değerindedir; sonuçların tekrarlanabilirliği sınırlıdır.
5. **Veri seti–model uyumsuzluğu:** SKAB (GroupKFold, ~%35 pozitif) ile BATADAL (~%4 pozitif) farklı istatistiksel özellikler taşımaktadır; aynı mimari her iki veri setinde eşit düzeyde verimli değildir.

### Temsilci Run Örnekleri

| Örnek türü | Run | F1 | Acc | Recall | ROC-AUC | Yorum |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| Accuracy yanılgısı | BATADAL LSTM seed42 (Grid 2) | 0.000 | 0.900 | 0.000 | 0.983 | Yüksek accuracy; anomali tahmini yapılmamaktadır |
| En yüksek performans | BATADAL LSTM seed123 (Grid 1) | 0.780 | 0.962 | 0.688 | 0.989 | Tek seed'e bağlı aykırı değer; genellenebilirlik sınırlı |
| Weighted BCE etkisi | BATADAL GRU seed42 | 0.407→0.041 | — | — | — | Grid 1'de kabul edilebilir F1; Grid 2'de belirgin düşüş |

### BATADAL Bulgularının Özeti

> SKAB veri setinde GRU tabanlı derin öğrenme modelleri F1≈0.62 düzeyinde tutarlı anomali tespiti sağlarken, BATADAL veri setinde aşırı sınıf dengesizliği (%4 pozitif) nedeniyle modeller çoğunlukla majority-class collapse sergilemiştir. Bu durum, black-box DL yaklaşımının veri seti karakteristiğine güçlü bağımlılığını göstermekte ve otomata tabanlı yorumlanabilir modellerle karşılaştırma için anlamlı bir zemin oluşturmaktadır.

**Raporlama notu:** BATADAL Tablo 1 ortalamaları (0.01–0.16 aralığı) tek başına özet metrik olarak sunulmamalıdır; **en yüksek performanslı run (seed123), collapse analizi ve medyan F1=0** birlikte değerlendirilmelidir.

---

## Tablo 2 — Senaryo Analizi (Her İki Grid)

Proje tanımında öngörülen Tablo 2 formatına uygun özet (`outputs/metrics/table2.md`, `table2_wbce.md`):

### Gürültü Etkisi (Original / Gürültülü F1)

| Model | Grid 1 SKAB | Grid 2 SKAB | Grid 1 BATADAL | Grid 2 BATADAL |
| :--- | :--- | :--- | :--- | :--- |
| LSTM | 0.463 / 0.453 | 0.474 / 0.467 | 0.156 / 0.163 | 0.152 / 0.152 |
| GRU | 0.492 / 0.481 | 0.618 / 0.606 | 0.143 / 0.146 | 0.025 / 0.056 |
| 1D-CNN | 0.577 / 0.548 | 0.576 / 0.557 | 0.065 / 0.069 | 0.012 / 0.013 |

### Unseen Analizi (F1 ± std)

| Model | Grid 1 SKAB | Grid 2 SKAB | Grid 1 BATADAL | Grid 2 BATADAL |
| :--- | :--- | :--- | :--- | :--- |
| LSTM | 0.270 ± 0.169 | 0.293 ± 0.141 | 0.189 ± 0.352 | 0.172 ± 0.203 |
| GRU | 0.220 ± 0.220 | 0.321 ± 0.191 | 0.241 ± 0.173 | 0.170 ± 0.117 |
| 1D-CNN | 0.235 ± 0.175 | 0.288 ± 0.179 | 0.027 ± 0.060 | 0.032 ± 0.022 |

---

## DL Deneylerinin Genel Değerlendirilmesi

| Konu | Sonuç |
| :--- | :--- |
| Referans DL sonucu | **Grid 2 / SKAB / GRU** — F1≈0.62 ± 0.10 |
| İki grid'in raporlanması | Grid 1: temel BCE deneyleri; Grid 2: sınıf dengesizliği müdahalesi |
| BATADAL performansı | Majority-class collapse baskın; veri seti–model etkileşimi analiz edilmeli |
| Otomata karşılaştırması | Tamamlandı — DL vs Otomata bölümüne bakınız |

### Bulguların Yorumlanması

| Desteklenen ifadeler | Kaçınılması gereken ifadeler |
| :--- | :--- |
| SKAB'da GRU F1=0.62±0.10, 5 seed üzerinde stabil | Weighted BCE'nin BATADAL performansını iyileştirdiği iddiası |
| BATADAL'da majority-class collapse (%56 run) | Bağlam olmaksızın "başarısızlık" vurgusu |
| Weighted BCE'nin SKAB GRU'da +12.6 puanlık F1 artışı | F1=0 iken yalnızca accuracy=%90 vurgusu |
| Weighted BCE'nin BATADAL seed42/GRU'da performans düşüşü | — |
| DL performansının veri seti karakteristiğine bağımlılığı | — |

---

## DL vs Otomata Karşılaştırması

Bu bölüm, derin öğrenme ve olasılıksal otomata modellerini dört eksen üzerinde karşılaştırır: performans, genellenebilirlik, gürültüye dayanıklılık ve açıklanabilirlik.

### SKAB — Model Karşılaştırması (Sabit Parametreler: window=4, alphabet=3)

| Model | F1 | Recall | Precision | Notlar |
| :--- | :---: | :---: | :---: | :--- |
| **DL — GRU (Grid 2)** | **0.618 ± 0.095** | — | — | 5 seed, GroupKFold, referans model |
| DL — LSTM (Grid 2) | 0.474 ± 0.095 | — | — | 5 seed |
| DL — 1D-CNN (Grid 2) | 0.576 ± 0.110 | — | — | 5 seed |
| **Automata (sabit params)** | 0.250 ± 0.072 | 0.195 ± 0.055 | 0.362 ± 0.144 | GroupKFold, 5 fold |
| **Automata (en iyi: w=5, a=6)** | **0.494 ± 0.069** | — | — | GroupKFold, 5 fold |

**SKAB Yorumu:** DL modelleri (özellikle GRU) belirgin şekilde daha yüksek F1 üretmektedir. Ancak otomata modeli, sabit parametrelerle makul bir performans sağlarken parametre optimizasyonuyla (%49.4 F1) DL’nin alt sınırına yaklaşmaktadır. Otomata modelinin temel avantajı her kararın matematiksel olarak gerekçelendirilebilmesidir.

### BATADAL — Model Karşılaştırması

| Model | F1 | Recall | Precision | Notlar |
| :--- | :---: | :---: | :---: | :--- |
| DL — En iyi (LSTM seed123) | 0.780 | 0.688 | — | Tek seed aykırı değeri |
| DL — Ortalama (Grid 2) | 0.156 ± 0.349 | — | — | Çoğunluk sınıfı çöküşü |
| **Automata (sabit params)** | **0.043** | 0.286 | 0.024 | Zaman sıralı değerlendirme |

**BATADAL Yorumu:** BATADAL veri setinde (~%4 pozitif sınıf) her iki model ailesi de zorlanmaktadır. DL modelleri %56 oranında majority-class collapse sergilemektedir. Otomata modeli düşük F1 üretse de yüksek recall (0.286) ile anomali tespitinde belirli bir duyarlılık sağlamaktadır; kararlara her zaman olasılıksal gerekçe eklenebilmektedir.

---

### Senaryo Karşılaştırması — Otomata (window=4, alphabet=3, 5 seed)

| Senaryo | SKAB F1 | BATADAL F1 | Yorum |
| :--- | :---: | :---: | :--- |
| Orijinal | 0.171 | 0.043 | Taban performans |
| Gaussian Noise (std=0.05) | 0.180 ± 0.005 | 0.045 ± 0.001 | Gürültüye karşı stabil; F1 ~+0.5% |
| Controlled Unseen (%10) | 0.289 ± 0.029 | 0.035 ± 0.012 | Küçük alfabe ile unseen oluşumu sınırlı |

**Gürültü Etkisi:** Sabit parametre (w=4, a=3) ile elde edilen SAX sembolik temsili, düşük seviyeli Gaussian gürültüye (std=0.05) karşı oldukça dayanıklıdır. Sembolik kodlama, gürültüyü yumuşatma etkisi göstermektedir.

**Unseen Davranışı:** w=4, a=3 konfigürasyonunda alfabe boyutu 3 harfle (a, b, c) sınırlı olduğundan, üretilen "unseen" pattern’ların büyük çoğunluğu yine de eğitim sözlüğünde yer almaktadır (unseen sayısı ≈ 0). Bu, küçük alfabe boyutlarında sözlüğün eğitim verisiyle doygunluğa ulaştığını ve Levenshtein mekanizmasının ağırlıklı olarak SAX haritalama hatasından kaçınmak için devreye girdiğini göstermektedir. Daha büyük alfabelerle (w=6, a=6) gerçek unseen oranı %10’un üzerine çıkmaktadır.

### Senaryo Karşılaştırması — DL (Grid 2, 5 seed)

| Senaryo | SKAB F1 | BATADAL F1 |
| :--- | :---: | :---: |
| Orijinal | 0.618 ± 0.095 (GRU) | 0.156 ± 0.349 |
| Gaussian Noise | 0.606 ± 0.095 (GRU) | ~0.152 |
| Unseen (drift) | 0.321 ± 0.191 (GRU) | ~0.170 |

---

### Parametre Analizi — Otomata SKAB (GroupKFold, mean ± std)

| window | alphabet | F1 | Recall | State Sayısı |
| :---: | :---: | :---: | :---: | :---: |
| 3 | 3 | 0.240 ± 0.026 | — | ~27 |
| 4 | 3 | 0.250 ± 0.072 | 0.195 | ~81 |
| 4 | 6 | 0.476 ± 0.091 | — | ~682 |
| 5 | 6 | **0.494 ± 0.069** | — | ~1420 |
| 6 | 6 | 0.392 ± 0.054 | — | ~1998 |

**Parametre Gözlemleri:**
- Alphabet boyutu arttıkça state sayısı ve F1 birlikte yükselir — ancak belirli bir noktadan sonra (w=6, a=6) F1 düşmeye başlar. Fazla granülerlik aşırı parçalanmaya neden olmaktadır.
- Optimal nokta: window=5, alphabet=6 (F1=0.494, GroupKFold)
- Geçiş yoğunluğu (transition density) büyük alphabelerle keskin düşer (0.11 → 0.0006), seyrek Markov zinciri yapısına işaret eder.

---

### Açıklanabilirlik Analizi

Otomata modelinin temel avantajı, her karar için tam açıklama üretmesidir:

```json
{
  "time_step": 142,
  "state": "bbbb",
  "pattern": "bbbc",
  "status": "seen",
  "mapped_to": null,
  "transitions": [
    {"transition": "bbbb -> bbbc", "probability": 0.0312},
    {"transition": "bbbc -> bbcc", "probability": 0.0891}
  ],
  "probability": 0.002780,
  "confidence_score": 0.002780,
  "decision": "anomaly"
}
```

- **Düşük olasılık → Anomali:** P(sequence) < 0.10 → sistem anomali işaretler
- **Yüksek olasılık → Normal:** P(sequence) ≥ 0.10 → beklenen davranış
- **Unseen mekanizması:** Görülmemiş pattern’lar Levenshtein ile en yakın bilinen state’e eşlenir; eşleme mesafesi raporlanır

DL modelleri bu türde bir karar gerekçesi üretemez. Kararlar yalnızca ağırlık matrislerinin çıktılarına dayanır ve anlaşılması güçtür.

### Genel Değerlendirme

| Kriter | DL (GRU, SKAB) | Automata (en iyi) |
| :--- | :---: | :---: |
| Performans (F1) | **0.618** | 0.494 |
| Genellenebilirlik | Orta (BATADAL’da collapse) | Orta (BATADAL zayıf) |
| Gürültüye dayanıklılık | Yüksek (~-2%) | **Çok yüksek (~+1%)** |
| Açıklanabilirlik | Düşük (black-box) | **Tam (her karar gerekçeli)** |
| Eğitim süresi | Yüksek (270 run/grid) | **Düşük (deterministik)** |
| Parametre hassasiyeti | Yüksek (seed, lr, arch) | Orta (window, alphabet) |

**Sonuç:** Yüksek doğruluk öncelikli senaryolarda DL tercih edilebilir. Yorumlanabilirlik, denetlenebilirlik veya kaynak kısıtı (CPU) gerektiren durumlarda olasılıksal otomata, makul performansıyla anlamlı bir alternatif sunmaktadır.

---

# Deneysel Senaryolar

Deneyler üç farklı senaryo altında yürütülmüş, **tüm karşılaştırmalar sabit parametrelerle** (window=4, alphabet=3) yapılmıştır:

| Senaryo | Açıklama | Parametre |
| :--- | :--- | :--- |
| **Orijinal** | İşlenmiş test verisi olduğu gibi kullanılır | — |
| **Gaussian Noise** | Test PC1 serisine Gaussian gürültü eklenir | std=0.05, 5 seed |
| **Controlled Unseen** | Test pattern’larının %10’u değiştirilir | 5 seed |

Senaryo karşılaştırma sonuçları ve yorumları için **Senaryo Karşılaştırması — Otomata** ve **DL vs Otomata Karşılaştırması** bölümlerine bakınız.

---

# Rapor Tabloları

Bu bölüm, proje rapor şablonundaki beş tabloyu SKAB ve BATADAL veri setleri üzerindeki gerçek deney
sonuçlarıyla sunmaktadır. DL model değerleri takım arkadaşı tarafından, Automata değerleri bu bölümde
doldurulmuştur.

---

## Tablo 1 — Model Performansı ve Stabilitesi (F1-score ± Standart Sapma)

> Değerlendirme: SKAB → 5-fold GroupKFold (source_file bazlı) | BATADAL → zaman sıralı %60/%20/%20
> Sabit parametreler: window=4, alphabet=3 | Seed'ler: 42, 123, 2026, 7, 999

| Model | SKAB | BATADAL |
| :--- | :---: | :---: |
| LSTM | 0.463 ± 0.216 | 0.156 ± 0.349 |
| GRU | 0.618 ± 0.095 | 0.025 ± 0.038 |
| 1D-CNN | 0.577 ± 0.091 | 0.012 ± 0.028 |
| **Automata** | **0.250 ± 0.072** | **0.043 ± 0.000** |

> **Automata notu:** SKAB değeri 5-fold GroupKFold ortalamasıdır. BATADAL değeri deterministik tek çalıştırma sonucudur (otomata rastgele işlem içermez).

---

## Tablo 2 — Gürültü Etkisi ve Unseen Senaryo Analizi

> Gürültü: Gaussian noise std=0.05 (5 seed) | Unseen: %10 pattern modifikasyonu (5 seed)
> Det. Rate = Recall (unseen senaryosunda anomali yakalama oranı) | Map. Acc. = F1 (eşleme sonrası doğruluk)

| Model | Orijinal F1 | Gürültülü F1 | Det. Rate | Map. Acc. |
| :--- | :---: | :---: | :---: | :---: |
| | **SKAB / BATADAL** | **SKAB / BATADAL** | **SKAB / BATADAL** | **SKAB / BATADAL** |
| LSTM | 0.463 / 0.156 | 0.453 / 0.163 | 0.416 / 0.252 | 0.270 / 0.189 |
| GRU | 0.618 / 0.025 | 0.606 / 0.056 | 0.519 / 0.216 | 0.321 / 0.170 |
| 1D-CNN | 0.577 / 0.012 | 0.548 / 0.013 | 0.432 / 0.019 | 0.288 / 0.032 |
| **Automata** | **0.171 / 0.043** | **0.180 / 0.045** | **0.314 / 0.286** | **0.289 / 0.035** |

> **Automata gürültü yorumu:** SAX sembolik temsili, düşük seviyeli gürültüyü (std=0.05) absorbe etmektedir;
> SKAB F1 değişimi yalnızca +0.009 (±0.005 std). Otomata gürültüye karşı en dayanıklı modeldir.
>
> **Automata unseen yorumu:** w=4, a=3 konfigürasyonunda alfabe küçük (3 harf) olduğundan modifiye edilen
> pattern'ların çoğu eğitim sözlüğünde kalır (unseen count≈0). Levenshtein mekanizması devreye
> girdiğinde sistem çalışmaya devam etmektedir.

---

## Tablo 3 — Cross-Dataset Genellenebilirlik (Automata, F1-score)

> Bir veri setinde eğitilmiş otomata modelinin diğer veri setine uygulanması.
> Farklı veri setlerinin farklı SAX sözlükleri oluşturduğu için Levenshtein eşleme devreye girer.

| Train / Test | SKAB | BATADAL |
| :--- | :---: | :---: |
| **Train: SKAB** | 0.250 ± 0.072 *(GKF)* | 0.080 |
| **Train: BATADAL** | 0.361 | 0.043 *(in-domain)* |

**Unseen pattern oranları (cross-dataset):**

| Yön | Unseen Oranı | Açıklama |
| :--- | :---: | :--- |
| SKAB→BATADAL | %0.0 | BATADAL PC1 örüntüleri SKAB sözlüğünde mevcut |
| BATADAL→SKAB | %30.0 | SKAB örüntülerinin %30'u Levenshtein ile eşlendi |

**Yorumlar:**
- **SKAB→BATADAL (F1=0.080):** Yüksek recall (0.857) ancak düşük precision (0.042). SKAB geçiş olasılıkları BATADAL normal operasyonunu "alışılmadık" bulmakta ve fazla anomali işaretlemektedir.
- **BATADAL→SKAB (F1=0.361):** SKAB örüntülerinin %30'u Levenshtein ile eşlendi; recall=0.648. BATADAL'da görece homojen veri yapısı, SKAB'a kısmen genellenebilmektedir.
- **Genel:** Otomata modeli cross-dataset uygulamada Levenshtein mekanizması sayesinde çalışmaya devam etmektedir; ancak farklı fiziksel sistemlerin farklı istatistiksel özellikler taşıması nedeniyle performans düşmektedir.

---

## Tablo 4 — Automata Parametre Duyarlılık Analizi (F1-score, GroupKFold)

> SKAB veri seti, 5-fold GroupKFold. Her hücre: o parametre kombinasyonu için GroupKFold F1 ortalaması.

### 4a. Tam 4×4 Parametre Tablosu

| Window \ Alphabet | a=3 | a=4 | a=5 | a=6 |
| :---: | :---: | :---: | :---: | :---: |
| **w=3** | 0.240 | 0.326 | 0.376 | 0.410 |
| **w=4** | 0.250 | 0.368 | 0.434 | 0.476 |
| **w=5** | 0.274 | 0.390 | **0.438** | **0.494** |
| **w=6** | 0.235 | 0.329 | 0.384 | 0.392 |

### 4b. Parametre Boyutu Başına Özet (diğer parametrenin ortalaması)

| Parametre | Değer=3 | Değer=4 | Değer=5 | Değer=6 |
| :--- | :---: | :---: | :---: | :---: |
| **Window Size** | 0.338 | 0.382 | **0.399** | 0.335 |
| **Alphabet Size** | 0.250 | 0.353 | 0.408 | **0.443** |

**Gözlemler:**
- **En iyi kombinasyon:** window=5, alphabet=6 → F1=0.494 ± 0.069
- Alphabet size artışı F1'i monoton artırır (a=3→6: +0.193)
- Window size için optimum w=5; w=6'da aşırı parçalanma nedeniyle düşüş
- State sayısı: w=3,a=3→27 | w=5,a=6→1420 | w=6,a=6→1998
- Geçiş yoğunluğu: büyük alphabet ile keskin düşüş (0.111→0.0006)

---

## Tablo 5 — Model Çalışma Süresi (Runtime) Karşılaştırması

> SKAB veri seti (train: 17954 sample, test: 4518 sample), CPU ortamı

| Model | Training Time | Inference Time | Notlar |
| :--- | :---: | :---: | :--- |
| LSTM | — | — | *DL bölümüne bakınız* |
| GRU | — | — | *DL bölümüne bakınız* |
| 1D-CNN | — | — | *DL bölümüne bakınız* |
| **Automata** | **~19 ms** | **~1.6 ms** | 4518 test sample, CPU |

> **Automata runtime yorumu:** Otomata modeli eğitimi yalnızca geçiş sayımından ibaret olduğundan
> son derece hızlıdır (~19 ms). DL modelleri tipik olarak dakikalar-saatler mertebesinde eğitim
> gerektirirken otomata tek geçişte tüm geçiş olasılıklarını öğrenir. Bu, kaynak kısıtlı ortamlarda
> otomata modelinin önemli bir avantajını oluşturmaktadır.

---

# Geliştiriciler

- Derya Gelmez
- Melih İyigören

Kocaeli Üniversitesi  
YazLab-II Projesi