# Evaluasi Model — logistic_regression

Dibuat oleh `python -m src.evaluate`.
Model artifact: `models\model.pkl` · dilatih 2026-11-21T13:46:14+00:00 · seed 42.

## 1. Perbandingan kandidat (cross-validation 5-fold di train)

| model | ROC-AUC (CV) | PR-AUC (CV) | F1@0.5 (CV) |
|---|---|---|---|
| dummy | 0.5000 ± 0.0000 | 0.2653 ± 0.0005 | 0.0000 ± 0.0000 |
| logistic_regression | 0.8444 ± 0.0116 | 0.6607 ± 0.0192 | 0.5952 ± 0.0213 |
| lightgbm | 0.8321 ± 0.0111 | 0.6423 ± 0.0215 | 0.5742 ± 0.0208 |

Model terpilih: **logistic_regression**.
Alasan: Logistic regression sudah model terbaik menurut PR-AUC.

## 2. Pemilihan threshold (val split, bukan test)

Asumsi biaya: satu false negative (pelanggan churn yang tidak ditangkap) =
**5.0×** biaya satu false positive (diskon retensi yang salah sasaran).

| threshold | precision | recall | FN | FP | biaya_relatif |
|---|---|---|---|---|---|
| 0.50 | 0.646 | 0.593 | 114 | 91 | 661 |
| 0.17 | 0.449 | 0.914 | 24 | 314 | 434 |

Threshold terpilih: **0.17**. Cost curve lengkap ada di `reports/figures/05_cost_curve.png`.
Menurunkan threshold menaikkan recall (lebih banyak churn tertangkap) dengan konsekuensi lebih banyak
pelanggan loyal yang diberi diskon. Keputusan ini murni konsekuensi dari asumsi biaya di `config.yaml`
— kalau rasio biayanya berubah, threshold harus dihitung ulang.

## 3. Performa di test set

Metrik yang tidak bergantung pada threshold (satu nilai, bukan per threshold):

| metrik | nilai |
|---|---|
| ROC-AUC | 0.8448 |
| PR-AUC | 0.6581 |
| Brier | 0.1362 |

Metrik yang bergantung pada threshold:

| metrik | test @0.50 | test @0.17 |
|---|---|---|
| Precision | 0.6916 | 0.4547 |
| Recall | 0.5267 | 0.8932 |
| F1 | 0.598 | 0.6026 |
| TP / FP / FN / TN | 148 / 66 / 133 / 710 | 251 / 301 / 30 / 475 |

Confusion matrix pada threshold 0.17:
TP=251 · FP=301 ·
FN=30 · TN=475.
Gambar: `reports/figures/06_confusion_matrix.png`.

Kalibrasi (Brier score 0.1362): `reports/figures/07_calibration.png`.
Semakin dekat titik ke garis diagonal, semakin jujur probabilitasnya sebagai probabilitas.

## 4. Fitur paling berpengaruh (permutation importance, test)

| peringkat | fitur | delta ROC-AUC |
|---|---|---|
| 1 | tenure | 0.2009 ± 0.0096 |
| 2 | Contract | 0.0399 ± 0.0023 |
| 3 | InternetService | 0.0261 ± 0.0067 |
| 4 | MonthlyCharges | 0.0214 ± 0.0039 |
| 5 | TotalCharges | 0.0169 ± 0.0044 |
| 6 | PaymentMethod | 0.0054 ± 0.0017 |
| 7 | OnlineSecurity | 0.0035 ± 0.0013 |
| 8 | PaperlessBilling | 0.0030 ± 0.0020 |
| 9 | StreamingTV | 0.0022 ± 0.0011 |
| 10 | StreamingMovies | 0.0021 ± 0.0019 |

Gambar: `reports/figures/08_permutation_importance.png`.
Permutation importance mengukur *dampak ke prediksi*, bukan kausalitas.

## 5. Contoh error

| customerID | aktual | prediksi | probabilitas | Contract | tenure | MonthlyCharges |
|---|---|---|---|---|---|---|
| 0870-VEMYL | 1 | 0 | 0.167 | Month-to-month | 5 | 53.85 |
| 0022-TCJCI | 1 | 0 | 0.046 | One year | 45 | 62.70 |
| 2668-TZSPS | 0 | 1 | 0.191 | Month-to-month | 1 | 20.45 |

Dua baris pertama (bila ada) adalah false negative — churn tidak terdeteksi, paling mahal menurut
asumsi biaya. Baris terakhir (bila ada) adalah false positive — pelanggan loyal yang menerima
diskon tanpa perlu.

## 6. Keterbatasan

- **Dataset kecil dan sintetis**: 7.043 baris dari satu perusahaan telekomunikasi; hasil tidak otomatis
  berlaku untuk perusahaan lain dengan struktur harga dan kontrak berbeda.
- **Tanpa dimensi waktu**: split acak, bukan temporal. Model tidak menguji apakah performa bertahan
  saat perilaku pelanggan bergeser dari waktu ke waktu.
- **Asumsi biaya adalah asumsi**: rasio FN:FP 5.0:1.0
  tidak diukur dari data biaya nyata. Threshold yang dipilih adalah konsekuensi langsung dari asumsi itu.
- **Test set dipakai sekali**: angka di atas belum punya confidence interval; dengan 1057 baris,
  selisih kecil antar model tidak bisa dianggap signifikan.
- **Tidak ada data pasca-churn**: model memprediksi niat berhenti, bukan efektivitas program retensi.
- **Artefak dilatih pada train split saja**; sebelum produksi, latih ulang pada train+val dengan
  threshold yang sudah dipilih di sini.
