# Model Card — Churn Prediction (churn-service)

Versi 0.1.0 · model `logistic_regression` · dilatih 2026-11-21T13:46:14+00:00 · seed 42

## 1. Tujuan model

Memprediksi probabilitas seorang pelanggan telekomunikasi akan **berhenti berlangganan
(churn)**, agar tim retensi bisa memprioritaskan siapa yang dihubungi lebih dulu.

- **Pengguna:** tim retensi/CRM, analis, dan engineer yang memanggil API.
- **Bentuk keluaran:** probabilitas 0–1 + keputusan biner pada threshold 0,17.
- **Keputusan yang didukung:** prioritas kampanye retensi — *bukan* penolakan layanan,
  bukan penentuan harga individual, dan bukan dasar penagihan.

## 2. Data

| Item | Nilai |
|---|---|
| Sumber | Kaggle `blastchar/telco-customer-churn` (IBM Telco) |
| Baris | 7.043 pelanggan |
| Fitur dipakai | 19 (4 numerik + 15 kategorikal) |
| Target | `Churn` (1 = berhenti), positif 26,54% |
| Split | train 4.930 · val 1.056 · test 1.057 (stratified, seed 42) |
| Excluded | `customerID` (identitas, disimpan untuk pelacakan saja) |

Pembersihan: `TotalCharges` yang berupa teks diubah ke numerik; 11 nilai kosong (semuanya
`tenure = 0`) diisi 0,0. Tidak ada baris yang dihapus.

## 3. Performa

Test set (1.057 baris), threshold 0,17:

| Metrik | Nilai |
|---|---|
| ROC-AUC | 0,8448 |
| PR-AUC | 0,6581 |
| Precision | 0,4547 |
| Recall | 0,8932 |
| F1 | 0,6026 |
| Brier | 0,1362 |
| Akurasi | 68,7% |

Confusion matrix: TP 251 · FP 301 · FN 30 · TN 475.

Bandingkan dengan baseline "semua pelanggan tidak churn" (akurasi 73,4% tetapi recall 0).
Model ini sengaja berkorban presisi untuk menangkap churn: dari 281 pelanggan yang benar-benar
churn di test set, 251 tertangkap (30 lolos), dengan konsekuensi 301 pelanggan loyal ikut
ditawari diskon.

### Metrik per segmen (test, threshold 0,17)

| Segmen | n | Churn aktual | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Contract: Month-to-month | 572 | 42,5% | 0,754 | 0,467 | 0,955 |
| Contract: One year | 229 | 14,4% | 0,743 | 0,358 | 0,576 |
| Contract: Two year | 256 | 2,0% | 0,800 | 0,000 | 0,000 |
| Tenure 0–12 bulan | 322 | 46,0% | 0,763 | 0,513 | 0,946 |
| Tenure 13–36 bulan | 269 | 31,6% | 0,798 | 0,438 | 0,918 |
| Tenure 37–72 bulan | 466 | 10,3% | 0,832 | 0,327 | 0,688 |
| Internet: DSL | 377 | 19,6% | 0,827 | 0,369 | 0,797 |
| Internet: Fiber optic | 441 | 41,5% | 0,798 | 0,519 | 0,967 |
| Internet: Tidak ada | 239 | 10,0% | 0,834 | 0,294 | 0,625 |

Pembacaan penting: pada pelanggan kontrak **Two year** tidak ada satu pun churn yang tertangkap
(0 dari 5 kasus) karena probabilitasnya hampir selalu di bawah 0,17. Model praktis tidak berguna
untuk segmen ini — dan itu wajar: churn di segmen tersebut sangat langka (2%).

## 4. Fitur paling berpengaruh (permutation importance, test)

| # | Fitur | Δ ROC-AUC |
|---|---|---|
| 1 | tenure | 0,2009 ± 0,0096 |
| 2 | Contract | 0,0399 ± 0,0023 |
| 3 | InternetService | 0,0261 ± 0,0067 |
| 4 | MonthlyCharges | 0,0214 ± 0,0039 |
| 5 | TotalCharges | 0,0169 ± 0,0044 |
| 6 | PaymentMethod | 0,0054 ± 0,0017 |

Permutation importance mengukur pengaruh ke prediksi, **bukan kausalitas**. Bahwa `tenure`
paling penting tidak berarti memperpanjang masa berlangganan akan menurunkan churn.

## 5. Batas pemakaian (out of scope)

Jangan pakai model ini untuk:

- **Pelanggan di luar domain dataset**: struktur harga, kontrak, dan layanan harus menyerupai
  Telco IBM. Perusahaan dengan model bisnis berbeda perlu melatih ulang.
- **Segmen kontrak dua tahun** (recall 0) tanpa aturan bisnis tambahan.
- **Menolak layanan atau menaikkan harga** berdasarkan prediksi.
- **Keputusan hukum/keuangan individual** — probabilitas bukan bukti perilaku.
- **Data tanpa `tenure`, `Contract`, atau `MonthlyCharges`** — performa akan turun drastis karena
  ketiganya fitur terkuat.

## 6. Risiko & bias

- **Bias seleksi**: dataset satu perusahaan, satu negara, satu periode. Segmen yang kurang
  terwakili (pelanggan sangat baru, pelanggan tanpa internet) punya estimasi lebih lemah.
- **Bias akibat threshold**: memilih threshold 0,17 berarti 301 dari 776 pelanggan loyal
  (39%) akan menerima intervensi yang tidak perlu. Ini konsekuensi asumsi biaya, bukan netral.
- **Perubahan perilaku**: model tidak memiliki dimensi waktu; kalau pola churn bergeser
  (misalnya pesaing baru), performa bisa turun tanpa peringatan kecuali drift dipantau.
- **Kalibrasi sedang**: Brier 0,1362 dan kurva reliabilitas menunjukkan probabilitas cukup
  dekat dengan frekuensi aktual, tetapi belum sempurna — jangan menafsirkan 0,30 sebagai
  "tepat 30%".

## 7. Pemantauan yang disarankan

1. Jalankan `python -m src.monitor.drift` setiap batch baru; tanggapi `ALERT` (PSI ≥ 0,20)
   dengan evaluasi ulang model.
2. Pantau proporsi prediksi positif di `logs/predictions.jsonl` — lonjakan mendadak biasanya
   berarti perubahan distribusi input, bukan perubahan perilaku pelanggan.
3. Ukur dampak nyata kampanye retensi (efektivitas intervensi) — model ini tidak mengukurnya.
4. Latih ulang minimal tiap kuartal atau saat PSI `ALERT`.
5. `logs/predictions.jsonl` belum dirotasi (shortcut yang diterima, ~200 byte/prediksi).
   Untuk volume sekarang ini tidak masalah; tambahkan rotasi/arsip sebelum log dipakai
   untuk retraining otomatis.

## 8. Informasi teknis

- Algoritma: Logistic Regression (C=1,0, max_iter=2000) di dalam sklearn `Pipeline`
  dengan imputasi + standard scaling (numerik) dan one-hot encoding (kategorikal).
- Artefak: `models/model.pkl` (pipeline lengkap) + `models/model_meta.json`.
- Environment tercatat: Python 3.14.7, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0,
  lightgbm 4.7.0, joblib 1.5.3.
- Model dilatih pada **train split saja**. Sebelum produksi: latih ulang pada train+val
  dengan threshold 0,17 yang sudah dipilih, lalu validasi ulang.
