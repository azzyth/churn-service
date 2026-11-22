# Temuan Data — Telco Customer Churn

Digenerate oleh `python -m src.data.make_dataset`.
Raw file SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` · 4930 baris train,
1056 val, 1057 test.

## Temuan 1 — `TotalCharges` bukan angka
Kolom ini bertipe teks di CSV. Ada **11 baris berisi string kosong**,
dan semuanya milik pelanggan dengan `tenure = 0` (pelanggan baru yang belum pernah ditagih).

**Keputusan:** `pd.to_numeric(errors="coerce")` lalu isi **0.0 hanya untuk `tenure = 0`**.
Tidak ada baris yang dihapus, dan tidak ada imputasi median yang mengarang tagihan.
Sisa nilai kosong setelah pembersihan: 0.

## Temuan 2 — Kelas tidak seimbang
Proporsi churn **26.5%** (7043 baris total). Baseline "semua tidak churn"
sudah menghasilkan akurasi 73.5% tanpa belajar apa pun.

**Keputusan:** akurasi tidak dipakai sebagai metrik utama. Metrik yang dipakai ROC-AUC dan PR-AUC,
ditambah pemilihan threshold berbasis biaya (FN = 5.0×,
FP = 1.0×). Data tidak di-resample/oversample.

## Temuan 3 — `customerID` adalah identitas, bukan fitur
7043 nilai unik untuk 7043 baris (unik, tanpa duplikat).

**Keputusan:** dibuang dari matriks fitur, tetapi tetap disimpan di file split untuk keperluan
batch scoring dan pelacakan error per pelanggan.

## Sinyal tambahan (bukan temuan teknis)
Rata-rata churn pada kontrak `Month-to-month` **42.7%** vs `Two year` **2.8%**.
`Contract` diperkirakan menjadi fitur paling kuat, dan ini masuk akal secara bisnis: pelanggan
tanpa ikatan kontrak paling mudah pindah.
