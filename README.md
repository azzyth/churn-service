# churn-service

Prediksi churn pelanggan telekomunikasi dalam satu project: unduh data, validasi,
latih model, pilih threshold dari biaya, bungkus jadi API, lalu uji dan pantau drift-nya.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7.043 pelanggan, 21 kolom, churn 26,54%).

## Hasil utama

| Metrik | Dummy baseline | Logistic Regression (CV) | LightGBM (CV) | Test (threshold 0,17) |
|---|---|---|---|---|
| ROC-AUC | 0,5000 | **0,8444 ± 0,0116** | 0,8321 ± 0,0111 | 0,8448 |
| PR-AUC | 0,2653 | **0,6607 ± 0,0192** | 0,6423 ± 0,0215 | 0,6581 |
| F1 | 0,0000 | 0,5952 ± 0,0213 | 0,5742 ± 0,0208 | 0,6026 |
| Precision | — | — | — | 0,4547 |
| Recall | — | — | — | **0,8932** |

Model terpilih: **logistic regression**. LightGBM kalah tipis di PR-AUC (0,6423 vs 0,6607),
dan aturan proyek menyebut model paling sederhana yang dipakai kalau selisihnya tidak jelas.

Threshold **0,17** dipilih dari val split dengan asumsi biaya FN = 5× FP:
pada ambang ini recall naik dari 0,593 ke 0,914 di val (FP naik 91 ke 314), dan total biaya
relatif turun dari 661 ke 434. Detail lengkap ada di [`reports/evaluation.md`](reports/evaluation.md).

## Struktur

```
churn-service/
├── config.yaml              # seed, path, hyperparameter, asumsi biaya
├── requirements.txt         # dependency pinned (UTF-8, siap pip)
├── requirements-dev.txt     # + kaggle CLI (sengaja tidak masuk image Docker)
├── run.ps1                  # task runner (pengganti Makefile)
├── Dockerfile
├── LICENSE                  # MIT
├── .github/workflows/       # CI: pipeline penuh + docker smoke test
├── data/
│   ├── raw/                 # CSV Kaggle (tidak di-commit)
│   ├── splits/              # train/val/test.csv + split_manifest.json
│   └── processed/           # sampel reference/normal/drifted untuk uji drift
├── src/
│   ├── config.py            # loader config + seed
│   ├── check.py             # smoke import semua modul
│   ├── data/{download,load,validate,make_dataset}.py
│   ├── features/build.py    # ColumnTransformer + Pipeline
│   ├── train.py             # CV, pemilihan model, artefak
│   ├── evaluate.py          # threshold biaya, metrik test, laporan
│   └── monitor/drift.py     # PSI per fitur
├── app/{main.py,schemas.py} # FastAPI + mount UI di /ui
├── app/static/              # UI noir/8-bit (HTML, CSS, JS, font lokal)
├── scripts/smoke_test.py    # smoke test HTTP untuk container (dipakai CI)
├── tests/                   # 35 test (fitur, API+UI, drift, evaluate)
├── models/                  # model.pkl + model_meta.json
├── reports/                 # evaluation.md, data_findings.md, figures/
└── logs/                    # predictions.jsonl, drift_report.json
```

## Cara menjalankan (dari nol)

Butuh Python 3.14 dan kredensial Kaggle (`~/.kaggle/kaggle.json`) untuk unduh dataset.

```powershell
cd churn-service

# 1. Dependency (kalau memakai venv di folder parent, langsung aktifkan saja)
..\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt   # termasuk kaggle CLI untuk langkah unduh

# 2. Cek kerangka
.\run.ps1 check

# 3. Unduh + bangun split + EDA figur
.\run.ps1 download
.\run.ps1 data

# 4. Latih, lalu evaluasi (threshold dipilih di sini)
.\run.ps1 train
.\run.ps1 evaluate

# 5. Test
.\run.ps1 test

# 6. Drift report (normal vs sengaja digeser)
.\run.ps1 drift

# 7. API
.\run.ps1 serve      # http://127.0.0.1:8000/docs
```

Semua langkah juga bisa dipanggil tanpa PowerShell:

```bash
python -m src.check
python -m src.data.make_dataset
python -m src.train
python -m src.evaluate
python -m src.monitor.drift
python -m pytest tests -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`.\run.ps1 all` menjalankan check, data, train, evaluate, drift, lalu test.

## Contoh request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65
  }'
```

```json
{
  "customer_id": null,
  "churn_probability": 0.691377,
  "churn_prediction": 1,
  "threshold": 0.17,
  "model_version": "0.1.0"
}
```

Endpoint:

| Method | Path | Keterangan |
|---|---|---|
| GET | `/` | redirect ke `/ui/` |
| GET | `/ui/` | antarmuka intake (statis, satu origin dengan API) |
| GET | `/health` | status + apakah model termuat |
| GET | `/model/info` | versi, threshold, metrik test, environment |
| POST | `/predict` | satu pelanggan |
| POST | `/predict/batch` | 1-500 pelanggan (`{"customers": [...]}`) |

Input divalidasi ketat lewat Pydantic: `tenure` 0-72, `MonthlyCharges` > 0, kategori harus
salah satu nilai dataset. Input di luar itu ditolak `422`, tidak dikoreksi diam-diam.
Setiap prediksi ditulis ke `logs/predictions.jsonl`.

## Antarmuka

Buka `http://127.0.0.1:8000/ui/` setelah `run.ps1 serve` (atau `/ui/` di container).
Tidak ada build frontend dan tidak ada CDN: HTML/CSS/JS dan font WOFF2 disajikan langsung
oleh FastAPI dari `app/static/`, jadi origin-nya sama dengan API dan CORS tidak diperlukan.

Tema: meja intake arsip noir dengan konsol putusan bergaya CRT 8-bit. Warnanya tinta hangat,
kertas tulang, lampu amber, dan stempel merah. Tipografi Silkscreen (pixel) dan Courier Prime
(typewriter), keduanya di-self-host.

Isi halaman:

- **Berkas pelanggan**: 19 field dengan urutan dan batas yang sama seperti schema API,
  tiga contoh cepat (fiber baru, setia dua tahun, DSL 18 bulan), serta isian total tagihan
  otomatis (`tenure × bulanan`).
- **Konsol putusan**: probabilitas, meter 24 sel dengan garis ambang, stempel
  `Prioritas retensi` / `Risiko rendah`, dan peringatan bila formulir diubah setelah
  hasil keluar (hasil lama tidak boleh terbaca sebagai hasil baru).
- **Intel model**: metrik test dari `/model/info` dan perbandingan ambang 0,50 vs 0,17.
- **Buku kasus**: 8 prediksi terakhir di sesi itu, disimpan di memori browser saja.

Aksesibilitas: label eksplisit, fokus terlihat, `aria-live` pada probabilitas, status
kesalahan `role="alert"`, dan animasi dimatikan saat `prefers-reduced-motion`.

## CI (GitHub Actions)

`.github/workflows/churn-service.yml` berjalan pada setiap push, pull request, dan bisa
dipicu manual:

1. **pipeline**: install dependency, unduh dataset dari mirror publik IBM dengan cek
   SHA-256 (`16320c9c…`), `src.check`, bangun split + EDA, buktikan split reproducible
   (dua kali `make_dataset`, hash harus sama), latih, evaluasi, drift, lalu seluruh test.
   Model dan laporan diunggah sebagai artifact.
2. **docker**: ambil model dari artifact, build image, jalankan container, tunggu
   `/health`, lalu cek `/ui/`, `/model/info`, dan `/predict` (0,691377 ± 1e-4).

Dataset tidak butuh kredensial Kaggle di CI karena memakai mirror resmi IBM. Isinya identik
dengan CSV Kaggle, hanya beda akhir baris: mirror memakai LF (`16320c9c…`, yang dicek CI),
salinan lokal dari Kaggle memakai CRLF (`88be4b93…`).

Repositori ini tidak menyimpan `data/`, `models/`, dan `logs/`; CI membangun ulang
split, model, dan laporan dari CSV mentah, sedangkan `reports/*.md` dan `reports/figures/`
ikut di-commit supaya bisa dibaca langsung di GitHub.

## Docker

```bash
docker build -t churn-service:0.1.0 .
docker run --rm -p 8000:8000 churn-service:0.1.0
```

UI tersedia di `http://127.0.0.1:8000/ui/`: `COPY app ./app` sudah membawa `app/static`,
jadi tidak ada langkah build frontend di image.

Build butuh `models/model.pkl` + `models/model_meta.json` sudah ada, jadi jalankan
`train` dan `evaluate` dulu (langkah 4 di atas).

Image memakai `python:3.14-slim` agar versi Python sama dengan venv pengembangan
(artefak joblib/scikit-learn tidak mismatch), berjalan sebagai user non-root,
dan punya `HEALTHCHECK` ke `/health`.

Image juga memasang `libgomp1`; tanpa itu `import lightgbm` gagal di image slim
(`OSError: libgomp.so.1: cannot open shared object file`) dan container langsung exit.

## Keputusan desain

- **Satu `Pipeline`** berisi preprocessing + model, sehingga tidak ada leakage dan artefak
  tunggal (`model.pkl`) sudah cukup untuk inference.
- **Test set disentuh sekali**, setelah threshold dipilih dari val.
- **Threshold dari biaya, bukan 0,5**: konsekuensi asumsi FN 5× FP di `config.yaml`.
- **Reproducible**: seed 42, split deterministic (SHA-256 train identik pada dua kali
  `make_dataset`), versi library dicatat di `model_meta.json`.
- **Tanpa dependency tambahan** untuk explainability & drift (permutation importance dan PSI
  dihitung sendiri).

## Drift

`python -m src.monitor.drift` membandingkan batch dengan data referensi:

- batch normal (test split): **19/19 fitur `OK`**, PSI tertinggi 0,038 (`TotalCharges`);
- batch yang sengaja digeser: **5 fitur `ALERT`**: `PaymentMethod`, `tenure`,
  `InternetService`, `Contract`, `MonthlyCharges` (PSI 3,6-8,8).

Ambang: `OK` < 0,10, `WARN` 0,10-0,20, `ALERT` ≥ 0,20 (diatur di `config.yaml`).

## Status verifikasi

| Fase | Check | Status |
|---|---|---|
| 0. Kerangka | `python -m src.check` | ✅ 11 modul ter-import |
| 1. Data | `python -m src.data.make_dataset` | ✅ 24 check PASS, 0 pelanggaran |
| 1. Reproducibility | jalankan dua kali, bandingkan SHA-256 | ✅ identik |
| 2. Training | `python -m src.train` | ✅ artefak + meta tertulis |
| 3. Evaluasi | `python -m src.evaluate` | ✅ `reports/evaluation.md` |
| 4. API | `pytest tests -q` | ✅ 35 passed |
| 4. UI | render 1440px + 390px, submit contoh, state rendah/tinggi, error API | ✅ dicek manual (screenshot), tanpa test browser otomatis |
| 4. CI | workflow di GitHub Actions | ✅ pipeline (data, train, evaluate, drift, 35 test) dan docker smoke test hijau |
| 4. Serving | uvicorn + curl `/health`, `/predict` | ✅ 200, probabilitas wajar |
| 5. Docker | `docker build` → run → curl | ✅ build sukses, `/health` 200, `/predict` identik dengan lokal (selisih < 1e-6), `/predict/batch` 3 baris identik, input invalid → 422, healthcheck `healthy`, user `appuser` |
| 6. Drift | `python -m src.monitor.drift` | ✅ normal OK, drifted ALERT |
| 7. Dokumentasi | README + model card | ✅ |

## Keterbatasan

- 7.043 baris dari satu perusahaan; tidak otomatis berlaku untuk perusahaan lain.
- Split acak, bukan temporal, jadi pergeseran perilaku antar waktu belum diuji.
- Rasio biaya 5:1 adalah asumsi, bukan hasil pengukuran biaya nyata.
- Angka test tanpa confidence interval; selisih kecil antar model belum tentu signifikan.
- Model memprediksi niat churn, bukan efektivitas program retensi.
- Artefak dilatih pada train split saja; sebelum produksi latih ulang pada train+val.
- FastAPI tidak bisa di-deploy ke Freebuff Sites (Sites hanya Cloudflare Worker); untuk
  URL publik perlu host Python seperti Render/Railway/Fly.
- `logs/predictions.jsonl` tumbuh tanpa rotasi (~200 byte per prediksi). Ini **shortcut yang
  diterima secara sadar**, ditandai `shortcut:` di `app/main.py`: aman untuk volume sekarang,
  tapi rotasi perlu ditambahkan sebelum file melewati ~1 GB atau log dipakai untuk retraining.
- Build ulang image **setelah** UI ditambahkan belum dijalankan (Docker daemon mati saat
  verifikasi). Dockerfile tidak berubah dan `COPY app ./app` mencakup `app/static`, tetapi
  penyajian `/ui/` dari dalam container belum terbukti pada turn ini.

Detail model, metrik per segmen, dan batas pemakaian: [`reports/model_card.md`](reports/model_card.md).

## Audit bug yang ditemukan & diperbaiki

Setiap bug di bawah punya regression test atau check yang menjalankannya ulang.

| # | Bug | Dampak | Perbaikan |
|---|---|---|---|
| 1 | Image Docker tanpa `libgomp1` | Container langsung exit saat `import lightgbm` | `apt-get install libgomp1` di Dockerfile |
| 2 | `psi_numeric` dengan referensi konstan selalu menghasilkan 0 | Drift tidak terdeteksi sama sekali (false negative senyap) | Bin sempit di sekitar nilai konstan; diuji `test_psi_detects_shift_from_constant_reference` |
| 3 | `drift_report` melempar `KeyError` mentah bila kolom hilang | Pesan tidak informatif saat batch salah bentuk | `_require_columns` → `ValueError` yang menyebut kolom hilang |
| 4 | `TotalCharges` dibatasi 10.000 di API/pydantic | Pelanggan sah (72 bulan × 118,75) ditolak 422 | Batas diselaraskan ke 15.000 (domain 72 × 200) di schema **dan** validator |
| 5 | `evaluate.py` memakai `error_rows[0].keys()` | Crash bila tidak ada error pada threshold terpilih | Kolom error ditetapkan eksplisit + fallback teks |
| 6 | `clean()` mengakses kolom sebelum validasi | `KeyError` mentah bila data mentah cacat | Cek kolom wajib di awal `clean()` → `ValueError` jelas |
| 7 | Test log prediksi memakai `pytest.skip` | Kegagalan logging bisa lolos diam-diam | Test menulis 1 prediksi lalu memastikan jumlah baris log bertambah tepat 1 |
| 8 | `psi_numeric` mengembalikan 0,0 bila batch tidak punya nilai numerik | Batch dengan kolom kosong dilaporkan "stabil" (false negative senyap) | Kembalikan divergensi tak hingga → `ALERT`; diuji `test_psi_flags_batch_with_no_numeric_values` |
| 9 | `cost_curve`/`pick_threshold` tanpa test | Ambang bisnis bisa berubah tanpa satu test pun gagal | `tests/test_evaluate.py`: biaya minimum terpilih, `min_precision` benar-benar menyaring |
| 10 | `kaggle` tidak dideklarasikan di jalur setup | `run.ps1 setup` lalu `download` gagal di mesin baru | `requirements-dev.txt` (di luar image Docker), dipasang `run.ps1 setup`, pesan error menyebut perintah pastinya |
| 11 | `load_clean` tidak dipanggil siapa pun + dua helper SHA-256 | Dua salinan hashing harus dirawat bersamaan | `load_clean` dihapus; satu `sha256_file` dipakai `make_dataset` |
| 12 | Laporan test set menduplikasi ROC-AUC/PR-AUC/Brier di dua kolom threshold | Terbaca seolah threshold mengubah AUC | Tabel dipisah: metrik bebas threshold vs metrik bergantung threshold |

Total test setelah audit + UI: **35 passed** (sebelumnya 24, lalu 32).
