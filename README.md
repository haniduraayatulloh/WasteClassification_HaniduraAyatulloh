# ♻️ Smart Waste Classifier: Deteksi Jenis Sampah Berbasis MobileNetV2

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Framework](https://img.shields.io/badge/Method-Transfer%20Learning-009688?style=for-the-badge)
![Accuracy Target](https://img.shields.io/badge/Accuracy-≥%2085%25-brightgreen?style=for-the-badge)

Sistem deteksi dan klasifikasi jenis sampah otomatis berbasis *Deep Learning* yang dirancang untuk memisahkan sampah ke dalam dua kategori utama secara akurat: **Organic** (Organik) dan **Inorganic** (Anorganik). Proyek ini dikembangkan menggunakan arsitektur **MobileNetV2** melalui metode *Transfer Learning* guna menghasilkan model yang ringan, cepat, namun tetap presisi tinggi.

> 📝 **Studi Kasus Seleksi Kerja:** Proyek ini disusun secara komprehensif untuk memenuhi seluruh kriteria dan spesifikasi teknis seleksi berkas *Machine Learning Engineer*.

---

## 🎯 Kriteria Seleksi & Pemenuhan Sistem

| Ketentuan Seleksi | Status Pemenuhan Proyek | Detail Teknis |
| :--- | :---: | :--- |
| **Dataset minimal 150 data** | ✅ **Terpenuhi** | Menggunakan total **7.500 gambar** (Recyclable and Household Waste). |
| **Proses training ML/DL** | ✅ **Terpenuhi** | Menggunakan ekosistem TensorFlow & Keras pipeline. |
| **Akurasi model minimal 80%** | ✅ **Terpenuhi** | Model berhasil menembus target akurasi minimum seleksi. |
| **Klasifikasi berdasarkan jenis** | ✅ **Terpenuhi** | Output biner: `Organic` dan `Inorganic`. |
| **Dokumentasi Dataset** | ✅ **Terpenuhi** | Terlampir pada bagian [Dokumentasi Dataset](#-dokumentasi-dataset). |
| **Penjelasan Metode/Model** | ✅ **Terpenuhi** | Terlampir pada bagian [Metode & Arsitektur](#-metode--arsitektur-model). |
| **Hasil Evaluasi Model** | ✅ **Terpenuhi** | Menyertakan loss, akurasi final, dan kurva pelatihan. |

---

## 📊 Dokumentasi Dataset

Sistem ini dilatih menggunakan dataset publik berskala besar untuk menjamin performa generalisasi model yang baik.

* **Sumber Dataset:** [Kaggle - Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)
* **Volume Data:** 7.500 Gambar Citra Sampah.
* **Pembagian Data (Split Ratio):** Dipecah secara acak menggunakan generator data menjadi proporsi **Training, Validation, dan Testing**.
* **Pemetaan Kelas (Label Mapping):**
  * **Organic:** Sisa makanan (*food waste*), bubuk kopi, kulit telur, kantong teh, dan sampah organik dapur lainnya.
  * **Inorganic:** Botol/kantong plastik, kardus, kaleng aluminium, kaca, kertas, styrofoam, hingga limbah tekstil.

---

## 🧠 Metode & Arsitektur Model

Proyek ini mengimplementasikan teknik **Transfer Learning** dengan memanfaatkan *pre-trained model* **MobileNetV2** yang sebelumnya telah dilatih pada miliaran gambar di dataset ImageNet.

### Alur Kerja (Pipeline) Sistem:
1. **Data Augmentation:** Untuk meningkatkan variasi data dan mencegah *overfitting*, diterapkan manipulasi gambar *real-time* via `ImageDataGenerator` berupa rotasi (`rotation_range`), pergeseran (`width/height shift`), perbesaran (`zoom_range`), dan pembalikan horizontal (`horizontal_flip`).
2. **Feature Extractor:** Menggunakan struktur dasar MobileNetV2 dengan bobot ImageNet. Beberapa *layer* teratas di-*freeze* untuk mempertahankan fitur visual dasar (garis, bentuk, tekstur).
3. **Custom Classifier Head:** Menambahkan arsitektur khusus di atas base model:
   * `GlobalAveragePooling2D()` untuk mereduksi dimensi matriks.
   * `Dropout(0.5)` sebagai regulasi jaringan agar tidak terjadi *overfitting*.
   * `Dense Layer` dengan aktivasi `Softmax` / `Sigmoid` untuk kalkulasi probabilitas kelas akhir.
4. **Optimasi:** Menggunakan optimizer `Adam` dengan *learning rate* adaptif dikombinasikan dengan callback `EarlyStopping` untuk menghentikan proses *training* secara otomatis ketika akurasi validasi telah optimal.

---

## 📈 Hasil Evaluasi Model

Proses evaluasi dilakukan menggunakan data uji (*Test Set*) yang sepenuhnya baru dan terisolasi dari proses latihan.

* **Metrik Akhir:** Nilai akurasi final berhasil melampaui batas minimal kelulusan (≥ 85%).
* **Visualisasi Performa:** Kode program secara otomatis memuat plot kurva **Loss vs Epoch** dan **Accuracy vs Epoch** untuk mendeteksi kestabilan proses *training* model secara visual.

---

## 📁 Struktur Repositori

```text
├── Klasifikasi_Sampah.ipynb     # Jupyter Notebook (Source Code Utama & Proses ML)
├── README.md                     # Dokumentasi Proyek (File ini)
└── models/                       # Direktori Penyimpanan Output Model
    ├── model_waste.h5            # Format Standar Keras
    └── model_waste.tflite        # Format Terkompresi untuk Mobile/Edge Device
