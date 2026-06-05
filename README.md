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


Key Highlights

✅ End-to-End Deep Learning Pipeline

✅ Automated Dataset Download from Kaggle

✅ Data Preprocessing & Augmentation

✅ Transfer Learning using MobileNetV2

✅ Model Evaluation using Accuracy, Confusion Matrix, and Classification Report

✅ Export Model into Multiple Deployment Formats

✅ Single Image Inference

✅ Batch Inference

---

# Project Structure

```bash
.
├── Klasifikasi_Sampah.ipynb
├── waste_classification_raw/
├── waste_dataset_extracted/
├── waste_classification_binary/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── SavedModel/
│   ├── model.h5
│   └── model.keras
└── README.md
```

---

# Technology Stack

| Category                   | Technology          |
| -------------------------- | ------------------- |
| Language                   | Python              |
| Deep Learning              | TensorFlow          |
| Transfer Learning          | MobileNetV2         |
| Image Processing           | OpenCV, Pillow      |
| Data Science               | NumPy, Pandas       |
| Machine Learning Utilities | Scikit-Learn        |
| Visualization              | Matplotlib, Seaborn |
| Dataset Source             | Kaggle API          |
| Environment                | Jupyter Notebook    |

---

# Dataset

Dataset used:

**Recyclable and Household Waste Classification**

Source:

https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

The notebook automatically downloads the dataset through Kaggle API and restructures it into binary classes:

| Original Categories                | Target Class |
| ---------------------------------- | ------------ |
| Food Waste, Garden Waste, etc.     | Organic      |
| Plastic, Metal, Glass, Paper, etc. | Inorganic    |

---

# System Workflow

```text
Dataset Download
        ↓
Dataset Extraction
        ↓
Class Mapping
(Organic vs Inorganic)
        ↓
Train / Validation / Test Split
        ↓
Image Augmentation
        ↓
Transfer Learning (MobileNetV2)
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Export
        ↓
Inference
```

---

# Installation

## 1. Clone Repository

```bash
git clone <repository-url>
cd waste-classification
```

---

## 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / MacOS:

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install tensorflow
pip install numpy pandas
pip install matplotlib seaborn
pip install opencv-python
pip install pillow
pip install scikit-learn
pip install kaggle
pip install jupyter
```

Or:

```bash
pip install -r requirements.txt
```

---

# Kaggle API Configuration

The notebook downloads the dataset automatically.

## Step 1

Open:

```text
Kaggle
→ Account
→ Create New API Token
```

Download:

```text
kaggle.json
```

---

## Step 2

Place the file inside:

Windows:

```text
C:\Users\<username>\.kaggle\
```

Linux/Mac:

```text
~/.kaggle/
```

---

## Step 3

Verify installation:

```bash
kaggle datasets list
```

---

# Running the Project

## Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Klasifikasi_Sampah.ipynb
```

---

## Execute Cells Sequentially

Run all notebook cells from top to bottom.

The notebook performs:

### Phase 1 — Dataset Preparation

* Download dataset
* Extract dataset
* Explore folder structure

### Phase 2 — Data Engineering

* Reorganize classes
* Create train/validation/test split
* Prepare image generators

### Phase 3 — Deep Learning Training

* Load MobileNetV2 pretrained weights
* Add custom classification layers
* Calculate class weights
* Train model

### Phase 4 — Evaluation

* Accuracy Curve
* Loss Curve
* Test Accuracy
* Confusion Matrix
* Classification Report

### Phase 5 — Deployment Preparation

* Save model in multiple formats
* Perform inference
* Batch prediction testing

---

# Model Architecture

Base Model:

```text
MobileNetV2
(ImageNet Pretrained)
```

Additional Layers:

```text
Global Average Pooling
↓
Dense Layer
↓
Dropout
↓
Output Layer (Softmax)
```

Benefits:

* Faster training
* Reduced overfitting
* High accuracy with limited data
* Suitable for edge deployment

---

# Data Augmentation

Training images undergo augmentation:

```text
Rotation
Width Shift
Height Shift
Zoom
Horizontal Flip
```

Benefits:

* Improves generalization
* Reduces overfitting
* Increases robustness

---

# Evaluation Metrics

The model is evaluated using:

### Accuracy

Measures overall prediction correctness.

### Confusion Matrix

Shows class-wise prediction performance.

### Precision

Measures prediction reliability.

### Recall

Measures detection completeness.

### F1 Score

Balances precision and recall.

---

# Model Export

The notebook exports trained models into:

### TensorFlow SavedModel

```text
models/SavedModel/
```

### Keras Format

```text
models/model.keras
```

### H5 Format

```text
models/model.h5
```

These formats allow future deployment into:

* Web Applications
* REST APIs
* Mobile Applications
* Edge Devices
* Embedded AI Systems

---

# Example Inference

```python
prediction = predict_image(
    model,
    "sample.jpg",
    class_names
)

print(prediction)
```

Expected Output:

```text
Organic Waste
```

or

```text
Inorganic Waste
```

---

# Business Value

This project demonstrates competencies highly relevant to AI Engineer, Machine Learning Engineer, and Data Scientist roles:

* Data Collection & Preparation
* Data Engineering
* Computer Vision
* Transfer Learning
* Deep Learning Optimization
* Model Evaluation
* Model Deployment Preparation
* Reproducible Machine Learning Workflow

The solution can be adapted for:

* Smart Waste Sorting Systems
* Recycling Automation
* Environmental Monitoring
* Smart City Applications
* Sustainable Waste Management Platforms

---

# Author

**HANIDURA AYATULLOH**

Machine Learning & Artificial Intelligence Enthusiast

Focus Areas:

* Deep Learning
* Computer Vision
* Machine Learning Engineering
* Data Science
* AI Product Development
