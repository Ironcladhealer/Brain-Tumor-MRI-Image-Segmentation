# 🧠 Brain Tumor MRI Analysis (Classification & Segmentation)

This project focuses on **automated brain tumor detection from MRI images** using **Deep Learning with PyTorch**.
It has **two major components**:

1. **Image Classification** – Detecting tumor type (Glioma, Meningioma, Pituitary, or No Tumor).
2. **Image Segmentation** – Binary segmentation to localize tumor regions in MRI scans.

---

## 🚀 Features

* **Image Classification (CNNs / Transfer Learning)**

  * Trained on Kaggle’s **Brain Tumor MRI dataset**.
  * Supports 4 tumor classes: `Glioma`, `Meningioma`, `Pituitary`, `No Tumor`.
  * Evaluated using **accuracy, precision, recall, F1-score**.

* **Image Segmentation (U-Net in PyTorch)**

  * Binary segmentation (tumor vs. non-tumor).
  * Trained on Kaggle’s **Brain Tumor Segmentation dataset**.
  * Evaluated using **IoU (Intersection over Union), Dice Coefficient**.

* **End-to-End Workflow**

  * Data preprocessing & augmentation.
  * Model training with GPU acceleration.
  * Evaluation & visualization of predictions.

---

## 📂 Project Structure

```
Brain-Tumor-MRI/
│── classification/
│   ├── dataset/               # MRI images for classification
│   ├── train.py               # Training script (classification)
│   ├── model.py               # CNN / Transfer learning model
│   └── utils.py               # Helper functions
│
│── segmentation/
│   ├── dataset/               # MRI images + masks for segmentation
│   ├── train_unet.py          # U-Net training script
│   ├── unet_model.py          # U-Net architecture
│   └── utils.py               # Loss functions, metrics
│
│── notebooks/
│   ├── classification.ipynb   # Training & evaluation (classification)
│   ├── segmentation.ipynb     # Training & evaluation (segmentation)
│
│── README.md                  # Project documentation
│── requirements.txt            # Python dependencies
```

---

## 🗂️ Datasets

* **Classification Dataset**: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets)
* **Segmentation Dataset**: [Brain MRI Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets)

Download using Kaggle API:

```bash
# Example (Classification dataset)
kaggle datasets download -d <dataset-username/dataset-name> -p ./classification/dataset --unzip
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Brain-Tumor-MRI.git
   cd Brain-Tumor-MRI
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate Kaggle API:

   ```bash
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 🏋️ Training

### 🔹 Classification

```bash
cd classification
python train.py --epochs 30 --batch-size 32 --lr 1e-4
```

### 🔹 Segmentation

```bash
cd segmentation
python train_unet.py --epochs 50 --batch-size 16 --lr 1e-4
```

---

## 📊 Results

### Classification (sample results)

| Class        | Precision | Recall | F1-score |
| ------------ | --------- | ------ | -------- |
| Glioma       | 0.57      | 0.24   | 0.33     |
| Meningioma   | 1.00      | 0.00   | 0.01     |
| No Tumor     | 0.38      | 0.98   | 0.55     |
| Pituitary    | 0.44      | 0.20   | 0.27     |
| **Accuracy** | **0.40**  |        |          |

### Segmentation (sample metrics)

* IoU (Jaccard Index): `~0.70`
* Dice Score: `~0.80`

---

## 📈 Future Work

* Improve classification accuracy via:

  * Data augmentation, balanced sampling.
  * Pre-trained models (ResNet, EfficientNet).
  * Fine-tuning hyperparameters.
* Multi-class segmentation (separating tumor types).
* Integration into a **medical imaging pipeline** with visualization.

---

## 🤝 Contributing

Pull requests and discussions are welcome!

---

## 📜 License

This project is licensed under the **MIT License**.
