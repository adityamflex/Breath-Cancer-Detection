# Breath-Cancer-Detection
# 🩺 Breast Cancer Detection using Machine Learning

This project is a machine learning-based solution for **early detection of breast cancer** using a dataset of clinical and diagnostic features. Built with **Python**, **Pandas**, and **scikit-learn**, it helps predict whether a tumor is **benign** or **malignant**.

---

## 🎯 Objective

Early detection of breast cancer can save lives. The goal of this project is to use machine learning to classify tumors based on input features derived from medical images and tests.

---

## 🧰 Technologies Used

- 🐍 **Python 3**
- 📊 **Pandas** – Data handling and analysis
- 📈 **Scikit-learn** – Machine learning models
- 📉 **Matplotlib / Seaborn** – Data visualization
- 🧪 **Jupyter Notebook** – For experimentation and development

---

## 📂 Dataset

We used the **Breast Cancer Wisconsin Diagnostic Dataset**, which includes:

- 569 samples
- 30 numerical features (e.g. radius, texture, smoothness)
- Target variable: `diagnosis` (`M` = malignant, `B` = benign)

You can download it from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) or use it via `sklearn.datasets`.

---

## 🧠 ML Models Applied

We experimented with several models, including:

- ✅ Logistic Regression
- 🌲 Random Forest
- 📈 Support Vector Machine (SVM)
- 🧠 K-Nearest Neighbors (KNN)

### Model Evaluation

We used:
- **Train-test split**
- **Accuracy, Precision, Recall, F1-score**
- **Confusion Matrix**
- **ROC Curve**

---

## 📊 Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 96.5%    |
| Random Forest       | 97.3%    |
| SVM (RBF Kernel)    | 98.2%    |
| KNN (k=5)           | 96.1%    |

> SVM with RBF kernel gave the best results in our tests.

---

## 🛠 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/breast-cancer-detection.git
cd breast-cancer-detection
