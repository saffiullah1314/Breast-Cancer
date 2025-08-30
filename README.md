# Breast Cancer Prediction using ANN  

This project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras to predict whether a breast tumor is **benign** or **malignant** based on real-world medical data. The model is integrated with **Streamlit** for interactive predictions.  

---

## 📊 Dataset  
- Source: Breast Cancer Wisconsin (Diagnostic) dataset  
- Features: 30 numerical features (e.g., mean radius, texture, smoothness, etc.)  
- Target: Binary classification (0 = Benign, 1 = Malignant)  

---

## 🏗️ Model Architecture  
The ANN model was built using Keras Sequential API:  
- **Input Layer:** Flatten layer with 30 input features  
- **Hidden Layer 1:** Dense (30 neurons, ReLU activation)  
- **Hidden Layer 2:** Dense (15 neurons, ReLU activation)  
- **Output Layer:** Dense (2 neurons, Sigmoid activation)  

---

## 📈 Model Training Results  
The model was trained for **20 epochs** with strong performance:  

- **Training Accuracy:** Reached up to **99.5%**  
- **Validation Accuracy:** Improved from **95.6%** (epoch 1) to **97.8%** (epoch 20)  
- **Loss:** Reduced from **0.0891** to **0.0558**  
- No major overfitting observed; validation accuracy improved alongside training accuracy.  

This shows the ANN generalizes well and is highly reliable for breast cancer classification.  

---

## 🚀 Deployment with Streamlit  
The project includes a **Streamlit app** for easy interaction:  
- Users can input tumor feature values manually  
- The app predicts whether the tumor is **Benign, Malignant, or Error**  
- Results are displayed with color-coded sections for better visualization  

## Run the app locally with:  
```bash
streamlit run app.py

```
---
## 📂 Project Structure  

📂 Breast-Cancer
- ├── app.py              # Streamlit web app  
- ├── model.h5            # Trained ANN model  
- ├── requirements.txt    # Dependencies  
- ├── README.md           # Project Documentation  
---
## ⚡ How to Use  

- **Clone the repository:**  
```bash
git clone https://github.com/saffiullah1314/Breast-Cancer.git  
cd Breast-Cancer
```
---

## Install dependencies:
```bash
pip install -r requirements.txt

```
## Run the Streamlit app:
```bash
streamlit run app.py
```
## ✅ Results

- The trained ANN achieves ***97-99%*** accuracy on the test data, making it an effective tool for assisting in breast cancer diagnosis.

