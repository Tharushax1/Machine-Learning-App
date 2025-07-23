# 🍷 Wine Quality Prediction App

This is a Streamlit web application that uses machine learning to predict wine quality (Good or Bad) based on physicochemical features such as acidity, sugar, alcohol, and type (red/white wine).

🔗 **GitHub Repository**: [Tharushax1/Machine-Learning-App](https://github.com/Tharushax1/Machine-Learning-App)

---

## 🚀 Features

- 🧠 Predict wine quality (binary: Good = 1, Bad = 0)
- 📊 Explore and filter the dataset interactively
- 📈 Visualize distributions, correlations, and trends
- 🔮 Make predictions with a trained ML model
- 📋 View model performance and feature importance

---

## 🧾 Dataset

- **Name**: [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- **File**: `winequalityN.csv`
- **Columns**:
  - `fixed acidity`
  - `volatile acidity`
  - `citric acid`
  - `residual sugar`
  - `chlorides`
  - `free sulfur dioxide`
  - `total sulfur dioxide`
  - `density`
  - `pH`
  - `sulphates`
  - `alcohol`
  - `type` (categorical: red/white)
  - `quality` (target: converted to 1 if ≥6, else 0)

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tharushax1/Machine-Learning-App.git
cd Machine-Learning-App

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
