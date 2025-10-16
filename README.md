# Ames Housing Price Predictor – Deployment

This repository hosts the **Streamlit web application** for the **Ames Housing Price Prediction** project — a machine learning app that predicts house prices based on key property features.

**Live App:** [https://house-price-predictor-sdia.onrender.com/](https://house-price-predictor-sdia.onrender.com)

---

## Overview

The app provides an interactive interface where users can:
- Input key house features (like **Overall Quality**, **Living Area**, **Garage Area**, etc.)
- Upload a CSV file for **batch predictions**
- View their entered details
- Instantly get a **predicted sale price** based on a trained **Ridge Regression model**

This deployment is powered by **Streamlit** and hosted on **Render**.  
All preprocessing artifacts (encoders, scalers, feature columns, and the trained model) are loaded from pre-saved joblib and pickle files.

---

## Model Information

The web app uses the **Ridge Regression model**, identified as the best performer in the main project repository after comparing:
- Linear Regression  
- Lasso, ElasticNet  
- Decision Tree, Random Forest  
- XGBoost  

### Model Summary
| Metric | Ridge Regression (Final Model) |
|:--------|:-------------------------------|
| RMSE | **0.1205** |
| R² | **0.9154** |
| MAE | **0.0801** |

The Ridge model balances simplicity and generalization, making it ideal for real-world deployment.

---

## Key Features

- **Batch Predictions:** Upload a CSV file to get predictions for multiple houses at once
- **Real-time Prediction:** Instantly predicts house prices based on user input
- **Feature Summary Table:** Displays all input values neatly in a dataframe
- **Model Transparency:** Uses pre-trained artifacts with explainable preprocessing
- **Lightweight UI:** Built with Streamlit, responsive and fast
- **Cloud Deployed:** Hosted on Render for public access  

---

## ⚙️ How to Run Locally

You can also run this app locally if you prefer:

```bash
# 1. Clone the repository
git clone https://github.com/Bloop15/House-Price-Predictor-Deployment.git
cd House-Price-Predictor-Deployment

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

Then open the link shown in your terminal (typically [http://localhost:8501](http://localhost:8501)).

---

## Deployment Details

- **Platform:** Render  
- **Framework:** Streamlit  
- **Python Version:** 3.11.13  

**Start Command:**
```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## Credits

- **Model & Pipeline:** [Ames Housing Price Prediction Project](https://github.com/Bloop15/House-Price-Predictor)  
- **Author:** Bloop15  
- **Deployed on:** Render Cloud

---

## Future Enhancements

- Add feature importance visualization for interpretability  
- Include comparison of multiple models in the app  
- Add API endpoint for external access to predictions
