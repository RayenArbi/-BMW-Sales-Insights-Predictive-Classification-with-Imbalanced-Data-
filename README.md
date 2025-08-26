# 🚗 BMW Sales Insights — EDA & Classification under Data Imbalance

This project focuses on analyzing **BMW car sales data** to extract key insights and build a **classification model** to predict sales categories, while handling the challenge of **imbalanced datasets**.

## 📂 Project Structure
```
│── README.md
│── BMW_Sales_Insights_EDA_and_Classification_under_Data_Imbalance.ipynb   # Full notebook
│── requirements.txt                                                       # Dependencies
│── best_model.pkl                                                         # Saved ML model
│── main.py                                                                # Run predictions from CLI
│── images/                                                                # Key visuals
    │── class_distribution.png
    │── feature_importance.png
    │── confusion_matrix.png
```

## 📊 Exploratory Data Analysis (EDA)
The notebook provides detailed **EDA** to uncover trends and patterns, including:
- Distribution of BMW car models, fuel types, and regions.
- Price and mileage variations across categories.
- Correlation analysis between sales features.
- Visual insights into **imbalanced target classes**.

## 🤖 Machine Learning Modeling
We built and evaluated classification models with techniques to address **data imbalance**:
- Oversampling & undersampling approaches (SMOTE, Random Undersampling).
- Algorithms tested: Logistic Regression, Random Forest, Gradient Boosting, etc.
- Metrics beyond accuracy: **Precision, Recall, F1-score, ROC-AUC**.

The **best-performing model** is saved as `best_model.pkl`.

## ⚙️ Usage

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run predictions from CLI
```bash
python main.py --input new_data.csv --output predictions.csv
```

### 3️⃣ Explore visuals
Check the `images/` folder for plots such as:
- Class distribution
- Feature importance
- Confusion matrix

## 📈 Results
- Key factors influencing BMW sales were identified.
- Addressing imbalance significantly improved recall for minority classes.
- Final model balances **performance + interpretability**.

## 📜 License
MIT License
