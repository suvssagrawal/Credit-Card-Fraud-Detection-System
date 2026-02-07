# 🔍 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using multiple classification algorithms and SMOTE for handling class imbalance.

**Author:** Suvan Agrawal

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Transactions** | 284,807 |
| **Fraudulent Transactions** | 492 (0.172%) |
| **Genuine Transactions** | 284,315 (99.828%) |
| **Features** | 31 (V1-V28 PCA components + Amount + Time) |
| **Class Imbalance Ratio** | 578:1 |

---

## 🔄 Data Preprocessing

### Feature Scaling
- Applied **StandardScaler** to Amount and Time features
- PCA-transformed features (V1-V28) already normalized

### Train-Test Split
- **Split Ratio:** 80-20 with stratification
- **Training Set:** 199,644 samples (199,280 genuine, 364 fraud)
- **Test Set:** 49,911 samples (49,820 genuine, 91 fraud)

### Handling Class Imbalance
- **Technique:** SMOTE (Synthetic Minority Over-sampling)
- **Before SMOTE:** 547:1 ratio
- **After SMOTE:** 1:1 ratio (199,280 samples each class)
- **Applied only to training data** to prevent data leakage

---

## 📈 Exploratory Data Analysis

- Class distribution visualization
- Transaction amount patterns (fraud vs genuine)
- Time-based fraud behavior analysis
- Feature correlation heatmap

---

## 🤖 Model Results

### Logistic Regression ✅

**Confusion Matrix:**
```
                    Predicted Genuine    Predicted Fraud
Actual Genuine           48,830              990
Actual Fraud                14               77
```

**Metrics:**
- **Recall:** 84.6% (77/91 frauds caught) 🏆
- **Precision:** 7.2%
- **False Negatives:** 14 🏆
- **False Positives:** 990

---

### Decision Tree ✅

**Confusion Matrix:**
```
                    Predicted Genuine    Predicted Fraud
Actual Genuine           49,249              571
Actual Fraud                18               73
```

**Metrics:**
- **Recall:** 80.2% (73/91 frauds caught)
- **Precision:** 11.3%
- **False Negatives:** 18
- **False Positives:** 571

---

### Random Forest ✅

**Confusion Matrix:**
```
                    Predicted Genuine    Predicted Fraud
Actual Genuine           49,801              9
Actual Fraud                20               71
```

**Metrics:**
- **Recall:** 78.0% (71/91 frauds caught)
- **Precision:** 88.8% 🏆
- **False Negatives:** 20
- **False Positives:** 9 🏆

---

### Support Vector Machine ⏳

**Status:** Not trained yet

---

## 📊 Model Comparison

| Metric | Logistic Reg | Decision Tree | Random Forest | Best Model |
|--------|--------------|---------------|---------------|------------|
| **Recall (Frauds Caught)** | **84.6%** (77/91) | 80.2% (73/91) | 78.0% (71/91) | 🏆 Logistic Reg |
| **Precision** | 7.2% | 11.3% | **88.8%** | 🏆 Random Forest |
| **False Positives** | 990 | 571 | **9** | 🏆 Random Forest |
| **False Negatives** | **14** | 18 | 20 | 🏆 Logistic Reg |

### 🔍 Key Insights

- **Logistic Regression** catches the most frauds (84.6% recall) but generates many false alarms (990 false positives)
- **Random Forest** has excellent precision (88.8%) with minimal false positives (only 9!) but misses more frauds
- **Decision Tree** offers a middle ground between the two approaches
- **Trade-off:** High recall vs High precision - depends on business requirements

---

## 💾 Model Deployment

### Selected Model: Random Forest
**Reason:** Best balance of precision (88.8%) and minimal false positives (9), suitable for production deployment

### Saving the Model
```python
import pickle

# Save Random Forest model
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
```

### Loading and Using the Model
```python
# Load the saved model
with open('fraud_detection_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Make predictions on new transactions
sample = X_test.iloc[0:1]
prediction = loaded_model.predict(sample)
probability = loaded_model.predict_proba(sample)[:, 1][0]

print(f"Prediction: {'FRAUD' if prediction[0]==1 else 'GENUINE'}")
print(f"Confidence: {probability*100:.2f}%")
```

**Model File:** `fraud_detection_model.pkl`

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn, pickle
- **ML Algorithms:** Logistic Regression, Decision Tree, Random Forest, SVM
- **Resampling:** SMOTE

---

## 🚀 Usage

1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook or Python script
4. Load the saved model using pickle
5. Make predictions on new transaction data

---

## 📋 Project Checklist

- [x] Load and explore dataset
- [x] Perform EDA and visualizations
- [x] Apply feature scaling
- [x] Handle class imbalance with SMOTE
- [x] Train Logistic Regression
- [x] Train Decision Tree
- [x] Train Random Forest
- [x] Evaluate and compare models
- [x] Save final model
- [ ] Train SVM model
- [ ] Hyperparameter tuning
- [ ] Deploy as API (optional)

---

## 📝 Notes

- PCA features (V1-V28) used for confidentiality
- In fraud detection, the choice between models depends on business priorities:
  - **Choose Logistic Regression** if catching every fraud is critical
  - **Choose Random Forest** if minimizing false alarms is priority
- SMOTE applied only on training data to avoid leakage
- Random Forest selected for deployment due to balanced performance

---

## 👨‍💻 Author

**Suvan Agrawal**

---

