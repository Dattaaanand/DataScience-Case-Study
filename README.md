# Heart Attack Risk Prediction - Data Science Case Study

## Project Overview
This case study demonstrates the end-to-end lifecycle of a Data Science project. The goal is to take a raw dataset, simulate real-world data quality issues (missing values, noise), and apply robust **Data Engineering** and **Machine Learning** techniques to build a reliable predictive model.

**Key Objectives:**
- **Data Simulation:** Injecting synthetic missing values to mimic real-world "messy" data.
- **Data Processing:** Implementing a modular pipeline for Imputation, Outlier Handling, Encoding, and Scaling.
- **Data Integrity:** Using a "Backpack Strategy" to ensure Patient IDs are preserved without leaking into the model.
- **Model Optimization:** Tuning the decision threshold to maximize **Recall** (saving lives) over raw Accuracy.

---

## Instructions

### Initialization
In your terminal, run `pip install -r requirements.txt`

---

### Order of Execution

1. Data Initialiation
2. Imputation
3. Outliers
4. Encoding
5. Feature Scaling
6. Training the model

---

### Steps to Execute
###### MAKE SURE YOU HAVE INSTALLED THE REQUIREMENTS/DEPENDENCIES

To execute, run `python3 main.py` or `python main.py`