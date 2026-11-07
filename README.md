# Online Shopper Purchase Prediction

## 1. Project Overview

In this project, I have used different machine learning classification algorithms to predict whether a visitor will make a purchase on an online shopping website.

I have trained and compared five classification models using the same dataset. I have also developed a Streamlit application where the user can upload test data, select a model, and check its performance.

**GitHub Repository:** https://github.com/pranalipatil-dev/online-shopper-purchase-prediction

**Streamlit Application:** https://online-shopper-purchase-prediction-2026aj05795.streamlit.app/

## 2. Dataset Information

**Dataset Name:** Online Shoppers Purchasing Intention Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset Link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains 12,330 records and 17 input features.

The target column is `Revenue`, which indicates whether a purchase was made during the shopping session.

- `True (1)` - Purchase was made
- `False (0)` - No purchase was made

The dataset contains information such as the number of pages visited, time spent on pages, bounce rates, exit rates, visitor type, and other session details.

This is a binary classification problem.

The dataset is loaded directly from the UCI Machine Learning Repository using its URL in the Jupyter Notebook. Therefore, the original dataset CSV does not need to be downloaded manually.

## 3. Data Preprocessing

Before training the models, I performed the following preprocessing steps:

1. Checked the dataset and removed duplicate records.
2. Separated the input features and target column.
3. Split the dataset into 80% training data and 20% testing data.
4. Used stratified splitting to maintain the class distribution.
5. Applied StandardScaler to numerical features.
6. Applied OneHotEncoder to categorical features.
7. Used ColumnTransformer and Pipeline to apply preprocessing consistently.

After removing 125 duplicate records, 12,205 records remained. These were split into 9,764 training records and 2,441 testing records.

I fitted the preprocessing steps only on the training data to avoid data leakage.

The same preprocessing pipeline is also used when making predictions through the Streamlit application.

## 4. Machine Learning Models Used

I implemented the following five classification models:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

I trained all five models on the same training dataset and evaluated them using the same test dataset so that their performance could be compared.

## 5. Evaluation Metrics

I used the following six metrics to evaluate the models:

- **Accuracy:** Percentage of correctly predicted records.
- **AUC:** Measures how well the model separates the two classes across different thresholds.
- **Precision:** Out of all predicted purchases, how many were actual purchases.
- **Recall:** Out of all actual purchases, how many were correctly identified.
- **F1 Score:** Balances precision and recall.
- **MCC:** Measures overall classification quality using all four confusion matrix outcomes.

Since the dataset has more non-purchase records than purchase records, I considered F1 Score and MCC along with accuracy while comparing the models.

## 6. Model Performance Comparison

The results obtained on the test dataset are:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8869 | 0.8996 | 0.7524 | 0.4136 | 0.5338 | 0.5032 |
| Decision Tree | 0.8566 | 0.7380 | 0.5400 | 0.5654 | 0.5524 | 0.4673 |
| KNN | 0.8734 | 0.7958 | 0.6622 | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.2773 | 0.7502 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8955 | 0.8729 | 0.7361 | 0.5183 | 0.6083 | 0.5614 |

## 7. Observations

### Logistic Regression

Logistic Regression gave an accuracy of 88.69% and the highest AUC of 0.8996 among all five models.

Its precision was also the highest at 0.7524. However, recall was only 0.4136, which means it was not able to identify many of the actual purchase cases.

Overall, the model performed well but could be improved in terms of recall.

### Decision Tree

Decision Tree achieved an accuracy of 85.66% and an F1 Score of 0.5524.

Its precision and recall were quite close to each other, indicating a relatively balanced performance for both measures.

However, its AUC was 0.7380, which was the lowest among the five models.

### K-Nearest Neighbors (KNN)

KNN achieved an accuracy of 87.34%.

Its precision was 0.6622, but recall was comparatively low at 0.3901. This means it missed many actual purchase cases.

Its F1 Score was 0.4909, which was lower than Logistic Regression, Decision Tree, and SVM.

### Gaussian Naive Bayes

Naive Bayes gave very different results compared to the other models.

It achieved the highest recall of 0.9791, which means it identified almost all actual purchase cases.

However, its precision was only 0.1756, indicating that many non-purchase cases were incorrectly predicted as purchases.

Because of this, its overall accuracy and F1 Score were quite low.

### Support Vector Machine (SVM)

SVM achieved the highest accuracy of 89.55% among the five models.

It also gave the highest F1 Score of 0.6083 and MCC of 0.5614.

Its precision was 0.7361 and recall was 0.5183. Although recall was not the highest, the model gave the best overall performance based on the selected evaluation criteria.

## 8. Best Performing Model

Based on the results, I selected **Support Vector Machine (SVM)** as the best-performing model.

The main reasons are:

- It achieved the highest accuracy of 89.55%.
- It had the highest F1 Score of 0.6083.
- It also achieved the highest MCC of 0.5614.

Logistic Regression performed better in terms of AUC and precision, but SVM provided better overall results based on F1 Score, MCC, and accuracy.

## 9. Streamlit Application

I created a Streamlit application to test and compare the trained models.

**Live Application:** https://online-shopper-purchase-prediction-2026aj05795.streamlit.app/

The application provides the following features:

1. Upload a CSV file containing test data.
2. Select one of the five classification models using a dropdown.
3. View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
4. View the confusion matrix.
5. View the classification report.
6. Compare the performance of all five models in one table.
7. View actual and predicted values.

I also added basic UI customizations such as a project title, sidebar information, metric cards, and a confusion matrix heatmap.

The trained models are saved using Joblib. The Streamlit application loads these saved models and makes predictions without retraining them.

## 10. Project Files

| File | Description |
|---|---|
| `2026AJ05795.ipynb` | Jupyter Notebook containing dataset loading, preprocessing, model training, and evaluation |
| `app.py` | Streamlit application code |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project details, results, and instructions |
| `test_data.csv` | Test dataset used for evaluation |
| `model_comparison.csv` | Comparison of all five models |
| `model/` | Folder containing the five saved model pipelines |

The original dataset is loaded from the UCI repository URL in the notebook, so a separate copy of the dataset is not required in the repository.

## 11. How to Run the Project

### Run the Jupyter Notebook

1. Open `2026AJ05795.ipynb` in Jupyter Notebook.
2. Install the required libraries if they are not already available.
3. Run the notebook cells in order.
4. The notebook loads the dataset from UCI, performs preprocessing, trains the five models, and displays the evaluation results.
5. The trained models and test data are saved for use in the Streamlit application.

An internet connection is required when loading the original dataset from UCI.

### Run the Streamlit Application

**Step 1:** Install the required libraries.

```bash
pip install -r requirements.txt
```

**Step 2:** Run the Streamlit application.

```bash
streamlit run app.py
```

**Step 3:** Open the Streamlit application in the browser.

**Step 4:** Upload `test_data.csv` and select the model to view its performance.

The application uses the already trained and saved models, so there is no need to train them again.

## 12. Conclusion

In this assignment, I implemented and compared five classification algorithms for predicting online shopping purchase intention.

From the results, I observed that accuracy alone is not sufficient to compare classification models, especially when the dataset has an unequal class distribution.

SVM performed the best overall based on F1 Score, MCC, and accuracy. Logistic Regression achieved the highest AUC and precision, while Naive Bayes achieved the highest recall but had very low precision.

This comparison helped me understand how different classification algorithms perform on the same dataset and why multiple evaluation metrics are important.

## Reference

UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
