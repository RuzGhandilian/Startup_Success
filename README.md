# Startup Success 

## Project Description 

The repository provides dataset describing 472 startups and their founders as well as information on the company status described a success or a failure. The main data is provided in `data.csv` file. `dictionary.csv` provides description for some of the features provided in the dataset.

*The objective of the analysis is to identify and explain driving factors behind startup **success** or **failure**.* 

For running the pipeline, use python to run `__main__.py` script. 

The repository also contains `Preprocessing.py`, `Model.py`, `Pipeline.py` python scripts and Jupyter Notebook `EDA.ipynb` that showcase EDA of the model.

## `Preprocessing.py` Module

- **Missing Values Handling:** Replaces 'No Info' with NaN and fills missing values in categorical columns with the mode.
- **Numerical Conversion:** Converts object columns to numeric where possible.
- **Standardization:** Standardizes numerical variables using StandardScaler.
- **One-Hot Encoding:** Converts categorical variables into binary vectors using OneHotEncoder.

## `Model.py` Module

This module defines a `Model` class for training, evaluating, and displaying results of a Balanced Random Forest Classifier. I selected the Balanced Random Forest Classifier for its ability to handle the dataset's imbalanced classes and high-dimensional features effectively. Despite testing various models, this one consistently outperformed the others. Hyperparameter tuning was conducted to optimize its performance further.

Here is the best classification report reached using BalancedRandomForestClassifier which handled data imbalance we have
```
Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.85      0.86        39
           1       0.93      0.94      0.93        79

    accuracy                           0.91       118
   macro avg       0.90      0.89      0.89       118
weighted avg       0.91      0.91      0.91       118

AUC-ROC: 0.9711132749107433

Accuracy: 0.907
Precision: 0.925
Recall: 0.937
F1 Score: 0.931

```

Finally, the goal of the task was to understand and explain the reason behind success or failure of startup companies. The `feature_importances.png` plot displays the most important features identified by our model for predicting startup success. These features provide valuable insights into the factors influencing the outcome of startup ventures, helping stakeholders understand key drivers of success or failure.