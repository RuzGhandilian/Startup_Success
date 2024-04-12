from Preprocessing import Preprocessing
from Model import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Pipeline:
    def __init__(self, data_path):
        # Initialize attributes
        self.df = pd.read_csv(data_path, encoding='iso-8859-1')
        self.cols_when_model_builds = None
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessing = Preprocessing()

    def run_pipeline(self):
        # Split data into features (X) and target variable (y)
        X = self.df.drop('Dependent-Company Status', axis=1)
        y = self.df['Dependent-Company Status']

        # Convert target variable to binary (0 for 'Failed', 1 for 'Success')
        y = [0 if element == 'Failed' else 1 for element in y]

        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        # Perform Preprocessing on train and test sets separately
        self.preprocessing.fit(self.X_train)
        self.X_train = self.preprocessing.transform(self.X_train)
        self.X_test = self.preprocessing.transform(self.X_test)

        # Train Model
        model = Model(self.X_train, self.X_test, self.y_train, self.y_test)
        model.train_model()

        # Get column names when the model builds
        cols_when_model_builds = self.X_train.columns[np.argsort(model.feature_importances_)[::-1]]

        # Select columns in test set as per the order of importance learned during training
        self.X_test = self.X_train[cols_when_model_builds]

        # Evaluate Model
        model.evaluate_model()
        model.display_results()
        model.plot_roc_curve()

        # Evaluate Feature Importancies
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        top_features = importance_df.head(10)

        # Visualize the feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top Features Contributing to Success or Failure')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
