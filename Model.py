import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report, roc_auc_score, roc_curve
from imblearn.ensemble import BalancedRandomForestClassifier


class Model:
    def __init__(self, X_train, X_test, y_train, y_test):
        # Initialize attributes
        self.roc_auc = None
        self.classification_report_str = None
        self.conf_matrix = None
        self.f1 = None
        self.y_pred = None
        self.recall = None
        self.precision = None
        self.feature_importances_ = None
        self.accuracy = None
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        # Train Balanced Random Forest Classifier
        self.model = BalancedRandomForestClassifier(n_estimators=800, random_state=42, sampling_strategy="all",
                                                    replacement=True, bootstrap=True)
        self.model.fit(self.X_train, self.y_train)
        self.feature_importances_ = self.model.feature_importances_

    def evaluate_model(self):
        # Evaluate the model
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        threshold = 0.45
        self.y_pred = (y_pred_prob > threshold).astype(int)

        # Calculate evaluation metrics
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.precision = precision_score(self.y_test, self.y_pred)
        self.recall = recall_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        self.classification_report_str = classification_report(self.y_test, self.y_pred)
        self.roc_auc = roc_auc_score(self.y_test, y_pred_prob)

    def display_results(self):
        # Display evaluation results
        print(f"Accuracy: {self.accuracy:.3f}")
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall: {self.recall:.3f}")
        print(f"F1 Score: {self.f1:.3f}")
        print("\nConfusion Matrix:")
        print(self.conf_matrix)
        print("\nClassification Report:")
        print(self.classification_report_str)
        print(f'AUC-ROC: {self.roc_auc:.3f}')

    def plot_roc_curve(self):
        # Plot ROC Curve
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
        roc_auc = roc_auc_score(self.y_test, y_pred_prob)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


