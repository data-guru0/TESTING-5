import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import mlflow
import mlflow.sklearn

import mlflow
import mlflow.sklearn

class ModelTraining:
    def __init__(self):
        self.processed_data_path = "artifacts/processed"
        self.model_path = "artifacts/models"
        os.makedirs(self.model_path, exist_ok=True)
        self.model = DecisionTreeClassifier(criterion="gini", max_depth=30, random_state=42)
    
    def load_data(self):
        X_train = joblib.load(os.path.join(self.processed_data_path, "X_train.pkl"))
        X_test = joblib.load(os.path.join(self.processed_data_path, "X_test.pkl"))
        y_train = joblib.load(os.path.join(self.processed_data_path, "y_train.pkl"))
        y_test = joblib.load(os.path.join(self.processed_data_path, "y_test.pkl"))
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        # Start MLflow run
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("criterion", self.model.criterion)
            mlflow.log_param("max_depth", self.model.max_depth)
            mlflow.log_param("random_state", self.model.random_state)
            joblib.dump(self.model, os.path.join(self.model_path, "model.pkl"))
            print("Model training complete. Model saved in artifacts/models/")
    
    def evaluate_model(self, X_test, y_test):
        # Start MLflow run for evaluation
        with mlflow.start_run():
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f2 = f1_score(y_test, y_pred, average='binary', beta=2)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f2_score", f2)
            
            # Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_image_path = os.path.join(self.model_path, "confusion_matrix.png")
            plt.savefig(cm_image_path)
            mlflow.log_artifact(cm_image_path)  # Log the confusion matrix image
            
            plt.show()

if __name__ == "__main__":
    trainer = ModelTraining()
    X_train, X_test, y_train, y_test = trainer.load_data()
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model(X_test, y_test)
