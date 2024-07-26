import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='SyedAmeerHamza1', repo_name='mlflow_center_repo_demo', mlflow=True)

mlflow.set_registry_uri("https://dagshub.com/SyedAmeerHamza1/mlflow_center_repo_demo.mlflow")





iris= load_iris()

X= iris.data
Y= iris.target

x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=42)






max_depth= 15
n_estimators= 50


# Applying MLFlow

mlflow.set_experiment("iris-rfc-dt")


with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

    rf.fit(x_train, y_train)
    y_pred= rf.predict(x_test)

    cm= confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(rf, "DecisionTree")

    mlflow.log_artifact(__file__)

    accuracy_score= accuracy_score(y_test, y_pred)
    print(accuracy_score)

    mlflow.log_metric("accuracy_score", accuracy_score)

    mlflow.set_tag("Author", "Sohail")
    mlflow.set_tag("Model", "Decision Tree")

    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)

