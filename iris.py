import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns





iris= load_iris()

X= iris.data
Y= iris.target

x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.2, random_state=42)






max_depth= 15


# Applying MLFlow

mlflow.set_experiment("iris-dt")


with mlflow.start_run(run_name="new run5", experiment_id='435664341241085501'):
    dt= DecisionTreeClassifier(max_depth=max_depth)

    dt.fit(x_train, y_train)
    y_pred= dt.predict(x_test)

    cm= confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    plt.savefig("Confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")
    mlflow.sklearn.log_model(dt, "DecisionTree")

    mlflow.log_artifact(__file__)

    accuracy_score= accuracy_score(y_test, y_pred)
    print(accuracy_score)

    mlflow.log_metric("accuracy_score", accuracy_score)

    mlflow.set_tag("Author", "Sohail")
    mlflow.set_tag("Model", "Decision Tree")

    mlflow.log_param("max_depth", max_depth)

