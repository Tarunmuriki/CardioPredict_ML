import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_theme(style="whitegrid")

DATA_PATH = os.path.join("datasets", "heart_disease_dataset.csv")
RESULTS_DIR = "results"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Please download the heart disease dataset "
            "and save it to the datasets folder as heart_disease_dataset.csv"
        )
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded:", df.shape)
    print(df.head())
    print(df.info())
    return df


def save_figure(fig, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved figure: {path}")
    plt.close(fig)


def plot_decision_tree(model, feature_names):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    ax.set_title("Decision Tree Visualization")
    save_figure(fig, "classification_tree.png")


def plot_feature_importance(feature_importance):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_importance,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    save_figure(fig, "important_features.png")


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Random Forest Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_figure(fig, "model_confusion_matrix.png")


def main():
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

    plot_decision_tree(dt_model, X.columns)

    deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    deep_tree.fit(X_train, y_train)
    print("Deep Decision Tree Train Accuracy:", deep_tree.score(X_train, y_train))
    print("Deep Decision Tree Test Accuracy:", deep_tree.score(X_test, y_test))

    shallow_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    shallow_tree.fit(X_train, y_train)
    print("Shallow Decision Tree Train Accuracy:", shallow_tree.score(X_train, y_train))
    print("Shallow Decision Tree Test Accuracy:", shallow_tree.score(X_test, y_test))

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    print("Decision Tree Accuracy:", dt_accuracy)
    print("Random Forest Accuracy:", rf_accuracy)

    importance = rf_model.feature_importances_
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": importance}
    ).sort_values(by="Importance", ascending=False)
    print(feature_importance)
    plot_feature_importance(feature_importance)

    cm = confusion_matrix(y_test, y_pred_rf)
    plot_confusion_matrix(cm)

    print("Classification Report:\n", classification_report(y_test, y_pred_rf))

    scores = cross_val_score(rf_model, X, y, cv=5)
    print("Cross Validation Scores:", np.round(scores, 4))
    print("Average Accuracy:", np.round(scores.mean(), 4))


if __name__ == "__main__":
    main()
