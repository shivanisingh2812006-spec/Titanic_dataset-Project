import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

X = df[["Pclass", "Gender", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Titanic Survival Prediction")
print("----------------------------")
print("Training data:", len(X_train))
print("Testing data:", len(X_test))
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3
)

plt.title("Titanic Survival Decision Tree")
plt.show()

sample = pd.DataFrame({
    "Pclass": [3],
    "Gender": [0],
    "Age": [25],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [10]
})

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nSample Prediction: Passenger may survive")
else:
    print("\nSample Prediction: Passenger may not survive")