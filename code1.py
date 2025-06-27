import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ✅ Load iris.data with correct column names
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("iris.data", header=None, names=columns)

print("First 5 rows:")
print(df.head())

# 🧠 Basic EDA
print("\nInfo:")
print(df.info())
print("\nSummary:")
print(df.describe())

# 📊 Pairplot
sns.pairplot(df, hue='species')
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()

# ✂️ Train/test split
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ✅ Predict & evaluate
y_pred = knn.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")

# 📉 Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 🎯 Optional scatter plot
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title("Sepal Length vs Width")
plt.show()
