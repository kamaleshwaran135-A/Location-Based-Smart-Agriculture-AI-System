import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Load dataset
data = pd.read_csv("agriculture_dataset.csv")

# Convert crop names to numbers
le = LabelEncoder()
data['Crop'] = le.fit_transform(data['Crop'])

# Features and target
X = data[['Farmers','Land_Area','Production']]
y = data['Crop']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# Models
svm = SVC()
rf = RandomForestClassifier()
lr = LogisticRegression()

# Train models
svm.fit(X_train,y_train)
rf.fit(X_train,y_train)
lr.fit(X_train,y_train)

# Predictions
svm_pred = svm.predict(X_test)
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)

# Accuracy
print("SVM Accuracy:",accuracy_score(y_test,svm_pred))
print("Random Forest Accuracy:",accuracy_score(y_test,rf_pred))
print("Linear Regression Accuracy:",accuracy_score(y_test,lr_pred))

# Confusion Matrix
cm = confusion_matrix(y_test,rf_pred)
sns.heatmap(cm,annot=True,cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Accuracy comparison graph
models = ['SVM','Random Forest','Linear']
acc = [
accuracy_score(y_test,svm_pred),
accuracy_score(y_test,rf_pred),
accuracy_score(y_test,lr_pred)
]

plt.bar(models,acc)
plt.title("Model Accuracy Comparison")
plt.show()

# Farmer percentage
farmer_percent = data.groupby('District')['Farmers'].sum()
farmer_percent.plot(kind='pie',autopct='%1.1f%%')
plt.title("Farmer Percentage by District")
plt.show()

# Highest land area district
high_land = data.sort_values(by='Land_Area',ascending=False)
print("\nDistrict with Highest Land Area:")
print(high_land[['District','Land_Area']].head())

# Crop percentage
crop_percent = data['Crop'].value_counts(normalize=True)*100
print("\nCrop Percentage:")
print(crop_percent)