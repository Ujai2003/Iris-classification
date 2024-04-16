import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Resize and save Setosa image
img_setosa = Image.open('static/setosa.jpg')
img_setosa_resized = img_setosa.resize((150, 150))
img_setosa_resized.save('static/resized_setosa.jpg')

# Resize and save Virginica image
img_virginica = Image.open('static/virginica.jpg')
img_virginica_resized = img_virginica.resize((150, 150))
img_virginica_resized.save('static/resized_virginica.jpg')

# Resize and save Versicolor image
img_versicolor = Image.open('static/versicolor.jpg')
img_versicolor_resized = img_versicolor.resize((150, 150))
img_versicolor_resized.save('static/resized_versicolor.jpg')

# Load Iris dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Target'])

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=1, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=1, test_size=0.5)

def training_model():
    model = DecisionTreeClassifier(random_state=42)
    trained_model = model.fit(X_train, y_train)
    return trained_model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = round(accuracy_score(y, y_pred), 2)
    precision = round(precision_score(y, y_pred, average='weighted'), 2)
    recall = round(recall_score(y, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y, y_pred, average='weighted'), 2)
    return accuracy, precision, recall, f1

