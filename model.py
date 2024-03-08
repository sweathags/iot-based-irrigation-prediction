import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#CropType,CropDays,SoilMoisture,temperature,Humidity,Irrigation

# Load data from CSV
data = pd.read_csv('crop_irrigation_dataset.csv')

# Assuming 'data' is your DataFrame with columns 'CropType' to 'Irrigation'
# Split data into features (X) and target (y)
X = data.drop('Irrigation', axis=1)
y = data['Irrigation']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical columns
categorical_columns = ['CropType']
numerical_columns = ['CropDays', 'SoilMoisture', 'temperature', 'Humidity']

# Create pipeline for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', 'passthrough', numerical_columns)
    ])

# Combine preprocessing with classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = pipeline.score(X_test, y_test)
print("Accuracy:", accuracy)


from joblib import dump

# Assuming 'pipeline' is your trained model
model_filename = 'random_forest_model.joblib'
dump(pipeline, model_filename)
