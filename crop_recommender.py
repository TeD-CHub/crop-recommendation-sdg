import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

def run_crop_recommendation_project():
    """
    Main function to load data, train a crop recommendation model, and evaluate it.
    """

    # --- 1. Load Data ---
    # We will use a popular Crop Recommendation dataset from Kaggle.
    # This dataset is hosted online, so the script is runnable anywhere (like Google Colab).
    data_url = "https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv"

    print(f"Loading dataset from {data_url}...")
    try:
        data = pd.read_csv(data_url)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection or the dataset URL.")
        return

    print("Dataset loaded successfully.")
    print("\n--- Data Head (First 5 Rows) ---")
    print(data.head())

    print("\n--- Dataset Info ---")
    data.info()

    # This dataset is very clean, so no need for fillna()
    # print("\nChecking for missing values...")
    # print(data.isnull().sum())

    # --- 2. Define Features (X) and Target (y) ---
    # X = The properties we use to make a prediction
    # y = The actual outcome (the recommended crop)
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'

    X = data[features]
    y = data[target]

    # Get the list of unique crop names for the report
    crop_names = sorted(y.unique())
    print(f"\nModel will learn to recommend from {len(crop_names)} crops.")

    # --- 3. Scale Features ---
    # Scaling ensures all features have a similar range.
    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 4. Split Data ---
    # We split our data into a training set (to teach the model)
    # and a testing set (to see how well it learned).
    # We use 'stratify=y' to ensure all crop types are represented
    # in both the training and testing sets.
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size:  {X_test.shape[0]} samples")

    # --- 6. Create and Train the Model ---
    # We'll use a RandomForestClassifier again, as it's excellent
    # for this type of multiclass classification problem.
    print("\nCreating and training the RandomForestClassifier model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # This is where the model "learns" from the training data
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 7. Evaluate the Model ---
    # Now we test the model on the 'X_test' data it has never seen before.
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = model.predict(X_test)

    # Compare the model's predictions (y_pred) to the true answers (y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    # The Classification Report gives us detailed metrics
    # for *each* crop type.
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=crop_names)
    print(report)

    # --- 8. Project Demo: Predict on New Samples ---
    # Let's see how our model performs on a few test samples.
    print("\n--- Project Demo: Predicting on 3 samples from the test set ---")
    for i in range(3):
        sample = X_test[i]
        true_label = y_test.iloc[i]

        # Reshape the sample because the model expects a 2D array
        sample_reshaped = sample.reshape(1, -1)

        prediction = model.predict(sample_reshaped)[0]

        print(f"\nSample #{i+1}:")
        print(f"  Model Prediction: {prediction}")
        print(f"  Actual Label:     {true_label}")

if __name__ == "__main__":
    run_crop_recommendation_project()

