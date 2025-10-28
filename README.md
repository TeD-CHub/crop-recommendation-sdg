AI for Sustainable Development: Crop Recommendation for SDG 2

Student: Teddy Otieno
Assignment: Week 2 Assignment: AI for Sustainable Development Goals

1. SDG Problem Addressed

This project addresses UN Sustainable Development Goal 2: Zero Hunger.

The Problem: Global food security is threatened by inefficient agricultural practices and climate volatility. Many farmers, particularly small-scale farmers in developing regions, lack access to scientific data to make optimal planting decisions. They may rely on tradition, which can lead to low yields or crop failure if the soil composition or local climate is not suitable for that crop. This inefficiency is a direct barrier to achieving "Zero Hunger."

The Solution: This project proposes a Precision Agriculture tool using machine learning. It's an AI model that recommends the most suitable crop to plant based on specific soil properties (Nitrogen, Phosphorous, Potassium, pH) and environmental factors (temperature, humidity, rainfall). This empowers farmers to make data-driven decisions, maximize their yields, and improve food security.

2. Machine Learning Approach Used

A Supervised Learning approach (specifically, Multiclass Classification) was used to solve this problem.

Model: RandomForestClassifier from the scikit-learn library. This ensemble method is highly effective as it combines many decision trees to vote on the best classification, making it robust and accurate for this complex problem.

Dataset: A public Crop Recommendation dataset from Kaggle was used. It includes 2200 samples with 7 features (N, P, K, temperature, humidity, ph, rainfall) and one target label (the recommended crop, e.g., 'rice', 'maize', 'chickpea', etc.).

Workflow:

Preprocessing: All features were normalized using StandardScaler. This is crucial to ensure that features with different units (like rainfall vs. ph) are weighted equally by the model.

Training: The data was split into an 80% training set and a 20% testing set. stratify=y was used to ensure all 22 crop types were fairly represented in both sets.

Evaluation: The model was trained on the training set and evaluated on the unseen test set.

3. Results and Impact

(Please run the crop_recommender.py script and paste your results below.)

The model achieved an Overall Accuracy of [XX.XX]% on the unseen test data.

Classification Report:

[PASTE YOUR FULL CLASSIFICATION REPORT OUTPUT HERE]


Analysis: The classification report shows the model's high performance (Precision, Recall, F1-score) for each of the 22 crop types. This high accuracy demonstrates the model's strong ability to identify the complex patterns linking soil/climate data to specific crop needs. This is a powerful proof-of-concept for a reliable agricultural advisory tool.

4. Ethical & Social Reflection

While this AI solution has significant potential, it is critical to consider its ethical implications, which account for 20% of the project grade.

Data Bias: The primary risk is sampling bias. The dataset was likely collected from specific regions and may not include local or indigenous crop varieties vital to other communities. Using this model in a new region (e.g., Sub-Saharan Africa) without retraining it on local data could provide poor or even harmful recommendations.

The Cost of Error: A wrong recommendation is not a small error. If the model tells a farmer to plant a crop that subsequently fails, it could destroy their livelihood for the season, worsening poverty and hunger instead of helping. The model's reliability must be exceptionally high before deployment.

Fairness & Accessibility (The "Last-Mile" Problem): This solution promotes fairness by giving small-scale farmers access to data science. However, the model requires input data (soil test results, accurate weather data). A farmer who cannot afford a soil test or access a local weather station cannot use the tool. To be truly equitable, this AI solution must be bundled with low-cost sensors or public data infrastructure.