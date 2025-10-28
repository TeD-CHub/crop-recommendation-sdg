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

The model achieved an Overall Accuracy of 99.55% on the unseen test data.

Classification Report:
(venv) PS C:\Users\teddy\Desktop\Crop SDG Project> python crop_recommender.py
Loading dataset from https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv...
Dataset loaded successfully.

--- Data Head (First 5 Rows) ---
    N   P   K  temperature   humidity        ph    rainfall label
0  90  42  43    20.879744  82.002744  6.502985  202.935536  rice
1  85  58  41    21.770462  80.319644  7.038096  226.655537  rice
2  60  55  44    23.004459  82.320763  7.840207  263.964248  rice
3  74  35  40    26.491096  80.158363  6.980401  242.864034  rice
4  78  42  42    20.130175  81.604873  7.628473  262.717340  rice

--- Dataset Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2200 entries, 0 to 2199
Data columns (total 8 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   N            2200 non-null   int64
 1   P            2200 non-null   int64
 2   K            2200 non-null   int64
 3   temperature  2200 non-null   float64
 4   humidity     2200 non-null   float64
 5   ph           2200 non-null   float64
 6   rainfall     2200 non-null   float64
 7   label        2200 non-null   object
dtypes: float64(4), int64(3), object(1)
memory usage: 137.6+ KB

Model will learn to recommend from 22 crops.
Scaling features using StandardScaler...
Splitting data into training (80%) and testing (20%) sets...
Training set size: 1760 samples
Testing set size:  440 samples

Creating and training the RandomForestClassifier model...
Model training complete.

--- Model Evaluation on Test Set ---
Overall Accuracy: 99.55%

Classification Report:
              precision    recall  f1-score   support

       apple       1.00      1.00      1.00        20
      banana       1.00      1.00      1.00        20
   blackgram       1.00      0.95      0.97        20
    chickpea       1.00      1.00      1.00        20
     coconut       1.00      1.00      1.00        20
      coffee       1.00      1.00      1.00        20
      cotton       1.00      1.00      1.00        20
      grapes       1.00      1.00      1.00        20
        jute       0.95      1.00      0.98        20
 kidneybeans       1.00      1.00      1.00        20
      lentil       1.00      1.00      1.00        20
       maize       0.95      1.00      0.98        20
       mango       1.00      1.00      1.00        20
   mothbeans       1.00      1.00      1.00        20
    mungbean       1.00      1.00      1.00        20
   muskmelon       1.00      1.00      1.00        20
      orange       1.00      1.00      1.00        20
      papaya       1.00      1.00      1.00        20
  pigeonpeas       1.00      1.00      1.00        20
 pomegranate       1.00      1.00      1.00        20
        rice       1.00      0.95      0.97        20
  watermelon       1.00      1.00      1.00        20

    accuracy                           1.00       440
   macro avg       1.00      1.00      1.00       440
weighted avg       1.00      1.00      1.00       440


--- Project Demo: Predicting on 3 samples from the test set ---

Sample #1:
  Model Prediction: orange
  Actual Label:     orange

Sample #2:
  Model Prediction: banana
  Actual Label:     banana

Sample #3:
  Model Prediction: cotton
  Actual Label:     cotton

Analysis: The classification report shows the model's high performance (Precision, Recall, F1-score) for each of the 22 crop types. This high accuracy demonstrates the model's strong ability to identify the complex patterns linking soil/climate data to specific crop needs. This is a powerful proof-of-concept for a reliable agricultural advisory tool.

Report Screenshot:
<img width="1366" height="768" alt="Final Report" src="https://github.com/user-attachments/assets/0cda3715-3637-4093-90a7-9e85292db4ce" />


4. Ethical & Social Reflection

While this AI solution has significant potential, it is critical to consider its ethical implications, which account for 20% of the project grade.

Data Bias: The primary risk is sampling bias. The dataset was likely collected from specific regions and may not include local or indigenous crop varieties vital to other communities. Using this model in a new region (e.g., Sub-Saharan Africa) without retraining it on local data could provide poor or even harmful recommendations.

The Cost of Error: A wrong recommendation is not a small error. If the model tells a farmer to plant a crop that subsequently fails, it could destroy their livelihood for the season, worsening poverty and hunger instead of helping. The model's reliability must be exceptionally high before deployment.

Fairness & Accessibility (The "Last-Mile" Problem): This solution promotes fairness by giving small-scale farmers access to data science. However, the model requires input data (soil test results, accurate weather data). A farmer who cannot afford a soil test or access a local weather station cannot use the tool. To be truly equitable, this AI solution must be bundled with low-cost sensors or public data infrastructure.
