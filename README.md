# ✈️ Airline Passenger Satisfaction Prediction Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker)
![Status](https://img.shields.io/badge/Project-Completed-success)

## 📌 Project Overview

This project predicts airline passenger satisfaction using **Artificial Neural Networks (ANN)** and machine learning techniques. It analyzes passenger demographics, travel information, and service ratings to classify passengers as **Satisfied** or **Neutral or Dissatisfied**.

The project includes data preprocessing, feature engineering, feature selection, model comparison, ANN fine-tuning, performance evaluation, and API deployment using FastAPI and Docker.

## 🎯 Problem Statement

Airline passenger satisfaction is influenced by multiple factors, including seat comfort, online boarding, in-flight entertainment, service quality, and travel class.

The objective of this project is to build a binary classification model that predicts passenger satisfaction from these factors. The model can help identify important service areas and support data-driven improvements in passenger experience.

## 📊 Dataset

- **Dataset:** Airline Passenger Satisfaction
- **Source:** [Maven Analytics Data Playground](https://mavenanalytics.io/data-playground)
- **Total Records:** 129,880
- **Original Columns:** 24
- **Target Variable:** `Satisfaction`
- **Problem Type:** Binary Classification

### Target Classes

| Class | Description |
|---|---|
| Satisfied | Passenger is satisfied with the airline experience |
| Neutral or Dissatisfied | Passenger is neutral or dissatisfied |

## 🛠️ Technologies Used

- **Python** — Data processing and model development
- **Pandas & NumPy** — Data manipulation and numerical operations
- **Matplotlib & Seaborn** — Exploratory data analysis and visualization
- **Scikit-learn** — Encoding, scaling, baseline models, and evaluation
- **TensorFlow / Keras** — ANN development and fine-tuning
- **Joblib** — Saving and loading preprocessing artifacts
- **FastAPI** — Building the prediction API
- **Uvicorn** — Running the API server
- **Docker** — Containerizing the application

## 🔄 Project Workflow

1. Data collection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Feature engineering
5. Categorical encoding
6. Train, validation, and test splitting
7. Feature selection
8. Feature scaling
9. Model training and comparison
10. ANN fine-tuning
11. Final model evaluation
12. Saving the model and preprocessing artifacts
13. FastAPI development
14. Docker deployment and API testing

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset's structure, distributions, missing values, target balance, and relationships between features.

The analysis included:

- Dataset shape and data types
- Missing-value analysis
- Duplicate-record analysis
- Target-class distribution
- Numerical feature distributions
- Correlation analysis
- Passenger service-rating analysis
- Identification of potentially important satisfaction factors

### Key Findings

- The dataset contains 129,880 passenger records.
- The target variable contains two satisfaction categories.
- `Arrival Delay` contained 393 missing values.
- No duplicate records were identified in the initial dataset.
- Passenger class, online boarding, and service-related ratings showed useful relationships with satisfaction.

## 🧹 Data Cleaning and Preprocessing

The following preprocessing steps were performed:

- Removed the unnecessary passenger identifier column.
- Handled missing values in `Arrival Delay`.
- Checked duplicate records.
- Encoded categorical variables into numerical representations.
- Applied one-hot encoding to the `Class` column.
- Created additional service-related features.
- Split the data into training, validation, and test sets.
- Selected the most relevant input features.
- Standardized the selected features using `StandardScaler`.

### Categorical Encoding

Label encoding was applied to binary or target categorical columns, including:

- `Satisfaction`
- `Gender`
- `Customer Type`
- `Type of Travel`

One-hot encoding was applied to `Class` to create numerical indicator columns.

### Feature Scaling

`StandardScaler` was used to standardize the selected numerical and encoded features. The scaler was fitted on the training data and then applied to the validation, test, and inference data using the same fitted transformation.

## ⚙️ Feature Engineering

Three additional features were created to represent broader aspects of the passenger experience:

- **Digital Experience Score** — Summarizes relevant digital-service ratings.
- **Comfort Score** — Combines relevant comfort-related ratings.
- **Service Experience Score** — Summarizes selected service-quality ratings.

These engineered features provide additional representations of related passenger-service attributes for the prediction model.

## 🎯 Feature Selection

Feature selection was performed to reduce the input features from **25 engineered and encoded features(excluded target and identifier) to 15 selected features**.

### Selection Method

A correlation-based approach was used to rank input features according to their absolute correlation with the target variable, `Satisfaction`.

1. Calculated the correlation between each input feature and the target.
2. Used absolute correlation values to consider both positive and negative relationships.
3. Ranked features by their absolute correlation values.
4. Selected the 15 highest-ranked features.
5. Applied the same selected feature set to the training, validation, and test datasets.
6. Saved the selected feature names in `selected_features.pkl` for consistent use during deployment.

### Selected Features

1. `Class_Business`
2. `Online Boarding`
3. `Comfort Score`
4. `Type of Travel`
5. `Class_Economy`
6. `In-flight Entertainment`
7. `Digital Experience Score`
8. `Service Experience Score`
9. `Seat Comfort`
10. `On-board Service`
11. `Leg Room Service`
12. `Cleanliness`
13. `Flight Distance`
14. `In-flight Wifi Service`
15. `Baggage Handling`

### Why Feature Selection?

- Reduced the number of model inputs.
- Prioritized features with stronger linear associations with the target.
- Simplified the ANN input layer.
- Helped reduce unnecessary input information.
- Ensured the same feature order was used during model training and API prediction.

**Note:** Correlation measures linear association and does not necessarily capture nonlinear relationships or feature interactions. Feature selection should be calculated using training data only to avoid information leakage.

## 🤖 Model Development

Five model approaches were evaluated to compare conventional machine learning with neural-network-based classification.

| Model | Purpose |
|---|---|
| Logistic Regression | Establishes a conventional classification baseline |
| Random Forest | Captures nonlinear relationships using an ensemble of decision trees |
| Baseline ANN | Establishes a neural-network baseline |
| Improved ANN | Evaluates an enhanced neural-network architecture |
| Final Fine-Tuned ANN | Uses selected hyperparameters for the final model |

The models were evaluated using classification metrics, including accuracy, precision, recall, and F1-score. The final ANN was evaluated on the held-out test set.

## 🧠 Final ANN Model Architecture

The final model uses a fully connected Artificial Neural Network with two hidden layers.

| Layer | Configuration | Purpose |
|---|---|---|
| Input Layer | 15 selected features | Receives passenger data |
| Dense Layer 1 | 256 neurons, ReLU | Learns complex patterns |
| Batch Normalization | — | Helps stabilize training |
| Dropout | 0.4 | Helps reduce overfitting |
| Dense Layer 2 | 64 neurons, ReLU | Learns higher-level patterns |
| Batch Normalization | — | Helps stabilize training |
| Dropout | 0.4 | Improves regularization |
| Output Layer | 1 neuron, Sigmoid | Produces a binary-classification probability |

### Hyperparameters

| Parameter | Value |
|---|---|
| Random Seed | 42 |
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Loss Function | Binary Cross-Entropy |
| Evaluation Metric | Accuracy |
| Early Stopping Patience | 8 epochs |
| Learning Rate Reduction Factor | 0.5 |
| Reduce LR Patience | 3 epochs |
| Minimum Learning Rate | 0.00001 |

### Training Callbacks

- **EarlyStopping:** Monitors validation loss and stops training when it fails to improve for eight consecutive epochs. The best weights are restored.
- **ReduceLROnPlateau:** Reduces the learning rate by half when validation loss fails to improve for three consecutive epochs, down to a minimum learning rate of `0.00001`.

### Why This Architecture?

- **ReLU:** Learns nonlinear relationships in the input data.
- **Batch Normalization:** Helps stabilize training.
- **Dropout:** Randomly disables 40% of activations in the relevant layers during training to help reduce overfitting.
- **Sigmoid:** Produces a probability between 0 and 1 for binary classification.
- **Early Stopping and Learning Rate Reduction:** Help control training and refine model optimization.

## 📈 Model Evaluation

The final ANN was evaluated using the held-out test dataset.

### Final ANN Test Results

| Metric | Score |
|---|---:|
| Accuracy | 95.27% |
| Precision | 95.90% |
| Recall | 93.10% |
| F1-Score | 94.48% |

### Interpretation

- **Accuracy — 95.27%:** The model correctly classified approximately 95% of the test samples.
- **Precision — 95.90%:** Most predictions for the positive class were correct.
- **Recall — 93.10%:** The model identified approximately 93% of actual positive-class samples.
- **F1-Score — 94.48%:** The model achieved a strong balance between precision and recall.

These metrics describe overall or positive-class performance as defined by the evaluation code. Class-wise precision and recall should be examined separately to understand how well the model identifies satisfied passengers compared with neutral or dissatisfied passengers.

## ⚠️ Error Analysis and Limitations

- **Class-wise Performance:** The model may identify satisfied passengers less effectively than neutral or dissatisfied passengers. Class-wise recall should be monitored to understand this difference.
- **Misclassification:** Passengers with similar service ratings may belong to different satisfaction categories, making some cases difficult to classify.
- **Feature Relationships:** Correlation-based selection may miss useful nonlinear relationships and interactions.
- **Generalization:** Performance may differ on new passenger records or datasets from other airlines.
- **Input Dependency:** Predictions depend on the quality and completeness of the passenger information and service ratings.
- **Deployment Scope:** The API has been tested locally through Docker; public cloud deployment has not been completed.
- **Feature Representation:** The selected feature set does not include a separate `Class_Economy Plus` indicator. This limits the model's ability to distinguish that class independently through the selected class-indicator features.

## 🚀 API Development with FastAPI

A FastAPI application was created to serve predictions using the saved ANN model and preprocessing artifacts.

The API performs the following steps:

1. Receives passenger information as input.
2. Creates the engineered service-score features.
3. Prepares the input using the saved feature selection and encoding logic.
4. Selects the 15 features in the expected order.
5. Scales the input using the saved scaler.
6. Passes the processed input to the ANN.
7. Returns the predicted satisfaction class and probability.

### API Response Example

```json
{
  "prediction": "Satisfied",
  "predicted_class": 1,
  "satisfaction_probability": 0.9975,
  "class_used": "Business",
  "features_used": 15
}
```

This is an example of a successful local prediction, not a guaranteed output for every input.

### Run the API Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

[http://localhost:8000/docs](http://localhost:8000/docs)

## 🐳 Docker Deployment

Docker was used to package the FastAPI application, trained ANN model, and preprocessing artifacts into a container.

### Build the Docker Image

Run the following command from the project directory containing the `Dockerfile`:

```bash
docker build -t airline-satisfaction-api .
```

### Run the Container

```bash
docker run -d --name airline-api -p 8000:8000 airline-satisfaction-api
```

### Check the Running Container

```bash
docker ps
```

### View Application Logs

```bash
docker logs airline-api
```

### Access the API

Open:

[http://localhost:8000/docs](http://localhost:8000/docs)

Use the Swagger interface to submit passenger details and inspect prediction responses.

**Deployment status:** The API was successfully tested in a local Docker container. It is not currently a public cloud-hosted API.

## 📁 Project Structure

```text
Airline-Passenger-Satisfaction/
│
├── app.py
├── ann_model.keras
├── scaler.pkl
├── selected_features.pkl
├── label_encoders.joblib
├── requirements.txt
├── Dockerfile
├── README.md
└── notebook/
    └── airline_passenger_satisfaction.ipynb
```

The notebook folder and notebook filename are illustrative; adjust them to match the actual repository structure.

### Important Files

| File | Description |
|---|---|
| `app.py` | FastAPI prediction application |
| `ann_model.keras` | Saved final ANN model |
| `scaler.pkl` | Fitted feature scaler |
| `selected_features.pkl` | Ordered list of selected features |
| `label_encoders.joblib` | Saved categorical encoding mappings |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Instructions for building the Docker image |
| `README.md` | Project documentation |

## 📦 Requirements

The project uses the following Python dependencies:

```text
fastapi
uvicorn[standard]
tensorflow
numpy
pandas
scikit-learn
joblib
```

For reproducible deployment, use compatible dependency versions, especially for TensorFlow and scikit-learn, which are used to load the saved model and scaler.

## 🔮 Future Improvements

- Evaluate additional feature-selection methods, including mutual information and model-based feature importance.
- Improve recall for the satisfied passenger class where class-wise evaluation identifies a weakness.
- Analyze misclassified passengers using the confusion matrix and classification report.
- Evaluate the model on additional airline datasets.
- Add input validation and clearer API error messages.
- Pin dependency versions for reproducible Docker builds.
- Deploy the API to a cloud platform.
- Add automated tests for preprocessing and prediction consistency.

## ✅ Conclusion

This project demonstrates an end-to-end machine learning workflow for airline passenger satisfaction prediction, from exploratory data analysis and feature engineering to ANN fine-tuning and API deployment.

The final ANN achieved **95.27% accuracy and a 94.48% F1-score** on the test dataset. FastAPI and Docker were used to package and test the prediction service locally.

The project provided practical experience in data preprocessing, feature selection, neural-network development, model evaluation, and deployment.

---

**Project Category:** Deep Learning  
**Project Title:** Airline Passenger Satisfaction Prediction Using Deep Learning  
**Student:** Yuvalakshmi Karthikeyan  
**Batch:** TN-DS-AN-B02  
**Submission Date:** 30 October 2026
