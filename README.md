# 🧠 Customer Feedback Sentiment Analysis

A simple **Machine Learning Sentiment Analysis System** that classifies customer reviews as **Positive** or **Negative** using **TF-IDF Vectorization** and **Logistic Regression**.

## 📌 Project Overview

This project analyzes customer feedback and predicts the sentiment of a review.

The system:

* Loads a customer feedback dataset.
* Converts text into numerical features using **TF-IDF**.
* Trains a **Logistic Regression** classification model.
* Evaluates the model using accuracy and a classification report.
* Provides an interactive system for entering new reviews.
* Maintains a session history of analyzed reviews.

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Dataset handling
* **Scikit-learn** – Machine Learning
* **TF-IDF Vectorizer** – Text feature extraction
* **Logistic Regression** – Sentiment classification
* **Classification Report** – Model evaluation

## 📂 Project Structure

```text
sentiment-analysis/
│
├── sentiment_analysis.py
├── README.md
└── requirements.txt
```

## 📦 Requirements

Create a file named `requirements.txt` with:

```text
pandas
scikit-learn
```

Install them using:

```bash
pip install -r requirements.txt
```

## ⚙️ How It Works

### 1. Dataset Creation

The project contains customer reviews labeled as:

* `Positive`
* `Negative`

Example:

```text
"I love this product" → Positive
"I hate this product" → Negative
```

### 2. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

### 3. TF-IDF Vectorization

TF-IDF converts text into numerical values that can be understood by the machine learning model.

```python
vectorizer = TfidfVectorizer(stop_words='english')
```

### 4. Logistic Regression

A Logistic Regression model is trained using the TF-IDF features.

```python
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
```

### 5. Model Evaluation

The model is evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score

### 6. Interactive Sentiment Analysis

After training, users can enter their own reviews.

Example:

```text
Enter your review: This product is amazing
Result -> Sentiment: POSITIVE
```

You can also use:

```text
history
```

to view previous reviews.

Use:

```text
exit
```

to close the application.

## 🚀 Installation

Make sure Python is installed.

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

Run the Python file:

```bash
python sentiment_analysis.py
```

The program will first train and evaluate the model and then start the interactive sentiment analysis system.

## 📊 Sample Output

```text
Total dataset size: 14 samples loaded successfully.

----- Model Evaluation Metrics -----
Accuracy Score: XX.XX%

Classification Report:
              precision    recall    f1-score
Negative        ...
Positive        ...

----------------------------------------

----- Interactive Sentiment Analysis System -----
Type your review below to analyze its sentiment.
Type 'history' to see past inputs, or 'exit' to quit.

Enter your review: I really love this product

Result -> Sentiment: POSITIVE
```

## 🔍 Example Predictions

| Customer Review             | Prediction |
| --------------------------- | ---------- |
| I love this product         | Positive   |
| This phone is amazing       | Positive   |
| Excellent quality           | Positive   |
| Very happy with my purchase | Positive   |
| I hate this product         | Negative   |
| Very bad experience         | Negative   |
| Worst product ever          | Negative   |
| The quality is terrible     | Negative   |

## 🎯 Features

* ✅ Text classification
* ✅ Positive/Negative sentiment prediction
* ✅ TF-IDF text processing
* ✅ Logistic Regression ML model
* ✅ Accuracy evaluation
* ✅ Classification report
* ✅ Interactive user input
* ✅ Review history
* ✅ Exit command

## 📈 Future Improvements

This project can be improved by:

* Adding a much larger real-world customer review dataset.
* Supporting **Neutral** sentiment.
* Using advanced NLP techniques.
* Adding stemming or lemmatization.
* Comparing multiple ML algorithms.
* Creating a web or desktop interface.
* Adding charts and sentiment analytics.
* Deploying the model as an API.


