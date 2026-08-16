import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Comprehensive Extended Dataset (Customer Feedback)
data = {
    "text": [
        # Positive Sentences
        "I recently purchased this item after reading several reviews, and I am genuinely blown away by how well it performs on a daily basis.",
        "The overall build quality is exceptionally sturdy, and the customer service team went completely out of their way to ensure all my setup questions were answered.",
        "I love this product", "This phone is amazing", "Excellent quality", "Very happy with my purchase", "The service is excellent",
        
        # Negative Sentences
        "This has undoubtedly been the most frustrating consumer experience of my entire year, as the device completely stopped working after barely two days.",
        "Not only is the hardware build quality extremely cheap and flimsy, but the support representatives were entirely unhelpful and rude.",
        "I hate this product", "Very bad experience", "Worst product ever", "The quality is terrible", "The service is very poor"
    ],
    "sentiment": [
        "Positive", "Positive", "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative", "Negative", "Negative"
    ]
}
# 2. Initialize and Prepare DataFrame
df = pd.DataFrame(data)
print(f"Total dataset size: {len(df)} samples loaded successfully.\n")

# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], 
    df["sentiment"], 
    test_size=0.2, 
    random_state=42
)

# 4. Vectorize Text Data using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# 6. Evaluate Model Performance
y_pred = model.predict(X_test_vectorized)
print("----- Model Evaluation Metrics -----")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("-" * 40)

# 7. Interactive Sentiment Analysis System Loop
print("\n----- Interactive Sentiment Analysis System -----")
print("Type your review below to analyze its sentiment.")
print("Type 'history' to see past inputs, or 'exit' to quit.\n")

history_log = []

while True:
    review = input("Enter your review: ").strip()
    
    if review.lower() == "exit":
        print("\nThank you for using the Sentiment Analysis System. Goodbye!")
        break
        
    if review.lower() == "history":
        print("\n--- Session History ---")
        if not history_log:
            print("No reviews entered yet.")
        else:
            for idx, (rev, sent) in enumerate(history_log, 1):
                print(f"{idx}. '{rev}' --> **{sent}**")
        print("-" * 25)
        continue
        
    if not review:
        print("Please enter a valid non-empty review.")
        continue
        
    # Transform input and predict sentiment
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    
    # Store in history
    history_log.append((review, prediction))
    
    # Output result cleanly
    print(f"Result -> Sentiment: **{prediction.upper()}**\n")