import pandas as pd
import nltk
import string
import os
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Download NLTK data (one-time setup) ---
# This is necessary for tokenization and stopword removal
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' tokenizer data...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading 'stopwords' data...")
    nltk.download('stopwords', quiet=True)

# --- Configuration ---
DATA_FILE = os.path.join("training", "emotions.csv")
MODEL_DIR = "models"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_classifier.pkl")

# --- Text Pre-processing Function ---
def preprocess_text(text, stop_words):
    """
    Cleans and tokenizes text.
    1. Lowercase
    2. Tokenize
    3. Remove punctuation and non-alphabetic tokens
    4. Remove stopwords
    5. Join back into a string
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a single string
    return " ".join(tokens)

# --- Main Training Function ---
def train_model():
    """
    Loads data, trains the model, and saves the artifacts.
    """
    print("Starting model training process...")

    # Load stop words
    stop_words = set(stopwords.words('english'))

    # 1. Load Data
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {DATA_FILE} successfully.")
    except FileNotFoundError:
        print(f"[ERROR] Data file not found at {DATA_FILE}")
        print("Please download the 'emotions.csv' dataset from Kaggle and place it in the 'training/' folder.")
        return

    # Handle potential missing values in 'text'
    df['text'] = df['text'].fillna('')

    # 2. Pre-process Data
    print("Pre-processing text data... (This may take a minute)")
    # We create a new column with the cleaned text
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, stop_words))
    print("Pre-processing complete.")

    # 3. Define Features (X) and Target (y)
    X = df['processed_text']
    y = df['label'] # The 'label' column (0, 1, 2, 3, 4, 5)

    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # 5. Vectorize Text
    print("Fitting TF-IDF vectorizer...")
    # We use TF-IDF to convert text into numerical features
    # max_features=5000 means we only keep the top 5000 most frequent words
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit on training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Only transform the test data (using the vocab from training)
    X_test_tfidf = vectorizer.transform(X_test)
    print("Vectorizing complete.")

    # 6. Train Model
    print("Training Logistic Regression classifier...")
    # Logistic Regression is a good, fast, and reliable model for text classification
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("Training complete.")

    # 7. Evaluate Model
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # 8. Save Artifacts
    print("Saving model and vectorizer...")
    
    # Ensure the 'models' directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    print("\nTraining process finished successfully!")

# --- Run the script ---
if __name__ == "__main__":
    train_model()
