import os
import joblib
import string
import nltk
import spotipy
from flask import Flask, render_template, request, jsonify
from spotipy.oauth2 import SpotifyClientCredentials
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()  # Load variables from .env

# --- Flask App Initialization ---
app = Flask(__name__, static_folder="static", template_folder="templates")

# --- NLTK Setup (for text preprocessing) ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Load Trained Models ---
try:
    model_path = os.path.join('models', 'emotion_classifier.pkl')
    vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ [INFO] ML models loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ [ERROR] Model files not found. Make sure you have run 'training/train_model.py'")
    print(f"Details: {e}")
    exit() # Exit if models aren't found

# --- Spotify API Setup ---
try:
    CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

    if not CLIENT_ID or not CLIENT_SECRET:
        print("❌ [ERROR] SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_SECRET not found in .env file.")
        raise ValueError("Missing Spotify credentials")

    print("✅ [INFO] Spotify credentials loaded from .env.")
    
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
    
    print("✅ [INFO] Spotify API credentials loaded.")
except Exception as e:
    print(f"❌ [ERROR] Could not initialize Spotify API.")
    print(f"Details: {e}")
    exit()

# --- Recommendation Logic & "Databases" ---

# 1. Map model output (0-5) to human-readable moods
MOOD_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 2. Define search keywords for each mood (SIMPLIFIED)
MOOD_KEYWORDS = {
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "love": "love",
    "fear": "dark",
    "surprise": "surprise",
    "uplift": "uplifting"
}

# 3. Define language search terms
LANGUAGE_KEYWORDS = {
    "en": "english",
    "te": "telugu",
    "hi": "hindi"
}

# --- Text Pre-processing Function ---
def preprocess_text(text):
    """Cleans and prepares text for the model."""
    tokens = word_tokenize(text.lower())
    processed_tokens = [
        stemmer.stem(word) for word in tokens 
        if word.isalnum() and word not in stop_words
    ]
    return " ".join(processed_tokens)

# --- Flask Routes ---

@app.route("/")
def home():
    """Serves the main HTML page."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Handles the main API request using the sp.search() endpoint.
    """
    try:
        # 1. Get user input from the JSON payload
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Invalid request. 'message' is required."}), 400

        text = data.get("message")
        language_choice = data.get("language", "en")
        user_preference = data.get("preference", "match")

        # 2. Pre-process text and predict mood
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        mood = MOOD_MAP.get(prediction, "joy")

        # 3. Determine the final search keyword
        if user_preference == "uplift" and mood in ["sadness", "anger", "fear"]:
            search_mood = "uplift"
        else:
            search_mood = mood

        # 4. Build the Spotify search query
        mood_keyword = MOOD_KEYWORDS.get(search_mood, "happy")
        lang_keyword = LANGUAGE_KEYWORDS.get(language_choice, "english")
        
        # Build a query like "happy telugu"
        query = f"{mood_keyword} {lang_keyword}"
        
        print(f"✅ [INFO] User mood: '{mood}', Search query: '{query}'")

        # 5. Call the Spotify Search API
        results = sp.search(
            q=query, 
            type="track", 
            limit=12, 
            market="IN"
        )
        
        # 6. Format Results
        songs = []
        for track in results["tracks"]["items"]:
            # We will now send all songs to the frontend.
            
            songs.append({
                "id": track["id"],
                "song_name": track["name"],  # <-- RENAMED (was "title")
                "artist": track["artists"][0]["name"],
                # Handle cases where album art might be missing
                "image_url": track["album"]["images"][0]["url"] if track["album"]["images"] else "https://placehold.co/100x100/222/fff?text=No+Art", # <-- RENAMED (was "album_art")
                "preview_url": track["preview_url"],
                "spotify_url": track["external_urls"]["spotify"]
            })
        
        print(f"✅ [INFO] Found {len(songs)} songs for the query.")
        return jsonify({"mood": search_mood, "songs": songs})

    except Exception as e:
        print(f"❌ [ERROR] in /api/predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)

