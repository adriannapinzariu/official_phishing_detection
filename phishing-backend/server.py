from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app)  

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\b\d+\b", "", text) 
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

def get_top_features(vectorizer, model, num_features=10):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.toarray()[0]
    sorted_features = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    
    phishing_features = [feature for feature, weight in sorted_features if weight > 0][:num_features]
    safe_features = [feature for feature, weight in sorted_features if weight < 0][:num_features]
    
    return phishing_features, safe_features

phishing_keywords, safe_keywords = get_top_features(vectorizer, model)


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    email_content = data.get("email_content", "")

    if not email_content:
        return jsonify({"error": "Email content is required"}), 400

    processed_text = preprocess_text(email_content)
    email_tfidf = vectorizer.transform([processed_text])
    prediction = model.predict(email_tfidf)[0]
    confidence = model.predict_proba(email_tfidf)[0][1] * 100

    #for ui testing purposes
    #phishing_keywords = ["urgent", "password", "click", "bank"]
    #safe_keywords = ["hello", "thank you", "regards", "attached"]

    phishing_features_in_email = [word for word in phishing_keywords if word in email_content.lower()]
    safe_features_in_email = [word for word in safe_keywords if word in email_content.lower()]

    result = {
        "result": "Phishing" if prediction else "Not Phishing",
        "likelihood": round(confidence, 2),
        "features": {
            "phishing": phishing_features_in_email,
            "safe": safe_features_in_email,
        },
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)

#