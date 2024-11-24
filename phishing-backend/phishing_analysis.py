import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    log_loss,
)
import re
import joblib

df = pd.read_csv("phishing_email.csv")
df = df.dropna().drop_duplicates()

phishing_emails = df[df["label"] == 1]
non_phishing_emails = df[df["label"] == 0]
phishing_sample = phishing_emails.sample(n=250, random_state=42)
non_phishing_sample = non_phishing_emails.sample(n=250, random_state=42)
df_sample = pd.concat([phishing_sample, non_phishing_sample]).reset_index(drop=True)
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

df_sample.to_csv("cleaned_phishing_dataset.csv", index=False)

X = df_sample["text_combined"]
y = df_sample["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

def preprocess_text(text, unwanted_terms):
    text = text.lower()
    text = re.sub(r"\b\d+\b", "", text)  
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    for term in unwanted_terms:
        text = re.sub(rf"\b{term}\b", "", text, flags=re.IGNORECASE)
    return text

unwanted_terms = ["enron", "forwarded"]
df_sample["text_combined"] = df_sample["text_combined"].apply(lambda x: preprocess_text(x, unwanted_terms))

svm_model = SVC(kernel="linear", probability=True, random_state=42)
svm_model.fit(X_train_tfidf, y_train)

joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

feature_names = tfidf.get_feature_names_out()
weights = svm_model.coef_.toarray()[0]
top_features = sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True)
top_keywords = [{"word": feature, "weight": weight} for feature, weight in top_features]
joblib.dump(top_keywords, "top_keywords.pkl")

y_pred_svm = svm_model.predict(X_test_tfidf)
y_pred_proba_svm = svm_model.predict_proba(X_test_tfidf)[:, 1]

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {accuracy_svm:.2f}")