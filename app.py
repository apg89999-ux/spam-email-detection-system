import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------- PAGE ----------------
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")

st.title("📧 Email Spam & Phishing Detector")
st.write("Paste any email, Gmail message, SMS, or long email to analyze spam risk.")

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))

# ---------------- INPUT ----------------
sender = st.text_input("Sender Email (optional)", placeholder="example@gmail.com")
subject = st.text_input("Email Subject", placeholder="Enter email subject")
body = st.text_area("Email Message", height=250, placeholder="Paste complete email here...")

# ---------------- ANALYSIS ----------------
if st.button("🔍 Analyze Email"):
    full_text = f"{subject} {body}".strip()
    text = full_text.lower()

    if full_text == "":
        st.warning("Please enter an email subject or message.")
    else:
        # ML prediction
        input_tfidf = vectorizer.transform([full_text])
        ml_prob = model.predict_proba(input_tfidf)[0][1] * 100

        # Strong spam indicators
        strong_keywords = [
            "urgent", "verify", "password", "otp", "winner", "prize",
            "claim now", "bank", "account blocked", "suspended",
            "click here", "free", "lottery", "reward", "limited time",
            "security alert", "confirm identity", "update account",
            "payment failed", "gift card", "act now", "immediately"
        ]

        found = [k for k in strong_keywords if k in text]

        # URL detection
        urls = re.findall(r'https?://\S+', full_text)

        # Scoring
        score = ml_prob
        score += len(found) * 12
        if urls:
            score += 25

        # Suspicious sender domains
        if sender:
            domain = sender.split("@")[-1].lower()
            suspicious_domains = ["bank-alerts.com", "secure-login.com", "update-account.net"]
            if domain in suspicious_domains:
                score += 20

        final_score = min(score, 100)

        # Final decision
        st.markdown("---")
        st.subheader("🛡 Security Result")

        if final_score >= 70:
            st.error(f"🚨 SPAM DETECTED — High Risk ({final_score:.1f}%)")
            signal = "🔴 HIGH RISK"
        elif final_score >= 40:
            st.warning(f"⚠️ SUSPICIOUS EMAIL — Medium Risk ({final_score:.1f}%)")
            signal = "🟠 MEDIUM RISK"
        else:
            st.success(f"✅ SAFE EMAIL — Low Risk ({100-final_score:.1f}% confidence)")
            signal = "🟢 LOW RISK"

        st.metric("Risk Level", signal)
        st.progress(int(final_score))

        # Details
        col1, col2 = st.columns(2)
        col1.metric("Spam Probability", f"{final_score:.1f}%")
        col2.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Sender analysis
        st.subheader("👤 Sender Analysis")
        if sender:
            st.write(f"**Sender:** {sender}")
            domain = sender.split("@")[-1]
            st.write(f"**Domain:** {domain}")
        else:
            st.write("No sender address provided.")

        # URL analysis
        st.subheader("🔗 Links Found")
        if urls:
            for u in urls:
                st.write(f"• {u}")
        else:
            st.write("No URLs detected.")

        # Keyword analysis
        st.subheader("⚠️ Suspicious Keywords")
        if found:
            for k in found:
                st.write(f"• {k}")
        else:
            st.write("No suspicious keywords detected.")

        # Explanation
        st.subheader("📝 Why this result?")
        reasons = []
        if ml_prob > 50:
            reasons.append("Machine learning model found spam-like text patterns.")
        if found:
            reasons.append("Urgent or suspicious words were detected.")
        if urls:
            reasons.append("The email contains clickable links.")
        if len(body) < 30:
            reasons.append("Very short promotional messages are commonly spam.")

        if reasons:
            for r in reasons:
                st.write(f"• {r}")
        else:
            st.write("The email resembles normal communication.")

# ---------------- EXAMPLE BOXES ----------------
st.markdown("---")
st.subheader("📌 Example Emails")
colA, colB = st.columns(2)

with colA:
    st.error("""
🚨 **Spam Example**

**Subject:** Urgent Account Verification

Dear Customer, your account will be suspended within 24 hours. Verify immediately at https://secure-update-login.com and claim your reward now.

**Result:** SPAM / HIGH RISK
    """)

with colB:
    st.success("""
✅ **Safe Example**

**Subject:** DBMS Lab Schedule

Hi Ankan, tomorrow's DBMS lab starts at 10 AM in Room 204. Please bring your notebook and ID card.

**Result:** SAFE / LOW RISK
    """)

st.markdown("---")
st.caption("Built by Ankan Pramanik | Python • Scikit-learn • Streamlit")