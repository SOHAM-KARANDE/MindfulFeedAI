import streamlit as st
from transformers import pipeline
import pandas as pd
from datetime import datetime

# Load the free local classifier (DistilBERT)
@st.cache_resource
def load_classifier():
    return pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_classifier()

# Classification logic
def classify_content(title, desc):
    text = title + " " + desc
    result = classifier(text)[0]
    label = result['label']
    return "Mindful" if label == "POSITIVE" else "Mindless"

# Streamlit UI
st.title("🧠 MindfulFeed AI")
st.markdown("Classify YouTube content as **Mindful** or **Mindless** using AI.")

title = st.text_input("🎬 Video Title")
desc = st.text_area("📝 Video Description")

if st.button("Classify"):
    if title and desc:
        label = classify_content(title, desc)
        st.success(f"✅ This content is: **{label}**")
        
        # Save to CSV
        log = pd.DataFrame([[datetime.now(), title, desc, label]], 
                           columns=["Timestamp", "Title", "Description", "Label"])
        try:
            old = pd.read_csv("log.csv")
            log = pd.concat([old, log], ignore_index=True)
        except FileNotFoundError:
            pass
        log.to_csv("log.csv", index=False)
    else:
        st.warning("Please enter both a title and description.")

if st.checkbox("📊 Show Classification Log"):
    try:
        df = pd.read_csv("log.csv")
        st.dataframe(df.tail(10))
    except:
        st.info("No logs found yet.")
