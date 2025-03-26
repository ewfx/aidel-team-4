# Import libraries

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM

# Load FinBERT Model for Risk Scoring
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

# Load Flan-T5 Model for Justification
justification_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
justification_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Risk label mapping
risk_labels = {0: "Low", 1: "Medium", 2: "High"}

# Function to get risk score using FinBERT
def get_risk_score(news_text):
    inputs = finbert_tokenizer(news_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
        risk_level = torch.argmax(outputs.logits).item()
        risk_score = torch.softmax(outputs.logits, dim=1)[0][risk_level].item()
    return risk_labels[risk_level], round(risk_score, 2)

# Function to generate justification using Flan-T5
def generate_justification(news_text, risk_category):
    prompt = f"News Report: {news_text}\nRisk Category: {risk_category}\nExplain why this entity is risky."
    inputs = justification_tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    with torch.no_grad():
        output = justification_model.generate(**inputs, max_length=100)
    
    justification = justification_tokenizer.decode(output[0], skip_special_tokens=True)
    return justification

# Combined function to analyze entity risk from news
def analyze_entity(news_text):
    risk_category, risk_score = get_risk_score(news_text)
    justification = generate_justification(news_text, risk_category)
    
    return {
        "risk_category": risk_category,
        "risk_score": risk_score,
        "justification": justification
    }

# Example news report (replace with actual news text)
news_text = """
XYZ Corporation is under investigation for potential money laundering activities linked to offshore accounts in the Cayman Islands. 
Authorities have identified suspicious transactions totaling over $500 million. 
Regulators are considering sanctions against the company for violating anti-money laundering laws.
"""

# Run analysis
result = analyze_entity(news_text)
print(result)
