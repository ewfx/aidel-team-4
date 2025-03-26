from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def compute_confidence_score(risk_score: int, justification: str) -> float:
    """
    Uses FinBERT to assess how well the justification aligns with the risk score.
    Returns a confidence score between 0 and 10.
    """
    # Format input text
    text = f"Risk Score: {risk_score}. Justification: {justification}"

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs).logits

    # Convert logits to probability using softmax
    probabilities = F.softmax(outputs, dim=-1).squeeze()

    # Take the probability of the 'entailment' class (index 2)
    confidence_score = probabilities[2].item()   # Scale to 0-10

    return {
        "risk_score": risk_score,
        "justification": justification,
        "confidence_score": confidence_score
    }

