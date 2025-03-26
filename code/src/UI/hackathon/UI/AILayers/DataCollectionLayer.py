import requests
import pandas as pd
import os
import re
import torch
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import pickle
import xml.etree.ElementTree as ET
import logging
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from datetime import datetime
import requests

import os





# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

OFAC_CSV_URL = "https://www.treasury.gov/ofac/downloads/sdn.csv"
OFAC_FILE = "ofac_sanctions.csv"
OFAC_EMBEDDINGS_FILE = "ofac_embeddings.pkl"

HEADERS = {"User-Agent": "my-sec-bot/1.0 (contact: myemail@example.com)"}

# Load NLP Models
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = pipeline("text-classification", model=finbert_model, tokenizer=tokenizer)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_model = spacy.load("en_core_web_sm")


def download_ofac_list():
    if os.path.exists(OFAC_FILE):
        logging.info("OFAC sanctions list already exists. Skipping download.")
        return

    response = requests.get(OFAC_CSV_URL, stream=True)
    if response.status_code == 200:
        with open(OFAC_FILE, "wb") as file:
            file.write(response.content)
        logging.info("OFAC sanctions list downloaded successfully.")
    else:
        logging.error("Failed to download OFAC list.")


def clean_name(name):
    return re.sub(r"[^a-zA-Z\s]", "", str(name)).strip().lower()

def load_ofac_data():
    try:
        download_ofac_list()
        df = pd.read_csv(OFAC_FILE, encoding="ISO-8859-1", header=None)

        # Extract columns 1 and 11
        df = df[[1, 11]]
        df.columns = ["Name", "Additional Info"]  # Rename for clarity

        # Clean the "Name" column
        df["Cleaned Name"] = df["Name"].apply(clean_name)

        return df
    except Exception as e:
        logging.error(f"Error loading OFAC data: {e}")
        return None

def get_sanctioned_embeddings():
    df = load_ofac_data()
    if df is None:
        return None, None

    sanctioned_names = df["Cleaned Name"].dropna().tolist()

    if os.path.exists(OFAC_EMBEDDINGS_FILE):
        with open(OFAC_EMBEDDINGS_FILE, "rb") as f:
            cached_data = pickle.load(f)
            if cached_data["names"] == sanctioned_names:
                logging.info("Loaded cached embeddings.")
                return df, cached_data["embeddings"]

    sanctioned_embeddings = model.encode(sanctioned_names, convert_to_tensor=True)

    with open(OFAC_EMBEDDINGS_FILE, "wb") as f:
        pickle.dump({"names": sanctioned_names, "embeddings": sanctioned_embeddings}, f)

    logging.info("Embeddings computed & cached.")
    return df, sanctioned_embeddings

def check_sanctions(name, bert_threshold=0.75, fuzzy_threshold=85):
    df, sanctioned_embeddings = get_sanctioned_embeddings()
    if df is None:
        return "Sanctions data not available."

    name_cleaned = clean_name(name)
    sanctioned_names = df["Cleaned Name"].dropna().tolist()

    input_embedding = model.encode(name_cleaned, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(input_embedding, sanctioned_embeddings)[0]

    bert_matches = [
        (df.iloc[i]["Name"], similarity_scores[i].item(), df.iloc[i]["Additional Info"])
        for i in range(len(sanctioned_names))
        if similarity_scores[i] >= bert_threshold
    ]

    fuzzy_matches = [
        (df.iloc[i]["Name"], fuzz.ratio(name_cleaned, sanctioned_names[i]), df.iloc[i]["Additional Info"])
        for i in range(len(sanctioned_names))
        if fuzz.ratio(name_cleaned, sanctioned_names[i]) >= fuzzy_threshold
    ]

    matches = set(bert_matches + fuzzy_matches)

    if matches:
        result = f"{name} is potentially sanctioned:\n"
        for matched_name, score, additional_info in matches:
            if additional_info and additional_info != "-0-":
                result += f"- {matched_name} (Score: {score:.2f}), Additional Info: {additional_info}\n"
            else:
                result += f"- {matched_name} (Score: {score:.2f})\n"
        return result.strip()

    return f"{name} is NOT on the sanctions list."

# Function to Perform Sentiment Analysis on Financial News


def get_financial_news(company_name):
    API_KEY = "9047082950c04406ad8378594370e334"  # Replace with your API key
    search_query = f'"{company_name}"'
    url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()

    if "articles" not in data:
        logging.warning(f"No financial news found for {company_name}")
        return []

    news_list = [
        {
            "title": article["title"], 
            "description": article["description"], 
            "url": article["url"]  # Include the article link
        }
        for article in data["articles"][:20]
    ]

    logging.info(f"Fetched {len(news_list)} news articles for {company_name}")
    print(f"Fetched {len(news_list)} articles")  # Debugging

    # Define expanded risk-related keywords
    risk_keywords = [
        "fraud", "lawsuit", "investigation", "sanctions",
        "indictment", "SEC", "DOJ", "probe", "money laundering",
        "bribery", "corruption", "scandal", "settlement",
        "penalty", "fine", "illegal", "regulatory action",
        "insider trading", "embezzlement", "whistleblower",
        "securities fraud", "criminal charges", "compliance violation",
        "shell company", "offshore", "money laundering", "front company"
    ]

    relevant_news = []
    for article in news_list:
        text = f"{article['title']} {article['description']}" if article["description"] else article["title"]

        # Loose matching: check if company name is in the text
        company_mentioned = company_name.lower() in text.lower() or fuzz.partial_ratio(company_name.lower(), text.lower()) >= 60

        # Check for risk-related keywords using simple lowercase matching
        found_keywords = [word for word in risk_keywords if word in text.lower()]
        contains_risk_keywords = bool(found_keywords)

        # Perform sentiment analysis
        sentiment = sentiment_analyzer(text[:512])[0]["label"].lower()

        # Keep articles that mention the company and contain risk terms
        if company_mentioned and contains_risk_keywords:
            relevant_news.append({
                "title": article["title"],
                "description": article["description"],
                "url": article["url"]  # Adding link to relevant news
            })

    logging.info(f"Filtered {len(relevant_news)} risk-related news articles for {company_name}")
    print(f"✅ Found {len(relevant_news)} relevant articles.")  # Debugging
    return relevant_news


#Funtions to get SEC data

def get_cik_number(company_name):
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        cik_data = response.json()
        for entry in cik_data.values():
            if company_name.lower() in entry["title"].lower():
                return str(entry["cik_str"]).zfill(10)
    return None

def get_sec_filings(cik_number):
    if not cik_number:
        return "Invalid CIK number"

    sec_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik_number}&output=atom"

    response = requests.get(sec_url, headers=HEADERS)
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall(".//atom:entry", ns)

        filings = []
        for entry in entries[:5]:  # Get latest 5 filings
            title_elem = entry.find("atom:title", ns)
            link_elem = entry.find("atom:link", ns)

            if title_elem is not None and link_elem is not None:
                title = title_elem.text
                link = link_elem.attrib.get("href", "No Link Available")
                filings.append({"title": title, "link": link})

        return filings if filings else "No filings found."
    else:
        return f"Error: {response.status_code}, {response.text}"

def get_financial_data(cik, concept):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        return data.get("units", {}).get("USD", [])
    return None

FIN_HUB_API_KEY = FIN_HUB_API
ALPHA_VANTAGE_API_KEY = ALPHA_VANTAGE_API

def get_company_ticker_alpha_vantage(company_name):
    """
    Fetch the stock ticker symbol using Alpha Vantage API with filtering.
    """
    search_url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={ALPHA_VANTAGE_API_KEY}"
    
    response = requests.get(search_url)
    data = response.json()

    if "bestMatches" in data:
        for match in data["bestMatches"]:
            matched_name = match.get("2. name", "").lower()
            symbol = match.get("1. symbol", "")

            # If the matched name is a close match, return the symbol
            if company_name.lower() in matched_name:
                return symbol
    
    return None

def get_company_info(company_name):
    """
    Fetch company details using FinHub API.
    - Uses Alpha Vantage to get the ticker first.
    """
    ticker = get_company_ticker_alpha_vantage(company_name)

    if not ticker:
        return {"error": f"No ticker found for {company_name}"}

    # Get company profile using FinHub API
    profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FIN_HUB_API_KEY}"
    
    response = requests.get(profile_url)
    company_info = response.json()

    if "name" not in company_info:
        return {"error": f"Company details not found for {company_name} ({ticker})"}

    return {
        "Company Name": company_info.get("name"),
        "Ticker": ticker,
        "Country": company_info.get("country", "N/A"),
        "Industry": company_info.get("finnhubIndustry", "N/A"),
        "Market Cap": company_info.get("marketCapitalization", "N/A"),
        "Exchange": company_info.get("exchange", "N/A"),
        "Website": company_info.get("weburl", "N/A"),
    }



def check_tax_haven(entity_country):
    TAX_HAVENS = {
    "BS", "BB", "BM", "KY", "VG", "AI", "AG", "VC", "DM", "MS", "TC",
    "MU", "SC", "LC", "IM", "JE", "GG", "SG", "HK", "MO", "PA", "UY"
}
    return entity_country in TAX_HAVENS

def check_ofc(entity_country):
    OFC_COUNTRIES = {
        "BS", "BB", "BM", "KY", "VG", "AI", "AG", "VC", "DM", "MS", "TC",
        "MU", "SC", "LC", "IM", "JE", "GG", "SG", "HK", "MO", "PA", "UY",
        "MT", "CY", "LI", "AD", "MC"
    }
    return entity_country in OFC_COUNTRIES

SHELL_REGISTRATION_HOTSPOTS = {"VG", "KY", "BM", "SC", "IM", "JE", "GG"}
HIGH_RISK_INDUSTRIES = {
    "Cryptocurrency", "Blockchain", "Forex Trading", "Casinos",
    "Offshore Banking", "Precious Metals Trading", "Private Banking"
}
def check_sanctioned_country(entity_country):
    """
    Checks if the entity's country is in the list of sanctioned countries.

    :param entity_country: Country code of the entity (ISO Alpha-2 format, e.g., 'RU' for Russia)
    :return: True if the country is sanctioned, False otherwise
    """
    SANCTIONED_COUNTRIES = {
        "AF",  # Afghanistan
        "BY",  # Belarus
        "CU",  # Cuba
        "IR",  # Iran
        "KP",  # North Korea
        "RU",  # Russia
        "SD",  # Sudan
        "SY",  # Syria
        "VE",  # Venezuela
    }
    return entity_country in SANCTIONED_COUNTRIES

def check_suspicious_domain(website_url):
    """
    Checks if the company's website uses a suspicious top-level domain (TLD).
    
    :param website_url: Website URL of the company
    :return: True if the domain is suspicious, False otherwise
    """
    SUSPICIOUS_TLDS = {".xyz", ".biz", ".top", ".tk", ".cc", ".ws", ".pw"}
    domain_extension = website_url.split(".")[-1].lower() if website_url else ""
    return f".{domain_extension}" in SUSPICIOUS_TLDS

def check_low_market_cap(market_cap):
    """
    Checks if the company's market cap is very low (potential penny stock or high risk).
    
    :param market_cap: Market capitalization in USD
    :return: True if the company has a very low market cap, False otherwise
    """
    return market_cap != '' and int(market_cap) < 500_000  # Under $0.5M is considered high-risk

import re

def check_generic_name(company_name):
    """
    Checks if the company name is overly generic or contains suspicious patterns.
    
    :param company_name: The name of the company
    :return: True if the name looks suspicious, False otherwise
    """
    GENERIC_PATTERNS = [
        r"global\s+trading", r"international\s+business", r"enterprise\s+group",
        r"holding\s+company", r"\b[a-zA-Z]*\d{3,}\b"  # Random numbers in name
    ]
    return any(re.search(pattern, company_name, re.IGNORECASE) for pattern in GENERIC_PATTERNS)




def generate_risk_report(company_name):
    cik = get_cik_number(company_name)
    sec_filings = get_sec_filings(cik) if cik else "CIK not found."
    sanctions_result = check_sanctions(company_name)
    company_info = get_company_info(company_name)
    financial_news = get_financial_news(company_name)
    entity_country = company_info.get('Country', 'N/A')
    
    # Format SEC Filings
    formatted_sec_filings = "\n".join([
        f"- {filing['title']} (🔗 {filing['link']})" for filing in sec_filings
    ]) if isinstance(sec_filings, list) else sec_filings
    
    # Extract key financial risk factors
    identified_risk_factors = "\n".join([
        f"- {article['title']} (🔗 {article['url']})" for article in financial_news
    ]) if financial_news else "No high-risk news detected."
    
    # Compile Report
    report = f"""
    # 🏢 ENTITY RISK REPORT

    ## 1️⃣ Entity Details
    - **Company Name:** {company_info.get('Company Name', 'N/A')}
    - **CIK Number:** {cik if cik else "Not found"}
    - **Ticker Symbol:** {company_info.get('Ticker', 'Not found')}
    - **Country:** {company_info.get('Country', 'N/A')}
    - **Industry:** {company_info.get('Industry', 'N/A')}
    - **Market Cap:** {company_info.get('Market Cap', 'N/A')}
    - **Exchange:** {company_info.get('Exchange', 'N/A')}
    - **Website:** {company_info.get('Website', 'N/A')}

    ---

    ## 2️⃣ 🚨 **Sanctions Check**
    **Sanctioned?** {sanctions_result}

    ---

    ## 3️⃣ 📑 **SEC Filings & Risk Analysis**
    ### 🔹 Recent Filings:
    {formatted_sec_filings}

    ---

    ## 4️⃣ 📰 **Financial News & Risk Factors**
    ### 🔹 Key News Articles:
    {identified_risk_factors}

    ---

    ## 5️⃣ 📊 **Final Summary**
    """
    
    # Summarization for LLAMA model prompt
    summary_text = summarizer(report[:1024], max_length=512, min_length=50, do_sample=False)
    report += f"{summary_text[0]['summary_text']}"
    if check_tax_haven(entity_country):
        report+="\n ⚠️ Entity is located in a tax haven!"
    if check_ofc(entity_country):
        report+="\n ⚠️ Entity is located in a Offshore Financial Center (OFC)!"
    if entity_country in SHELL_REGISTRATION_HOTSPOTS:
        report += "\n ⚠️ This company is registered in a known shell company hotspot!"
    if company_info.get("Industry", "N/A") in HIGH_RISK_INDUSTRIES:
        report += "\n ⚠️ Company operates in a high-risk industry!"
    if "government" in company_info.get("Industry", "").lower():
        report += "\n ⚠️ Company is involved in government contracts—heightened corruption risk!"
    if check_sanctioned_country(entity_country):
        report += "\n ⚠️ Country is under international sanctions! High-risk entity."
    if check_suspicious_domain(company_info.get("Website", "")):
        report += "\n ⚠️ Suspicious top-level domain (TLD):" + str({company_info.get('Website', '')})
    if check_low_market_cap(company_info.get('Market Cap', '')):
        report += "\n ⚠️ Market Cap is very low! High-risk entity."
    if check_generic_name(company_name):
        report += "\n ⚠️ Company name is overly generic or contains suspicious patterns."


    
    return report
