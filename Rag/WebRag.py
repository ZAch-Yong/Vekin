#using google custom search API
import os
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

# Your API credentials (replace these with yours)
API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not API_KEY or not CSE_ID:
    raise EnvironmentError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variables.")

# Load text generation model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Function to perform a web search using Google Programmable Search
def google_search(query, api_key, cse_id, num=10): #can increase upto 10 to fetch more
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return [item['link'] for item in res.get('items', [])]

# Function to scrape and extract content from a webpage
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:4000]  # Limit to prevent overly long input
    except Exception as e:
        print(f"Failed to extract from {url}: {e}")
        return ""

# --- Simple keyword checker ---
def keyword_checker(output, expected_keywords):
    return all(keyword.lower() in output.lower() for keyword in expected_keywords)

# --- LLM-based explainable checker ---
def llm_checker(output, question, reference):
    prompt = f"""
    Question: {question}
    System Answer: {output}
    Reference: {reference}
    Is the system's answer correct? Answer "Yes" or "No" and explain briefly.
    """
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return result.strip()

# --- Main RAG function with checkers ---
def web_rag_with_checkers(query, expected_keywords=None, reference_answer=None):
    try:
        urls = google_search(query, API_KEY, CSE_ID)
        context_parts = [extract_text_from_url(url) for url in urls]
        context = " ".join(context_parts)
        prompt = f"Context: {context} Question: {query}"
        result = generator(prompt, max_new_tokens=100)[0]['generated_text']

        # Run keyword check
        keyword_check = None
        if expected_keywords:
            keyword_check = keyword_checker(result, expected_keywords)

        # Run explainable evaluation
        explainable_check = None
        if reference_answer:
            explainable_check = llm_checker(result, query, reference_answer)

        return {
            "answer": result,
            "keyword_check_passed": keyword_check,
            "llm_verification": explainable_check
        }

    except Exception as e:
        return {"error": f"Error during Web RAG: {e}"}

# --- Example Usage ---
query = "who is Vekin Chief Operating Officer?"
expected_keywords = ["Vekin", "Cheif Operating Officer"]
reference_answer = "The Chief Operating Officer of Vekin is John Doe."  # Replace with verified info

response = web_rag_with_checkers(query, expected_keywords, reference_answer)

# Output
print("Answer:", response["answer"])
print("Keyword Check Passed:", response["keyword_check_passed"])
print("LLM Verification:", response["llm_verification"])