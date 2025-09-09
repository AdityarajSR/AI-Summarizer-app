from keybert import KeyBERT
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="AI Notes Summarizer")
summarizer = pipeline("summarization")
kw_model = KeyBERT()

# Request model
class SummarizeRequest(BaseModel):
    text: str
    summaryLength: str  # "short", "medium", "long"

# Response model
class SummarizeResponse(BaseModel):
    summary: str
    keywords: List[str]

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    text = request.text
    length = request.summaryLength

    # Placeholder summarization
    # summary = text[: min(len(text), 200)]  # first 200 characters

    summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summary = summary_result[0]['summary_text']

    # Placeholder keywords
    # keywords = ["AI", "notes", "summarizer"]
    
    # Real keywords
    keywords_raw = kw_model.extract_keywords(text, top_n=5)
    keywords = [kw[0] for kw in keywords_raw]


    return SummarizeResponse(summary=summary, keywords=keywords)
