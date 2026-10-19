"""
Customer Support Agentic AI (CrewAI)

Agent 1 -> searches the RAG file (PDF)
Agent 2 -> searches the web ONLY if Agent 1 found nothing
Agent 3 -> logs the question, answer and source to Google Sheets
"""
import os
import json
import hashlib
import datetime

import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
import gspread
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RAG_FILE = os.getenv("RAG_FILE", "knowledge_base.pdf")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
TOP_K = 4
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
GOOGLE_CREDS = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")

llm = LLM(model=CHAT_MODEL, temperature=0.2)


# ----------------------------------------------------------------------------
# Tool 1: RAG search over the PDF (HuggingFace embeddings + FAISS)
# ----------------------------------------------------------------------------
_model = None
_index_cache = {}


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _embed(texts):
    vecs = _get_model().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")


def _chunk(text: str, size: int = 800, overlap: int = 150):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def _build_index(path: str):
    """Build (or load cached) FAISS index + chunk metadata for the PDF."""
    with open(path, "rb") as f:
        digest = hashlib.md5(f.read() + EMBED_MODEL.encode()).hexdigest()
    if digest in _index_cache:
        return _index_cache[digest]

    idx_file, meta_file = f".rag_cache_{digest}.faiss", f".rag_cache_{digest}.json"
    if os.path.exists(idx_file) and os.path.exists(meta_file):
        index, meta = faiss.read_index(idx_file), json.load(open(meta_file))
    else:
        chunks, meta = [], []
        for page_no, page in enumerate(PdfReader(path).pages, start=1):
            for c in _chunk(page.extract_text() or ""):
                chunks.append(c)
                meta.append({"page": page_no, "text": c})
        vecs = _embed(chunks)
        index = faiss.IndexFlatIP(vecs.shape[1])  # inner product == cosine (normalized vectors)
        index.add(vecs)
        faiss.write_index(index, idx_file)
        json.dump(meta, open(meta_file, "w"))

    _index_cache[digest] = (index, meta)
    return index, meta


class RagInput(BaseModel):
    query: str = Field(..., description="The customer's question")


class RagSearchTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = (
        "Searches the company knowledge-base PDF. Returns 'STATUS: FOUND' with "
        "relevant passages, or 'STATUS: NOT_FOUND' if nothing relevant exists."
    )
    args_schema: type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        index, meta = _build_index(RAG_FILE)
        scores, ids = index.search(_embed([query]), TOP_K)
        hits = [
            (float(s), meta[i])
            for s, i in zip(scores[0], ids[0])
            if i != -1 and s >= SIMILARITY_THRESHOLD
        ]
        if not hits:
            return "STATUS: NOT_FOUND"
        body = "\n\n".join(f"[page {m['page']} | score {s:.2f}]\n{m['text']}" for s, m in hits)
        return f"STATUS: FOUND\n\n{body}"


# ----------------------------------------------------------------------------
# Tool 2: Web search
# ----------------------------------------------------------------------------
class WebInput(BaseModel):
    query: str = Field(..., description="Search query")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Searches the web and returns the top results (title, snippet, url)."
    args_schema: type[BaseModel] = WebInput

    def _run(self, query: str) -> str:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No web results found."
        return "\n\n".join(f"{r['title']}\n{r['body']}\n{r['href']}" for r in results)


# ----------------------------------------------------------------------------
# Tool 3: Google Sheets logger
# ----------------------------------------------------------------------------
class LogInput(BaseModel):
    customer_name: str = Field(..., description="Customer name as given by the customer")
    customer_contact: str = Field(..., description="Phone or email as given by the customer")
    question: str = Field(..., description="The customer's original question")
    source: str = Field(..., description="RAG, WEB or NONE")
    answer: str = Field(..., description="The final answer given to the customer")


class GoogleSheetLogTool(BaseTool):
    name: str = "log_to_google_sheet"
    description: str = "Appends a row (timestamp, customer name, contact, question, source, answer) to the Google Sheet log."
    args_schema: type[BaseModel] = LogInput

    def _run(self, customer_name: str, customer_contact: str, question: str, source: str, answer: str) -> str:
        gc = gspread.service_account(filename=GOOGLE_CREDS)
        ws = gc.open_by_key(SHEET_ID).sheet1
        if not ws.get_all_values():
            ws.append_row(["Timestamp", "Customer Name", "Contact", "Question", "Source", "Answer"])
        ws.append_row([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            customer_name, customer_contact, question, source, answer[:45000],
        ])
        return "Logged successfully."


# ----------------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------------
rag_agent = Agent(
    role="Knowledge Base Specialist",
    goal="Answer customer questions strictly from the company knowledge-base PDF.",
    backstory="You only trust the internal documentation. You never guess or use outside knowledge.",
    tools=[RagSearchTool()],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

web_agent = Agent(
    role="Web Research Specialist",
    goal="Find an accurate answer on the web, but only when the knowledge base has no answer.",
    backstory="You are the fallback. If the knowledge base already answered, you do nothing.",
    tools=[WebSearchTool()],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)

logger_agent = Agent(
    role="Support Ticket Logger",
    goal="Record every customer question, its answer and the answer source in Google Sheets.",
    backstory="You keep accurate support records and deliver the final reply to the customer.",
    tools=[GoogleSheetLogTool()],
    llm=llm,
    allow_delegation=False,
    verbose=True,
)


# ----------------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------------
rag_task = Task(
    description=(
        "Customer question: {question}\n\n"
        "Use the knowledge_base_search tool. If it returns STATUS: FOUND, write a clear, "
        "friendly answer using ONLY the retrieved text. If NOT_FOUND, do not answer."
    ),
    expected_output=(
        "Exactly this format:\nSTATUS: FOUND or NOT_FOUND\nANSWER: <answer, or 'none'>"
    ),
    agent=rag_agent,
)

web_task = Task(
    description=(
        "Customer question: {question}\n\n"
        "Look at the previous task's STATUS.\n"
        "- If STATUS is FOUND: do NOT use any tool. Repeat the same answer with SOURCE: RAG.\n"
        "- If STATUS is NOT_FOUND: use web_search, then write a concise, accurate answer from "
        "the results and mention the main source URL. Use SOURCE: WEB. If the web has nothing "
        "useful, use SOURCE: NONE and apologise politely."
    ),
    expected_output="Exactly this format:\nSOURCE: RAG or WEB or NONE\nANSWER: <final answer>",
    agent=web_agent,
    context=[rag_task],
)

log_task = Task(
    description=(
        "Customer name: {customer_name}\nCustomer contact: {customer_contact}\n"
        "Customer question: {question}\n\n"
        "Take the SOURCE and ANSWER from the previous task and call log_to_google_sheet with "
        "the customer name, contact, original question, source and answer. Then output the "
        "final reply for the customer."
    ),
    expected_output="The final customer-facing answer only (no SOURCE/STATUS labels).",
    agent=logger_agent,
    context=[web_task],
)


# ----------------------------------------------------------------------------
# Crew
# ----------------------------------------------------------------------------
crew = Crew(
    agents=[rag_agent, web_agent, logger_agent],
    tasks=[rag_task, web_task, log_task],
    process=Process.sequential,
    verbose=True,
)


def ask(question: str, customer_name: str = "Unknown", customer_contact: str = "Not provided") -> str:
    return str(crew.kickoff(inputs={
        "question": question,
        "customer_name": customer_name,
        "customer_contact": customer_contact,
    }))


if __name__ == "__main__":
    print("Customer Support AI ready. Type 'exit' to quit.\n")
    print("To help you and keep a record of your request, we save your name and contact "
          "details together with your question. Leave a field blank to skip it.\n")
    name = input("Your name: ").strip() or "Unknown"
    contact = input("Your phone or email: ").strip() or "Not provided"
    while True:
        q = input("\nCustomer: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(f"\nSupport: {ask(q, name, contact)}")