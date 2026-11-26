
import json
import re
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
ABSTAIN_THRESHOLD = 0.35

app = FastAPI(title="RAG Colosseum API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=TOP_K, ge=1, le=20)


# --------------------------------------------------
# Load knowledge base and embedding model
# --------------------------------------------------

if not KB_PATH.exists():
    raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

with open(KB_PATH, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

if not isinstance(knowledge_base, list):
    raise ValueError("knowledge_base.json must contain a list.")

if not all(isinstance(item, str) for item in knowledge_base):
    raise ValueError("Every knowledge-base passage must be a string.")

if not knowledge_base:
    raise ValueError("Knowledge base is empty.")

print(f"Loaded {len(knowledge_base)} passages.")

model = SentenceTransformer(MODEL_NAME, device="cpu")

embeddings = model.encode(
    knowledge_base,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
)

embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("FAISS index created successfully.")


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def retrieve(question: str, top_k: int):
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    k = min(top_k, len(knowledge_base))

    scores, indices = index.search(query_embedding, k)

    retrieved = []

    for score, passage_id in zip(scores[0], indices[0]):
        if passage_id < 0:
            continue

        retrieved.append({
            "passage_id": int(passage_id),
            "text": knowledge_base[int(passage_id)],
            "score": float(score),
        })

    return retrieved


# --------------------------------------------------
# Answer extraction
# --------------------------------------------------

def extract_answer(question: str, retrieved: list):
    if not retrieved:
        return None, [], [], 0.0

    sentences = []

    for item in retrieved:
        passage_sentences = re.split(
            r"(?<=[.!?])\s+",
            item["text"].strip(),
        )

        for sentence in passage_sentences:
            sentence = sentence.strip()

            if sentence:
                sentences.append({
                    "text": sentence,
                    "passage_id": item["passage_id"],
                })

    if not sentences:
        return None, [], [], 0.0

    sentence_texts = [
        item["text"] for item in sentences
    ]

    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarities = np.dot(
        sentence_embeddings,
        question_embedding[0],
    )

    best_index = int(np.argmax(similarities))
    best_sentence = sentences[best_index]
    best_sentence_score = float(similarities[best_index])

    best_passage_id = best_sentence["passage_id"]

    # Confidence calculation
    confidence = float(
        np.clip(
            (best_sentence_score - 0.20) / 0.60,
            0.0,
            1.0,
        )
    )

    # Abstain when confidence is below the threshold
    if confidence < ABSTAIN_THRESHOLD:
        return None, [], [], round(confidence, 4)

    # Cite the passage that actually contains the answer
    best_passage = next(
        item for item in retrieved
        if item["passage_id"] == best_passage_id
    )

    source_ids = [best_passage["passage_id"]]
    source_scores = [round(best_passage["score"], 4)]

    return (
        best_sentence["text"],
        source_ids,
        source_scores,
        round(confidence, 4),
    )


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "RAG Colosseum API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "passages": len(knowledge_base),
        "model": MODEL_NAME,
    }


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        retrieved = await run_in_threadpool(
            retrieve,
            question,
            request.top_k,
        )

        answer, sources, scores, confidence = await run_in_threadpool(
            extract_answer,
            question,
            retrieved,
        )

        return {
            "answer": answer,
            "sources": sources,
            "scores": scores,
            "confidence": confidence,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

