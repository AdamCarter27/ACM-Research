import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY")

MODEL = "meta-llama/llama-3.3-70b-instruct"

# local knowledge base
documents = [
    {
        "title": "Interpretability",
        "text": 
        (
            "Model interpretability concerns understanding why "
            "a machine learning system produces particular outputs. "
            "Post-hoc explanations attempt to explain a model's "
            "behavior after it has been trained."
        )
    },
    {
        "title": "AI Alignment",
        "text": 
        (
            "AI alignment is the effort to make AI systems behave "
            "according to intended human goals and values. "
            "Misalignment occurs when a system optimizes an "
            "objective that differs from the intended outcome."
        )
    },
    {
        "title": "LIME",
        "text": 
        (
            "LIME stands for Local Interpretable Model-agnostic "
            "Explanations. It explains an individual prediction "
            "by perturbing an input, observing the black-box "
            "model's predictions, and fitting a simple interpretable "
            "model weighted toward nearby examples."
        )
    },
    {
        "title": "Robustness",
        "text": 
        (
            "Model robustness is the ability to maintain reliable "
            "performance under changes or disturbances in the "
            "input. Adversarial examples are specially constructed "
            "inputs that can cause incorrect model predictions."
        )
    }
]

# Build a TF-IDF index
texts = [doc["text"] for doc in documents]
vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(texts)


def retrieve(query, k=2):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(
        query_vector, document_vectors
    )[0]

    top_indices = scores.argsort()[::-1][:k]

    return [
        documents[i] for i in top_indices
        if scores[i] > 0
    ]


def ask_model(question, context=None):
    if context:
        system_prompt = (
            "Answer using the provided context. "
            "If the context is insufficient, say so. "
            "Do not invent information."
        )
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        system_prompt = "Answer the question clearly and concisely."
        user_prompt = question

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=
        {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 250
        },
        timeout=60
    )

    if not response.ok:
        raise RuntimeError(response.text)

    return response.json()["choices"][0]["message"]["content"]


questions = [
    "How does LIME explain a model prediction?",
    "What is the difference between alignment and robustness?",
    "What are adversarial examples?"
]

for question in questions:
    retrieved_docs = retrieve(question)

    context = "\n\n".join(
        f"{doc['title']}: {doc['text']}"
        for doc in retrieved_docs
    )

    print(f"\nQUESTION: {question}")
    print("RETRIEVED:", [d["title"] for d in retrieved_docs])

    print("\nWITHOUT RAG:")
    print(ask_model(question))

    print("\nWITH RAG:")
    print(ask_model(question, context))
    print("-" * 60)
