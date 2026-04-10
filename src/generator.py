"""
Answer Generator with Citations
Takes retrieved context chunks and a question, sends to GPT-4o,
and returns an answer with mapped source citations.
"""
 
import os
import json
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache
 
load_dotenv()

try:
    import streamlit as st
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)
 
SYSTEM_PROMPT = """You are a legal contract analyst. Your role is to answer
questions about contracts based ONLY on the provided context excerpts.
 
Rules:
- Answer ONLY based on the provided context. If the context does not contain
  enough information to answer, say so explicitly.
- ALWAYS cite your sources using [Source N] notation.
- Every factual claim must have at least one citation.
- Be precise with legal terminology.
- If multiple sources provide different information, note the discrepancy.
- Never invent or assume information not present in the context.
 
Respond in JSON format:
{
    "answer": "Your detailed answer with [Source 1], [Source 2] citations...",
    "cited_sources": [1, 2, 3],
    "confidence": "high" | "medium" | "low",
    "confidence_reason": "Brief explanation of your confidence level"
}"""
 
 
def format_context(retrieved_chunks: list) -> tuple[str, dict]:
    """
    Format retrieved chunks into labeled context for the prompt.
    Returns the formatted string and a source map for citation resolution.
    """
    context_parts = []
    source_map = {}
 
    for i, chunk in enumerate(retrieved_chunks):
        source_num = i + 1
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
 
        context_parts.append(
            f"[Source {source_num}] (From: {source}, Page: {page})\n"
            f"{chunk['content']}\n"
        )
 
        source_map[source_num] = {
            "source": source,
            "page": page,
            "chunk_index": chunk["metadata"].get("chunk_index", "N/A"),
            "relevance_score": chunk.get("rrf_score", chunk.get("score", 0)),
        }
 
    return "\n---\n".join(context_parts), source_map
 

# def generate_answer(query: str, retrieved_chunks: list) -> dict:
#     """
#     Generate an answer using GPT-4o with the retrieved context.
#     Returns:
#         dict with answer, citations, confidence, and source details.
#     """
#     if not retrieved_chunks:
#         return {
#             "answer": "No relevant context found. Try rephrasing your question.",
#             "citations": [],
#             "confidence": "low",
#         }
 
#     context_text, source_map = format_context(retrieved_chunks)
 
#     user_prompt = f"""Context excerpts from the contract(s):
 
# {context_text}
 
# ---
# Question: {query}
 
# Analyze the context carefully and provide your answer in JSON format."""
 
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             max_tokens=2048,
#             temperature=0.3,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": user_prompt}
#             ]
#         )
 
#         raw_text = response.choices[0].message.content
 
#         # Parse JSON response
#         text = raw_text.strip()
#         if text.startswith("```json"): text = text[7:]
#         elif text.startswith("```"): text = text[3:]
#         if text.endswith("```"): text = text[:-3]
#         text = text.strip()
 
#         parsed = json.loads(text)
 
#         # Map cited source numbers to actual document metadata
#         citations = []
#         for src_num in parsed.get("cited_sources", []):
#             if src_num in source_map:
#                 citations.append(source_map[src_num])
 
#         return {
#             "answer": parsed.get("answer", "No answer generated."),
#             "citations": citations,
#             "confidence": parsed.get("confidence", "unknown"),
#             "confidence_reason": parsed.get("confidence_reason", ""),
#             "source_map": source_map,
#             "input_tokens": response.usage.prompt_tokens,
#             "output_tokens": response.usage.completion_tokens,
#         }
 
#     except json.JSONDecodeError:
#         return {"answer": raw_text, "citations": [], "confidence": "low"}
 
#     except Exception as e:
#         return {"answer": f"Error: {str(e)}", "citations": [], "confidence": "low"}
# 

@lru_cache(maxsize=100)
def _cached_api_call(query: str, context_hash: str, context_text: str) -> str:
    """Cached LLM call. Returns raw response text."""
    user_prompt = f"""Context excerpts from the contract(s):

{context_text}

---
Question: {query}

Analyze the context carefully and provide your answer in JSON format."""

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2048,
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens


def generate_answer(query: str, retrieved_chunks: list) -> dict:
    """
    Generate an answer using GPT-4o with the retrieved context.
    Uses LRU cache to avoid duplicate API calls.
    """
    if not retrieved_chunks:
        return {
            "answer": "No relevant context found. Try rephrasing your question.",
            "citations": [],
            "confidence": "low",
        }

    context_text, source_map = format_context(retrieved_chunks)

    # Hash the context so cache invalidates when documents change
    context_hash = hashlib.md5(context_text.encode()).hexdigest()

    try:
        raw_text, input_tokens, output_tokens = _cached_api_call(query, context_hash, context_text)

        # Parse JSON response
        text = raw_text.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        text = text.strip()

        parsed = json.loads(text)

        citations = []
        for src_num in parsed.get("cited_sources", []):
            if src_num in source_map:
                citations.append(source_map[src_num])

        return {
            "answer": parsed.get("answer", "No answer generated."),
            "citations": citations,
            "confidence": parsed.get("confidence", "unknown"),
            "confidence_reason": parsed.get("confidence_reason", ""),
            "source_map": source_map,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    except json.JSONDecodeError:
        return {"answer": raw_text, "citations": [], "confidence": "low"}

    except Exception as e:
        return {"answer": f"Error: {str(e)}", "citations": [], "confidence": "low"}