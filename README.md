# OccultumAI-
GenAI project using Retrieval-Augmented Generation (RAG) with Large Language Models

OccultumAI — Spellbook AI
OccultumAI is a fantasy-themed Retrieval-Augmented Generation (RAG) assistant that behaves like a wizard’s spellbook.
It retrieves knowledge from a text-based grimoire and uses a Large Language Model (LLM) to generate grounded, explainable answers(while keeping everything framed a mystical tone).

Knowledge Base (The Grimoire)
The system uses a collection of plain text files as its knowledge source.
Each file represents a set of “spells” written in a wizard-like tone but containing real information.
These files contain factual explanations about topics such as dark stars, black holes, and quantum physics.
They are embedded and stored in a vector database for semantic search.

How It ACTUALLY Works
1.Text files are split into small chunks.
2.Each chunk is converted into an embedding.
3.Embeddings are stored in a vector database (Chroma).
4.User queries are embedded and matched against stored spells.
5.Relevant spells(/data) are retrieved and passed to an LLM.
6.The LLM generates an answer grounded in the retrieved knowledge.

The wizard-like personality comes from prompt design, while factual accuracy comes from retrieval.

Tech Stack:
Python
sentence-transformers (MiniLM)
ChromaDB
GPT-4
Gradio (UI)


A magical theme built on serious GenAI architecture.

If you want an even shorter version (resume-friendly) or a one-paragraph project description, I can condense it further.
