import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import gradio as gr


# Load API key
load_dotenv()
NEBIUS_KEY = os.getenv("NEBIUS_API_KEY")

# Nebius client (OpenAI-compatible)
client = OpenAI(
    api_key=NEBIUS_KEY,
    base_url="https://api.studio.nebius.ai/v1/"
)

# Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Vector DB
# NEW CHROMA CLIENT (correct)
db = chromadb.PersistentClient(path="vector_db")

collection = db.get_or_create_collection(
    name="occultum",
    metadata={"hnsw:space": "cosine"}
)


# LLM function
def cast_spell(context, query):
    prompt = f"""
You are **OccultumAI**, an ancient magical spellbook.
Use the scrolls below to answer the apprentice.

Scrolls:
{context}

User's Question: {query}

Answer in a mystical, wizard-like tone:
"""

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# RAG pipeline
def ask_occultum(query):
    # retrieve using embeddings
    results = collection.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=2
    )
    
    # combine all retrieved documents
    context = "\n\n".join(results["documents"][0])
    
    # generate answer
    answer = cast_spell(context, query)
    return answer


# ... (rest of your code remains the same until the Gradio UI section) ...

# Gradio UI with Enhanced Aesthetics
def interface(query):
    # This is your existing function to call the RAG pipeline
    return ask_occultum(query)

custom_css = """
/* General body styling for a deep, mystical background */
body {
    background-color: #0d0d1a; /* Even darker blue/purple for deep space/void feel */
    color: #e0e0e0; /* Lighter text for contrast */
    font-family: 'Times New Roman', serif; /* Classic, old-book font */
    /* Grimoire texture background */
    background-image: url('https://www.transparenttextures.com/patterns/dark-bark.png'); /* A subtle dark texture */
    background-attachment: fixed;
    background-size: cover;
}

/* Overall Gradio container styling */
.gradio-container {
    background-color: rgba(20, 20, 40, 0.85); /* Slightly transparent dark panel */
    border: 3px solid #4a0072; /* Thicker, magical purple border */
    border-radius: 15px;
    box-shadow: 0 0 30px rgba(74, 0, 114, 0.7); /* Stronger purple glow */
    padding: 25px;
    margin-top: 30px;
    margin-bottom: 30px;
}

/* Main title - even more enchanted */
.gradio-container h1 {
    color: #ffd700; /* Gold */
    text-shadow: 0 0 15px rgba(255, 215, 0, 0.8), 0 0 5px rgba(255, 215, 0, 0.5); /* Enhanced golden glow */
    font-size: 3em; /* Larger title */
    letter-spacing: 2px;
    text-align: center;
    margin-bottom: 20px;
    /* Optional: A magical shimmering effect for the title */
    /* animation: shimmer 2s infinite alternate; */
}

/* Keyframe for shimmering effect if enabled */
/*
@keyframes shimmer {
    from { text-shadow: 0 0 15px rgba(255, 215, 0, 0.8); }
    to { text-shadow: 0 0 25px rgba(255, 215, 0, 1), 0 0 8px rgba(255, 215, 0, 0.7); }
}
*/

/* Description text */
.gradio-container p {
    color: #b0b0b0; /* Slightly desaturated silver */
    font-style: italic;
    text-align: center;
    margin-bottom: 30px;
}

/* Input and Output textboxes */
.gr-textbox {
    border: 2px solid #6a0dad !important; /* Deeper magical purple border */
    border-radius: 10px;
    background-color: #1e1e3f; /* Darker background for mystical input */
    color: #f0f0f0;
    padding: 10px;
}

/* Styling the output area to look like an ancient scroll */
#occultum-output-box .gr-textbox-body {
    background-color: #fdf5e6; /* Antique white for 'scroll' paper */
    color: #5c4033; /* Darker sepia text */
    border: 5px double #8b4513; /* Double border for an ornate look */
    border-radius: 5px; /* Slightly rounded edges for scroll */
    padding: 25px;
    box-shadow: 8px 8px 15px rgba(0, 0, 0, 0.5), inset 0 0 10px rgba(0, 0, 0, 0.2); /* Stronger shadow with inner glow */
    line-height: 1.6; /* Better readability for long text */
    font-size: 1.1em;
    text-align: justify;
    background-image: url('https://www.transparenttextures.com/patterns/parchment.png'); /* Subtle parchment texture */
}

/* Styling the submit button (The 'Spell Casting' button) */
.gr-button-primary {
    background: linear-gradient(145deg, #8a2be2, #4b0082) !important; /* Deeper violet gradient */
    color: #ffffff !important;
    font-weight: bold;
    font-size: 1.2em;
    border-radius: 25px; /* More prominent gem-like button */
    padding: 12px 25px;
    transition: all 0.4s ease;
    border: none;
    box-shadow: 0 5px 15px rgba(138, 43, 226, 0.6); /* Persistent glow */
}

/* Hover effect for the button with a stronger magical pulse */
.gr-button-primary:hover {
    box-shadow: 0 0 20px #9932cc, 0 0 30px #da70d6; /* Brighter, multi-color glow */
    transform: translateY(-3px) scale(1.02);
    cursor: pointer;
}

/* Clear and Flag buttons - more subtle */
.gr-button {
    background-color: #3a3a5e !important;
    color: #cccccc !important;
    border: 1px solid #5a5a8e !important;
    border-radius: 10px;
    transition: background-color 0.3s ease;
}

.gr-button:hover {
    background-color: #5a5a8e !important;
}

/* Custom sparkle effect for labels (using a unicode character for simplicity) */
label.gr-text-input > span:first-child::before {
    content: '✨ '; /* Sparkle emoji */
    color: #ffcc00; /* Golden sparkle color */
}

label.gr-text-input > span:first-child {
    color: #ffd700; /* Gold label text */
    font-weight: bold;
    font-size: 1.1em;
}

/* Adjusting the scrollbar for a darker theme */
::-webkit-scrollbar {
    width: 12px;
}

::-webkit-scrollbar-track {
    background: #2a2a4a; /* Darker track */
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #6a0dad; /* Magical purple thumb */
    border-radius: 10px;
    border: 3px solid #2a2a4a; /* Padding around thumb */
}

::-webkit-scrollbar-thumb:hover {
    background: #8a2be2; /* Lighter purple on hover */
}
"""

# Enhanced Gradio Interface Setup
ui = gr.Interface(
    fn=interface,
    title="🪄 OccultumAI — The Ancient Spellbook",
    description="Ask your wizardly questions and observe the cosmic response...",
    inputs=gr.Textbox(label="🔍 Whisper Your Query into the Void (Enter Spell):"),
    outputs=gr.Textbox(label="📜 Occultum's Magical Scroll Unfurls:", elem_id="occultum-output-box",lines=15),
    
    # 1. Use a theme that is naturally darker or more complementary
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="stone"
    ),
    # 2. Apply the custom CSS
    css=custom_css 
)


if __name__ == "__main__":
  ui.launch()
