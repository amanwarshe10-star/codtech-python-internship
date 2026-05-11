"""
================================================================================
CODTECH INTERNSHIP - TASK 3
AI Chatbot with NLP
--------------------------------------------------------------------------------
Description : NLP chatbot using TF-IDF cosine similarity for knowledge-base
              retrieval, plus intent detection for greetings, farewells, etc.
              Uses scikit-learn for vectorization (no external NLTK downloads).
Libraries   : scikit-learn, re, numpy
Run         : python task3_nlp_chatbot.py          # demo
              python task3_nlp_chatbot.py --chat   # interactive
================================================================================
"""

import re
import string
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. KNOWLEDGE BASE ─────────────────────────────────────────────────────────
KNOWLEDGE_BASE = """
Python is a high-level interpreted programming language known for its simplicity and readability.
It was created by Guido van Rossum and first released in 1991.
Python supports procedural, object-oriented, and functional programming paradigms.
Python is widely used in web development, data science, AI, machine learning, and automation.

Machine learning is a subset of artificial intelligence where systems learn from data without explicit programming.
Popular algorithms include linear regression, decision trees, random forests, SVMs, and neural networks.
Scikit-learn is the primary Python library for machine learning tasks.

Data science involves extracting insights from large datasets using statistics, programming, and domain knowledge.
Common Python data science libraries include pandas, numpy, matplotlib, seaborn, and scikit-learn.
Data scientists perform tasks like data cleaning, EDA, feature engineering, and model building.

Artificial intelligence simulates human intelligence in machines, including learning, reasoning, and problem-solving.
AI subfields include natural language processing, computer vision, robotics, and expert systems.
Deep learning uses neural networks with many hidden layers to solve complex problems.

Natural language processing (NLP) enables computers to understand and generate human language.
NLP tasks include tokenization, stemming, lemmatization, named entity recognition, and sentiment analysis.
Popular NLP libraries are NLTK, spaCy, Gensim, and Hugging Face Transformers.

A chatbot is software that simulates conversation with humans using rule-based or AI-powered methods.
Chatbots are used in customer service, healthcare, education, and entertainment applications.
Modern chatbots use transformer models like BERT, GPT, and T5 for better understanding.

Scikit-learn is a free machine learning library for Python built on NumPy and SciPy.
It provides tools for classification, regression, clustering, dimensionality reduction, and model evaluation.
Common scikit-learn classes include train_test_split, TfidfVectorizer, and LogisticRegression.

Pandas is a Python library for data manipulation and analysis.
It provides DataFrame and Series data structures for handling tabular and time-series data.
Pandas supports reading CSV, Excel, JSON, and SQL data sources.

NumPy is the fundamental Python package for numerical and scientific computing.
It adds support for large multi-dimensional arrays and high-level mathematical functions.
NumPy operations are vectorized, making them much faster than standard Python loops.

GitHub is a web-based platform for Git version control and software collaboration.
Developers use GitHub to store code repositories, track issues, and collaborate via pull requests.
GitHub Actions enables CI/CD workflows for automated testing and deployment.
"""

# ── 2. TEXT PREPROCESSING ─────────────────────────────────────────────────────
# Hardcoded common English stop words (no NLTK download needed)
STOP_WORDS = {
    "a","an","the","is","it","in","on","at","to","for","of","and","or","but",
    "are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","shall","can","not","no","nor",
    "this","that","these","those","with","from","by","as","so","if","then",
    "there","their","they","them","its","which","who","what","how","when","where",
    "i","me","my","we","our","you","your","he","his","she","her","it","its",
    "about","more","also","than","such","into","out","up","down","over","under",
}

def preprocess(text: str) -> str:
    """Lowercase, remove punctuation, remove stop words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def sentence_tokenize(text: str) -> list:
    """Simple sentence splitter using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# Preprocess the knowledge base into sentences
KB_SENTENCES = sentence_tokenize(KNOWLEDGE_BASE)

# ── 3. TF-IDF RESPONSE RETRIEVAL ─────────────────────────────────────────────
def get_best_response(user_input: str) -> str:
    """Return the most relevant sentence from the knowledge base."""
    clean_input = preprocess(user_input)
    clean_kb    = [preprocess(s) for s in KB_SENTENCES]

    corpus = clean_kb + [clean_input]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_idx   = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score < 0.1:
        return None
    return KB_SENTENCES[best_idx]


# ── 4. INTENT PATTERNS ────────────────────────────────────────────────────────
INTENTS = {
    "greeting": {
        "patterns": ["hello","hi","hey","howdy","sup","greetings","namaste",
                     "good morning","good evening","good afternoon"],
        "responses": ["Hello! How can I help you today?",
                      "Hi there! Ask me anything about Python or Data Science.",
                      "Hey! What would you like to know?"]
    },
    "farewell": {
        "patterns": ["bye","goodbye","quit","exit","see you","later","thanks bye","cya"],
        "responses": ["Goodbye! Keep coding!", "See you later!", "Bye! Best of luck with your internship!"]
    },
    "thanks": {
        "patterns": ["thanks","thank you","thx","ty","thank"],
        "responses": ["You're welcome!", "Happy to help!", "Anytime! Feel free to ask more."]
    },
    "help": {
        "patterns": ["help","topics","what do you know","what can you do"],
        "responses": ["I can answer questions about:\n"
                      "  - Python  - Machine Learning  - Data Science\n"
                      "  - NLP     - AI / Deep Learning - Pandas / NumPy\n"
                      "  - Scikit-learn  - GitHub  - Chatbots\n"
                      "Just ask me anything on these topics!"]
    }
}

def check_intent(user_input: str):
    """Match user input against intent patterns."""
    low = user_input.lower().strip()
    for intent, data in INTENTS.items():
        if any(p in low for p in data["patterns"]):
            response = random.choice(data["responses"])
            return response, (intent == "farewell")
    return None, False


# ── 5. MAIN CHAT FUNCTION ─────────────────────────────────────────────────────
FALLBACKS = [
    "I don't have info on that. Try asking about Python, ML, or AI!",
    "That's outside my knowledge. Type 'help' to see what I know.",
    "Hmm, I'm not sure about that. Rephrase or ask a different question!",
]

def chat():
    """Interactive chat loop."""
    print("=" * 60)
    print("  CodTech — Task 3: AI Chatbot with NLP")
    print("=" * 60)
    print("\nBot: Hi! I'm an NLP-powered chatbot.")
    print("     Type 'help' for topics | 'quit' to exit\n" + "-"*60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("Bot: Please type something!")
            continue

        intent_resp, is_exit = check_intent(user_input)
        if intent_resp:
            print(f"Bot: {intent_resp}")
            if is_exit:
                break
            continue

        response = get_best_response(user_input)
        print(f"Bot: {response if response else random.choice(FALLBACKS)}")


# ── 6. DEMO ───────────────────────────────────────────────────────────────────
def demo():
    """Automated demo without user input."""
    test_queries = [
        "Hello",
        "What is Python used for?",
        "Tell me about machine learning",
        "What is NLP?",
        "How does pandas work?",
        "What is scikit-learn?",
        "Tell me about deep learning",
        "What is GitHub?",
        "thanks",
        "bye",
    ]
    print("=" * 60)
    print("  TASK 3 — AI Chatbot with NLP  [DEMO MODE]")
    print("=" * 60)

    for q in test_queries:
        print(f"\nYou: {q}")
        intent_resp, _ = check_intent(q)
        if intent_resp:
            print(f"Bot: {intent_resp}")
        else:
            resp = get_best_response(q)
            print(f"Bot: {resp or random.choice(FALLBACKS)}")

    print("\n" + "="*60)
    print("  Demo complete! Use --chat flag for interactive mode.")
    print("="*60)


if __name__ == "__main__":
    import sys
    if "--chat" in sys.argv:
        chat()
    else:
        demo()
