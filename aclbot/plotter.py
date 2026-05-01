# Copyright 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# Licensed under the Apache License, Version 2.0

import os
import hashlib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from pathlib import Path
import random
from typing import Optional, Any
from neo4j import GraphDatabase

# Configurable switches
DRAW_BARCHARTS = True
DRAW_SCATTERPLOTS = True
DRAW_HEATMAPS = True

# Neo4j
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Hardcoded keywords grouped by section
KEYWORDS = {
    "Architectures": [
        "Transformer", "BERT", "RoBERTa", "LSTM", "T5", "BART", "GPT-2", "GPT-3", "GPT-4", "GPT-3.5",
        "XLNet", "ELECTRA", "ALBERT", "DistilBERT", "BioBERT", "SciBERT", "DeBERTa", "CamemBERT",
        "XLM-R", "ChatGPT", "mBERT", "Flan-T5", "LLaMA", "Mistral", "GPT-4o"
    ],
    "Methods": [
        "Fine-tuning", "Prompting", "Few-shot Learning", "Zero-shot Learning", "Instruction Tuning",
        "Reinforcement Learning", "Knowledge Distillation", "Continual Learning", "Domain Adaptation",
        "Multi-task Learning", "Contrastive Learning", "Self-training", "Active Learning",
        "Transfer Learning", "Adversarial Training", "LoRA", "Back-Translation", "Ensemble Learning"
    ],
    "Tasks": [
        "Text Classification", "Named Entity Recognition", "Question Answering", "Summarization",
        "Machine Translation", "Sentiment Analysis", "Coreference Resolution", "Relation Extraction",
        "Semantic Parsing", "Language Modeling", "Part-of-Speech Tagging", "Dependency Parsing",
        "Word Sense Disambiguation", "Semantic Role Labeling", "Text Simplification"
    ],
    "Metrics": [
        "Accuracy", "Precision", "Recall", "F1-score", "BLEU", "ROUGE-L", "METEOR",
        "Exact Match", "Perplexity", "BERTScore", "AUC", "Macro-F1", "Micro-F1", "CIDEr"
    ]
}

USAGE_STATS = {}

def normalize_filename(base, suffix=""):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_str = hashlib.md5(base.encode()).hexdigest()[:6]
    return f"plots/{timestamp}_{hash_str}{suffix}.png"

def extract_entity_type(user_input):
    text = user_input.lower()
    for section, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                singular = section[:-1] if section.endswith('s') else section
                return singular, kw
    return None, None

def generate_cypher(entity_type, keyword=None):
    if entity_type == "PaperYear":
        return """
        MATCH (p:Paper)
        WHERE p.year IS NOT NULL
        RETURN p.year AS key, COUNT(*) AS count
        ORDER BY key
        """

    # Select relation type based on node category
    if entity_type in {"PretrainedModel", "Method", "Architecture"}:
        rel = ":USES"
    elif entity_type in {"Task", "Dataset"}:
        rel = ":WORKS_ON"
    elif entity_type == "Metric":
        # Metric is a special case
        if keyword:
            return f"""
            MATCH (p:Paper)-[:REPORTS]->(r:Result)-[:ON]->(m:Metric {{name: '{keyword}'}})
            WHERE p.year IS NOT NULL
            RETURN p.year AS year, AVG(r.value) AS score
            ORDER BY year
            """
        else:
            return """
            MATCH (p:Paper)-[:REPORTS]->(r:Result)-[:ON]->(m:Metric)
            WHERE p.year IS NOT NULL
            RETURN p.year AS year, m.name AS metric, AVG(r.value) AS score
            ORDER BY year
            """

    else:
        rel = ":USES"

    if keyword:
        return f"""
        MATCH (p:Paper)-[{rel}]->(e:{entity_type} {{name: '{keyword}'}})
        WHERE p.year IS NOT NULL
        RETURN p.year AS key, COUNT(*) AS count
        ORDER BY key
        """
    else:
        return f"""
        MATCH (p:Paper)-[{rel}]->(e:{entity_type})
        WHERE p.year IS NOT NULL
        RETURN p.year AS key, COUNT(*) AS count
        ORDER BY key
        """

def try_plot(entity_type, data, filename, keyword=None):
    if not data:
        return False
    Path("plots").mkdir(exist_ok=True)
    df = pd.DataFrame(data)
    if df.empty:
        return False

    if DRAW_SCATTERPLOTS and "score" in df.columns:
        plt.figure(figsize=(10, 5))
        if "metric" in df.columns:
            sns.scatterplot(data=df, x="year", y="score", hue="metric", alpha=0.6)
        else:
            sns.lineplot(data=df, x="year", y="score", marker="o")
        plt.title(f"{keyword or entity_type} Score Over Time")
        plt.xlabel("Year")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True

    elif DRAW_BARCHARTS and "key" in df.columns and "count" in df.columns:
        x = df["key"].tolist()
        y = df["count"].tolist()
        if len(x) < 2:
            return False
        plt.figure(figsize=(10, 4))
        plt.bar(x, y)
        plt.xlabel("Year")
        plt.ylabel("Count")
        title = f"{keyword} Mentions Over Time" if keyword else f"{entity_type} Mentions Over Time"
        plt.title(title)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True

    elif DRAW_HEATMAPS and {'m1', 'm2', 'count'}.issubset(df.columns):
        pivot = df.pivot(index="m1", columns="m2", values="count").fillna(0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".0f")
        plt.title(f"{entity_type} Co-occurrence Heatmap")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        return True

    return False


def plot(user_input: str, bot_response: str):

    TREND_WORDS = ["trend", "over time", "growth", "increase", "history", "timeline"]

    # Only generate fallback plots if user is asking about trends
    if not any(w in user_input.lower() for w in TREND_WORDS):
        return None

    text = user_input.lower()

    entity_type, keyword = extract_entity_type(text)
    if not entity_type:
        return None

    cypher = generate_cypher(entity_type, keyword)
    if not cypher:
        return None

    with driver.session() as session:
        try:
            data = session.run(cypher).data()
        except Exception as e:
            print(f"❌ Cypher query failed: {e}")
            return None

    filename = normalize_filename(user_input)
    success = try_plot(entity_type, data, filename, keyword)
    if success:
        print(filename)
        return {"content_type": "image", "text": filename}
    return None
