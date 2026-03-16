from dotenv import load_dotenv
load_dotenv()

import json

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from app.query_reasoner import reason_query
from openai import OpenAI
client = OpenAI()

def embed_query(text):

    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return res.data[0].embedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def detect_intent(parsed):

    skills = parsed.get("technical_skills", [])
    role = parsed.get("role", "")
    comps = parsed.get("competencies", [])

    if skills:
        return "skill"

    if role:
        return "role"

    if any(c in ["leadership", "culture", "communication"] for c in comps):
        return "leadership"

    return "general"


def build_query_text(parsed):

    parts = []

    if parsed.get("role"):
        parts.append(parsed["role"])

    parts += parsed.get("technical_skills", [])
    parts += parsed.get("competencies", [])

    return " ".join(parts)

def add_solution_prefix(url):

    if "solutions/products" in url:
        return url

    return url.replace(
        "shl.com/products",
        "shl.com/solutions/products"
    )

class HybridRecommender:
    def __init__(self, dataset_path):

        with open(dataset_path) as f:
            data = json.load(f)

        self.df = pd.DataFrame(data)

        # load catalog dataset for output fields
        with open("data/assessments_catalog.json", encoding="utf-8") as f:
            catalog = json.load(f)

        self.catalog = pd.DataFrame(catalog)

        # normalize catalog url
        self.catalog["url"] = self.catalog["url"].str.lower()

        self.df["retrieval_text"] = self.df.apply(self.build_retrieval_text, axis=1)

        self.model = SentenceTransformer(MODEL_NAME)

        self.embeddings = np.vstack(self.df["embedding"].values)

        self.tfidf = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=50000
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.df["retrieval_text"])
        self.tfidf_matrix = normalize(self.tfidf_matrix)

    def build_retrieval_text(self, row):

        parts = []

        if row.get("description"):
            parts.append(row["description"])

        if isinstance(row.get("assessed_skills_norm"), list):
            parts.append("skills " + " ".join(row["assessed_skills_norm"]))

        if isinstance(row.get("target_roles_norm"), list):
            parts.append("roles " + " ".join(row["target_roles_norm"]))

        if isinstance(row.get("cognitive_dimensions_norm"), list):
            parts.append("competencies " + " ".join(row["cognitive_dimensions_norm"]))

        return " ".join(parts)

    def retrieve_candidates(self, query_text, top_k=100):

        q_emb = np.array(embed_query(query_text))

        sims = cosine_similarity([q_emb], self.embeddings)[0]

        idx = np.argsort(sims)[::-1][:top_k]

        return idx, sims

    def adaptive_score(self, parsed, intent, idx, sims, query_text):

        q_tfidf = self.tfidf.transform([query_text])

        tfidf_scores = (self.tfidf_matrix[idx] @ q_tfidf.T).toarray().ravel()

        role = parsed.get("role", "")
        skills = " ".join(parsed.get("technical_skills", []))
        comps = " ".join(parsed.get("competencies", []))

        role_emb = np.array(embed_query(role)) if role else None
        skill_emb = np.array(embed_query(skills)) if skills else None
        comp_emb = np.array(embed_query(comps)) if comps else None

        role_sim = cosine_similarity([role_emb], self.embeddings[idx])[0] if role_emb is not None else np.zeros(len(idx))
        skill_sim = cosine_similarity([skill_emb], self.embeddings[idx])[0] if skill_emb is not None else np.zeros(len(idx))
        comp_sim = cosine_similarity([comp_emb], self.embeddings[idx])[0] if comp_emb is not None else np.zeros(len(idx))

        if intent == "skill":
            score = 0.5 * skill_sim + 0.25 * sims[idx] + 0.15 * tfidf_scores + 0.10 * comp_sim

        elif intent == "role":
            score = 0.5 * role_sim + 0.25 * sims[idx] + 0.15 * tfidf_scores + 0.10 * comp_sim

        else:
            score = 0.4 * sims[idx] + 0.3 * tfidf_scores + 0.3 * comp_sim

        return score


class RecommenderService:

    def __init__(self, dataset_path):

        self.engine = HybridRecommender(dataset_path)

    def recommend_v2(self, raw_text):

        parsed = reason_query(raw_text)

        intent = detect_intent(parsed)

        query_text = build_query_text(parsed)

        idx, sims = self.engine.retrieve_candidates(query_text)

        scores = self.engine.adaptive_score(parsed, intent, idx, sims, query_text)

        res = self.engine.df.iloc[idx].copy()

        res["score"] = scores

        res = res.sort_values("score", ascending=False).head(10)

        urls = res["url"].str.lower().tolist()

        # join with catalog dataset
        final = self.engine.catalog[
            self.engine.catalog["url"].isin(urls)
        ].copy()

        final["url"] = final["url"].apply(add_solution_prefix)

        cols = [
            "url",
            "name",
            "adaptive_support",
            "description",
            "duration",
            "remote_support",
            "test_type"
        ]

        final = final[cols]

        records = final.to_dict("records")

        clean_records = []

        for r in records:
            clean = {}

            for k, v in r.items():

                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    clean[k] = None

                elif k == "test_type" and isinstance(v, list):
                    # remove whitespace around each value
                    clean[k] = [str(x).strip() for x in v]

                else:
                    clean[k] = v

            clean_records.append(clean)

        return clean_records