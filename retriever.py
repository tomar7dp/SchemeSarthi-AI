"""
retriever.py

Lightweight RAG-style retrieval over the scheme dataset.

Instead of a heavy vector DB (overkill for 19 documents), this uses
TF-IDF similarity -- fast, no API keys needed, runs instantly, and is
conceptually the same retrieval idea (embed query + documents, find
nearest match) just with a classical instead of neural embedding.
For your report, you can honestly describe this as a TF-IDF-based
retriever and mention swapping in sentence-transformer embeddings +
FAISS/Chroma as a natural upgrade path.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SchemeRetriever:
    def __init__(self, schemes_csv="scheme_eligibility_data.csv"):
        self.df = pd.read_csv(schemes_csv)
        # Build one retrievable "document" per scheme by concatenating
        # its descriptive text fields -- this is the corpus we search.
        self.df["_doc_text"] = self.df.apply(self._build_doc_text, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.df["_doc_text"])

    @staticmethod
    def _build_doc_text(row):
        parts = [
            str(row.get("scheme_name", "")),
            str(row.get("target_category", "")),
            str(row.get("education_requirement", "")),
            str(row.get("ownership_requirement", "")),
            str(row.get("business_type_allowed", "")),
            str(row.get("special_conditions", "")),
            str(row.get("source_notes", "")),
        ]
        return " ".join(p for p in parts if p and p.lower() != "nan")

    def get_scheme_row(self, scheme_name):
        match = self.df[self.df["scheme_name"] == scheme_name]
        return match.iloc[0] if len(match) else None

    def retrieve_clause(self, scheme_name, query, top_k_sentences=2):
        """
        Given a scheme name and a query (e.g. 'eligibility for SC
        category'), return the most relevant sentence(s) from that
        scheme's text as a grounding citation.
        """
        row = self.get_scheme_row(scheme_name)
        if row is None:
            return "No matching scheme document found."

        doc_text = row["_doc_text"]
        sentences = [s.strip() for s in doc_text.split(";") if s.strip()]
        if not sentences:
            return doc_text

        sent_vectors = self.vectorizer.transform(sentences)
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, sent_vectors)[0]

        top_idx = sims.argsort()[::-1][:top_k_sentences]
        top_sentences = [sentences[i] for i in top_idx if sims[i] > 0]

        if not top_sentences:
            return sentences[0]  # fallback: first sentence of the doc
        return "; ".join(top_sentences)

    def search_schemes(self, query, top_k=5):
        """General semantic-ish search across all scheme documents."""
        query_vector = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            (self.df.iloc[i]["scheme_name"], sims[i])
            for i in top_idx if sims[i] > 0
        ]


if __name__ == "__main__":
    # Quick smoke test
    r = SchemeRetriever()
    print("Testing retrieval for PMEGP + 'SC category subsidy amount':")
    print(r.retrieve_clause(
        "Prime Minister's Employment Generation Programme (PMEGP)",
        "SC category subsidy amount"
    ))
    print()
    print("Testing general search for 'street vendor loan':")
    for name, score in r.search_schemes("street vendor loan"):
        print(f"  {score:.3f}  {name}")
