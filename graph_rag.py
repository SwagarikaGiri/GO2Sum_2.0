import os
import pickle
import networkx as nx
import numpy as np
from goatools.obo_parser import GODag
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# === Step 1: Parse GO .obo File ===
def load_go_graph(obo_path):
    go_dag = GODag(obo_path)
    G = nx.DiGraph()
    for go_id, term in go_dag.items():
        G.add_node(go_id, name=term.name, namespace=term.namespace, definition=term.defn)
        for parent in term.parents:
            G.add_edge(go_id, parent.id)
    return G, go_dag

# === Step 2: Generate Embeddings ===
def generate_embeddings(go_dag, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = {}
    for go_id, term in go_dag.items():
        if term.defn:
            try:
                emb = model.encode(term.defn)
                embeddings[go_id] = emb
            except Exception as e:
                print(f"Embedding error for {go_id}: {e}")
    return embeddings

# === Step 3: Cluster GO Terms ===
def cluster_go_terms(go_ids, embeddings, n_clusters=5):
    valid_go_ids = [go for go in go_ids if go in embeddings]
    if not valid_go_ids:
        return []

    X = np.array([embeddings[go] for go in valid_go_ids])
    cluster = AgglomerativeClustering(n_clusters=min(n_clusters, len(valid_go_ids)))
    labels = cluster.fit_predict(X)

    clusters = {}
    for go, label in zip(valid_go_ids, labels):
        clusters.setdefault(label, []).append(go)

    # Select representative from each cluster (e.g., first item)
    representatives = [gos[0] for gos in clusters.values()]
    return representatives

# === Example Usage ===
if __name__ == "__main__":
    obo_path = "data/gene_ontology.obo"  # path to your .obo file
    G, go_dag = load_go_graph(obo_path)
    embeddings = generate_embeddings(go_dag)

    # Example input GO terms (you can replace with your actual list)
    input_go_terms = ["GO:0003723", "GO:0005634", "GO:0005737", "GO:0005829", "GO:0008486"]
    reduced_go_terms = cluster_go_terms(input_go_terms, embeddings, n_clusters=3)

    print("Original GO terms:")
    print(input_go_terms)
    print("\nReduced GO terms (representatives):")
    print(reduced_go_terms)
