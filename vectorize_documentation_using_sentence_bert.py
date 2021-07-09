import os
import sys
from sentence_transformers import SentenceTransformer


DOCS = []
SHAS = []


def load_doc_content(folder: str):
    for filename in os.listdir(folder):
        with open(f"{folder}/{filename}", 'r') as f:
            SHAS.append(filename)
            DOCS.append(f.read().replace("\n", ""))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python3 vectorize_documentation.py "
              "documentation_folder output_folder")
        sys.exit(1)
    documentation_folder = sys.argv[1]
    output_folder = sys.argv[2]
    print("[*] Loading paraphrase-mpnet-base-v2 model...")
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    print("[*] Loading documentation...")
    load_doc_content(documentation_folder)
    print("[*] Encoding documentation...")
    docs_embeddings = model.encode(DOCS)
    with open(f"{output_folder}/documentation_vectors.txt", "w") as doc_vec:
        for i in range(0, len(DOCS)):
            res = f"{SHAS[i]};{' '.join(str(f) for f in docs_embeddings[i])}"
            doc_vec.write(f"{res}\n")
