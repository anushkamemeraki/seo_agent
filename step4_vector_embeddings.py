import pandas as pd
import numpy as np
import torch

from sentence_transformers import (
    SentenceTransformer
)

#config 
INPUT_FILE = "retrieval_ready_data.csv"

OUTPUT_FILE = (
    "focused_retrieval_embeddings.csv"
)

EMBEDDING_FILE = "focused_embeddings.npy"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

BATCH_SIZE = 64

#loading csv
print("\nLoading CSV file...")

df = pd.read_csv(INPUT_FILE)

#required columns check
required_columns = [
    "title",
    "category",
    "art_style",
    "motifs",
    "keywords",
    "subjects",
    "tags"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Missing column: {col}"
        )
#created focused retrieval text by combining and weighting important columns
print(
    "\nCreating focused retrieval text..."
)

focused_texts = []

valid_indices = []

for idx, row in df.iterrows():

    title = str(
        row.get("title", "")
    ).strip()

    category = str(
        row.get("category", "")
    ).strip()

    art_style = str(
        row.get("art_style", "")
    ).strip()

    motifs = str(
        row.get("motifs", "")
    ).strip()

    keywords = str(
        row.get("keywords", "")
    ).strip()

    subjects = str(
        row.get("subjects", "")
    ).strip()

    tags = str(
        row.get("tags", "")
    ).strip()
    #skipping weak rows
    combined = (
        title
        + category
        + art_style
        + motifs
        + keywords
        + subjects
        + tags
    ).lower()

    if (
        not combined
        or combined == "nan"
        or "not specified" in combined
    ):

        continue

    
    # SEMANTIC WEIGHTING
    # IMPORTANT:
    # Repeat important concepts to give them more weight in the embedding space

    focused_text = f"""
    Title: {title}

    Category:
    {category}
    {category}

    Art Style:
    {art_style}
    {art_style}

    Motifs:
    {motifs}
    {motifs}

    Subjects:
    {subjects}
    {subjects}

    Keywords:
    {keywords}

    Tags:
    {tags}

    Traditional Indian Art
    Folk Art
    Handmade Artwork
    Cultural Heritage
    Indigenous Art
    """

    focused_text = " ".join(
        focused_text.split()
    )

    # IMPORTANT:
    # BGE retrieval instruction
    focused_text = (
        f"passage: {focused_text}"
    )

    focused_texts.append(
        focused_text
    )

    valid_indices.append(idx)
#saved focused retrieval text in dataframe for reference and debugging
df["focused_retrieval_text"] = None

for idx, text in zip(
    valid_indices,
    focused_texts
):

    df.at[
        idx,
        "focused_retrieval_text"
    ] = text

#current device is CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

#loading model
print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

#generating embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(

    focused_texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)

print(
    f"\nEmbeddings shape: "
    f"{embeddings.shape}"
)

#save embeddings as numpy array for efficient loading during retrieval
np.save(
    EMBEDDING_FILE,
    embeddings
)

print(
    f"\nEmbeddings saved to:"
    f"\n{EMBEDDING_FILE}"
)

#save the dataframe with focused retrieval text for reference and debugging
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


print("\n" + "=" * 70)

print(
    "FOCUSED RETRIEVAL "
    "EMBEDDINGS GENERATED"
)

print(
    f"\nCSV saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nEmbeddings saved to:\n"
    f"{EMBEDDING_FILE}"
)

print("=" * 70)