import pandas as pd
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

# =====================================================
# CONFIG
# =====================================================

CSV_FILE = "focused_retrieval_embeddings.csv"

EMBEDDING_FILE = "focused_embeddings.npy"

MODEL_NAME = "BAAI/bge-large-en-v1.5"

TOP_K = 5

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading retrieval data...")

df = pd.read_csv(CSV_FILE)

# =====================================================
# LOAD EMBEDDINGS
# =====================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"\nEmbeddings shape: "
    f"{embeddings.shape}"
)

# =====================================================
# FILTER VALID ROWS
# MUST MATCH EMBEDDING SCRIPT
# =====================================================

valid_mask = (

    df["focused_retrieval_text"]

    .fillna("")

    .astype(str)

    .str.strip()

    .apply(
        lambda x:
        (
            x != ""
            and x.lower() != "nan"
            and "not specified"
            not in x.lower()
        )
    )
)

filtered_df = df[
    valid_mask
].reset_index(drop=True)

print(
    f"\nValid retrieval rows: "
    f"{len(filtered_df)}"
)

# =====================================================
# SAFETY CHECK
# =====================================================

if len(filtered_df) != len(embeddings):

    raise ValueError(
        "\nMismatch between "
        "filtered dataframe rows "
        "and embeddings count.\n"
    )

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

# =====================================================
# QUERY EXPANSION
# =====================================================

def expand_query(query):

    query = query.lower().strip()

    expanded = f"""
    {query}

    traditional indian art
    folk art
    tribal art
    handmade artwork
    indigenous painting
    ethnic motifs
    handcrafted art
    cultural heritage
    traditional painting
    """

    return " ".join(
        expanded.split()
    )

# =====================================================
# KEYWORD BOOSTING
# =====================================================

def keyword_boost(
    query,
    row
):

    boost = 0.0

    query = query.lower()

    retrieval_text = str(
        row.get(
            "focused_retrieval_text",
            ""
        )
    ).lower()

    # ================================================
    # ANIMAL BOOST
    # ================================================

    animal_terms = [
        "animal",
        "deer",
        "bird",
        "fish",
        "elephant",
        "wildlife",
        "fauna",
        "forest"
    ]

    if "animal" in query:

        matches = sum(
            term in retrieval_text
            for term in animal_terms
        )

        boost += matches * 0.03

    # ================================================
    # TRIBAL BOOST
    # ================================================

    tribal_terms = [
        "tribal",
        "gond",
        "bhil",
        "warli",
        "sohrai",
        "indigenous",
        "folk"
    ]

    if "tribal" in query:

        matches = sum(
            term in retrieval_text
            for term in tribal_terms
        )

        boost += matches * 0.04

    # ================================================
    # CONTENT TYPE BOOSTING
    # ================================================

    page_type = str(
        row.get(
            "Landing page type",
            ""
        )
    ).lower()

    # boost collections/products
    if "collection" in page_type:

        boost += 0.08

    if "product" in page_type:

        boost += 0.10

    # reduce blogs
    if "blog" in page_type:

        boost -= 0.05

    # ================================================
    # TITLE BOOST
    # ================================================

    title = str(
        row.get(
            "title",
            ""
        )
    ).lower()

    query_words = query.split()

    title_matches = sum(
        word in title
        for word in query_words
    )

    boost += title_matches * 0.05

    return boost

# =====================================================
# SEARCH LOOP
# =====================================================

while True:

    print("\n" + "=" * 80)

    query = input(
        "\nEnter search query: "
    ).strip()

    if query.lower() in [
        "exit",
        "quit"
    ]:

        print("\nExiting semantic search...\n")

        break

    # =================================================
    # EXPAND QUERY
    # =================================================

    expanded_query = expand_query(
        query
    )

    # IMPORTANT FOR BGE
    expanded_query = (
        f"query: {expanded_query}"
    )

    # =================================================
    # GENERATE QUERY EMBEDDING
    # =================================================

    query_embedding = model.encode(

        [expanded_query],

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    # =================================================
    # BASE COSINE SIMILARITY
    # =================================================

    similarities = cosine_similarity(

        query_embedding,

        embeddings

    )[0]

    # =================================================
    # APPLY BOOSTING
    # =================================================

    final_scores = []

    for idx, similarity in enumerate(
        similarities
    ):

        row = filtered_df.iloc[idx]

        boost = keyword_boost(
            query,
            row
        )
        #final score is a combination of semantic similarity and keyword boosting
        final_score = (
            similarity + boost
        )

        final_scores.append(
            final_score
        )

    final_scores = np.array(
        final_scores
    )

    # =================================================
    # GET TOP RESULTS
    # =================================================

    top_indices = final_scores.argsort(
    )[-TOP_K:][::-1]

    print("\nTOP RESULTS:\n")

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        row = filtered_df.iloc[idx]

        similarity_score = similarities[idx]

        boosted_score = final_scores[idx]

        print("=" * 80)

        print(f"Rank: {rank}")

        print(
            f"\nSemantic Score: "
            f"{round(similarity_score, 4)}"
        )

        print(
            f"Final Boosted Score: "
            f"{round(boosted_score, 4)}"
        )

        # =================================================
        # TITLE
        # =================================================

        print(
            f"\nTitle:\n"
            f"{row.get('title', '')}"
        )

        # =================================================
        # CATEGORY
        # =================================================

        print(
            f"\nCategory:\n"
            f"{row.get('category', '')}"
        )

        # =================================================
        # ART STYLE
        # =================================================

        print(
            f"\nArt Style:\n"
            f"{row.get('art_style', '')}"
        )

        # =================================================
        # MOTIFS
        # =================================================

        print(
            f"\nMotifs:\n"
            f"{row.get('motifs', '')}"
        )

        # =================================================
        # SUBJECTS
        # =================================================

        print(
            f"\nSubjects:\n"
            f"{row.get('subjects', '')}"
        )

        # =================================================
        # TAGS
        # =================================================

        print(
            f"\nTags:\n"
            f"{row.get('tags', '')}"
        )

        # =================================================
        # LINK
        # =================================================

        if (
            "Landing page path"
            in filtered_df.columns
        ):

            print(
                f"\nBest Matching Link:\n"
                f"{row['Landing page path']}"
            )

        print("=" * 80)