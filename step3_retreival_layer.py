import pandas as pd
import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found"
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "gemini-2.5-flash"

INPUT_FILE = "semantic_summaries_test.csv"

OUTPUT_FILE = "retrieval_ready_data.csv"

MAX_WORKERS = 5



# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(INPUT_FILE)


# =====================================================
# PROMPT
# =====================================================

PROMPT_TEMPLATE = """
You are an expert retrieval enrichment engine
for semantic search systems, embeddings,
vector databases, SEO retrieval,
and RAG pipelines.

Your task is to extract highly retrieval-friendly
metadata from the input content.

You must identify:

1. Primary art category
2. Art style or tradition
3. Major visual motifs
4. Important keywords
5. Cultural themes
6. Geographic origin
7. Materials or techniques
8. Subjects depicted
9. Semantic retrieval tags

Return STRICTLY in this format:

CATEGORY:
...

ART_STYLE:
...

MOTIFS:
...

KEYWORDS:
keyword1, keyword2, keyword3

THEMES:
...

REGION:
...

TECHNIQUES:
...

SUBJECTS:
...

TAGS:
tag1, tag2, tag3

INPUT:
{input_text}
"""

# =====================================================
# GENERATE RETRIEVAL FEATURES
# =====================================================

def generate_retrieval_metadata(text):

    if pd.isna(text):

        return {}

    text = str(text).strip()

    if not text:

        return {}

    prompt = PROMPT_TEMPLATE.format(
        input_text=text
    )

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt,

            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=4096,
            )
        )

        output = ""

        if response.candidates:

            candidate = response.candidates[0]

            if (
                hasattr(candidate, "content")
                and candidate.content
            ):

                parts = []

                for part in candidate.content.parts:

                    if hasattr(part, "text"):

                        parts.append(part.text)

                output = "\n".join(parts)

        metadata = {
            "category": "",
            "art_style": "",
            "motifs": "",
            "keywords": "",
            "themes": "",
            "region": "",
            "techniques": "",
            "subjects": "",
            "tags": ""
        }

        current_key = None

        for line in output.split("\n"):

            line = line.strip()

            if not line:
                continue

            upper = line.upper()

            if upper.startswith("CATEGORY:"):

                current_key = "category"

                metadata[current_key] = (
                    line.replace(
                        "CATEGORY:",
                        ""
                    ).strip()
                )

            elif upper.startswith("ART_STYLE:"):

                current_key = "art_style"

                metadata[current_key] = (
                    line.replace(
                        "ART_STYLE:",
                        ""
                    ).strip()
                )

            elif upper.startswith("MOTIFS:"):

                current_key = "motifs"

                metadata[current_key] = (
                    line.replace(
                        "MOTIFS:",
                        ""
                    ).strip()
                )

            elif upper.startswith("KEYWORDS:"):

                current_key = "keywords"

                metadata[current_key] = (
                    line.replace(
                        "KEYWORDS:",
                        ""
                    ).strip()
                )

            elif upper.startswith("THEMES:"):

                current_key = "themes"

                metadata[current_key] = (
                    line.replace(
                        "THEMES:",
                        ""
                    ).strip()
                )

            elif upper.startswith("REGION:"):

                current_key = "region"

                metadata[current_key] = (
                    line.replace(
                        "REGION:",
                        ""
                    ).strip()
                )

            elif upper.startswith("TECHNIQUES:"):

                current_key = "techniques"

                metadata[current_key] = (
                    line.replace(
                        "TECHNIQUES:",
                        ""
                    ).strip()
                )

            elif upper.startswith("SUBJECTS:"):

                current_key = "subjects"

                metadata[current_key] = (
                    line.replace(
                        "SUBJECTS:",
                        ""
                    ).strip()
                )

            elif upper.startswith("TAGS:"):

                current_key = "tags"

                metadata[current_key] = (
                    line.replace(
                        "TAGS:",
                        ""
                    ).strip()
                )

            else:

                if current_key:

                    metadata[current_key] += (
                        " " + line
                    )

        return metadata

    except Exception as e:

        print(f"ERROR: {e}")

        return {}

# =====================================================
# PROCESS SINGLE ROW
# =====================================================

def process_row(index, row):

    print(f"Processing row {index}")

    title = str(
        row.get("title", "")
    )

    semantic_summary = str(
        row.get("semantic_summary", "")
    )

    combined_input = f"""
    TITLE:
    {title}

    SUMMARY:
    {semantic_summary}
    """

    metadata = generate_retrieval_metadata(
        combined_input
    )

    retrieval_text = f"""
    Title: {title}

    Category: {metadata.get('category', '')}

    Art Style: {metadata.get('art_style', '')}

    Motifs: {metadata.get('motifs', '')}

    Keywords: {metadata.get('keywords', '')}

    Themes: {metadata.get('themes', '')}

    Region: {metadata.get('region', '')}

    Techniques: {metadata.get('techniques', '')}

    Subjects: {metadata.get('subjects', '')}

    Tags: {metadata.get('tags', '')}

    Summary:
    {semantic_summary}
    """

    metadata["retrieval_text"] = (
        " ".join(retrieval_text.split())
    )

    return index, metadata

# =====================================================
# EXECUTION
# =====================================================

results = [{} for _ in range(len(df))]

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = []

    for index, row in df.iterrows():

        futures.append(

            executor.submit(
                process_row,
                index,
                row
            )
        )

    for future in as_completed(futures):

        index, metadata = future.result()

        results[index] = metadata

# =====================================================
# ADD COLUMNS
# =====================================================

metadata_df = pd.DataFrame(results)

df = pd.concat(
    [df, metadata_df],
    axis=1
)

# =====================================================
# SAVE
# =====================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)

print("RETRIEVAL ENRICHMENT COMPLETED")

print(f"\nSaved to: {OUTPUT_FILE}")

print("=" * 60)