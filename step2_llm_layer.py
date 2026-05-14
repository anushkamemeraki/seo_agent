import pandas as pd
import os
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file"
    )

print("\nLOADED API KEY\n")


client = genai.Client(
    api_key=GEMINI_API_KEY
)


MODEL_NAME = "gemini-2.5-flash"


INPUT_FILE = "cleaned_descriptions.csv"

OUTPUT_FILE = "semantic_summaries_test.csv"

INPUT_COLUMN = "cleaned_description"

OUTPUT_COLUMN = "semantic_summary"



MAX_WORKERS = 10

df = pd.read_csv(INPUT_FILE)




PROMPT_TEMPLATE = """
You are an expert semantic enrichment engine for SEO, semantic search, embeddings, vector databases, AI retrieval systems, and RAG pipelines.

Your task is to enrich the input text while faithfully preserving its original meaning, tone, storytelling style, emotional continuity, and cultural authenticity.

The goal is NOT aggressive summarization or heavy rewriting.

The output should feel like the same human-written content — simply cleaner, semantically richer, more retrieval-friendly, and better structured for embeddings and SEO.

IMPORTANT:
- Preserve the original voice and natural phrasing whenever possible.
- Preserve complete meaning, sentence flow, and emotional continuity.
- Preserve full sentence endings, reflective statements, and concluding thoughts.
- If a sentence is already strong and natural, preserve it almost exactly.
- Do not unnecessarily shorten, simplify, or compress the writing.

The final output MUST:
1. Retain the original tone and storytelling style.
2. Preserve natural human phrasing and cultural authenticity.
3. Highlight the key characteristics of the main subject.
4. Preserve awards, recognitions, and achievements if mentioned.
5. Preserve unique techniques, craftsmanship methods, and materials.
6. Include geographical origins, regional influences, and cultural context.
7. Preserve artistic, historical, and thematic significance.
8. Preserve semantic relationships and contextual continuity.
9. Preserve important descriptive richness and nuance.
10. Be written as a SINGLE PARAGRAPH only.
11. Ensure the paragraph is complete and fully coherent.

PRESERVE:
- artist names
- art forms
- regional references
- cultural context
- historical references
- techniques
- materials
- symbolism
- themes
- stylistic elements
- semantic relationships
- important descriptive details
- long-tail keywords
- contextual meaning
- artistic identity
- traditional significance
- emotional continuity
- storytelling flow

REMOVE ONLY:
- obvious marketing fluff
- CTA language
- duplicated sentences
- excessive ecommerce phrasing
- unnecessary promotional wording
- navigation-like clutter

DO NOT:
- hallucinate information
- invent details
- over-compress the text
- convert the content into a short summary
- remove meaningful context
- remove culturally important information
- over-formalize the writing
- rewrite in encyclopedia style
- replace authentic phrasing unnecessarily
- dramatize the narrative
- add poetic embellishments
- trim sentence endings
- simplify reflective statements
- make the text sound robotic or AI-generated

STYLE:
- semantically rich
- natural
- grounded
- authentic
- context-preserving
- information-dense
- SEO-friendly
- embedding-friendly
- retrieval-friendly
- culturally respectful
- human-readable

The final output should read like the original author’s writing — with improved semantic clarity, contextual richness, and retrieval quality while preserving the original tone and completeness.

INPUT:
{input_text}
"""
def generate_semantic_summary(text):

    if pd.isna(text):
        return ""

    text = str(text).strip()

    if not text:
        return ""

    prompt = PROMPT_TEMPLATE.format(
        input_text=text
    )

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
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

                output = " ".join(parts)

        return " ".join(output.split())

    except Exception as e:

        print(f"ERROR: {e}")

        return ""

# =========================
# PROCESS ROW
# =========================

def process_row(index, text):

    print(f"Processing row {index}")

    summary = generate_semantic_summary(
        text
    )

    return index, summary

# =========================
# MULTITHREADED EXECUTION
# =========================

semantic_outputs = [""] * len(df)

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = []

    for index, row in enumerate(
        df[INPUT_COLUMN]
    ):

        futures.append(
            executor.submit(
                process_row,
                index,
                row
            )
        )

    for future in as_completed(futures):

        index, summary = future.result()

        semantic_outputs[index] = summary

# =========================
# SAVE
# =========================

df[OUTPUT_COLUMN] = semantic_outputs

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDONE")
print(f"Saved to: {OUTPUT_FILE}")