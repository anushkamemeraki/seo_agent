# rule based cleaning of descriptions to remove low value sentences, SEO fluff, and preserve important semantic content about the art forms
import pandas as pd
import re
from bs4 import BeautifulSoup
from collections import Counter

#loading csv file into dataframe
INPUT_FILE = "enriched_sheet.csv"

df = pd.read_csv(INPUT_FILE)

# for testing
# df = df.head(10)

#configurations for LLM summary generation
DESCRIPTION_COLUMN = "description"

MAX_SENTENCES = 10

MIN_SENTENCE_WORDS = 5


#values/patterns indicating low value sentences that can be removed without losing important semantic content
LOW_VALUE_PATTERNS = [

    # Ecommerce fluff
    r"discover",
    r"shop now",
    r"buy now",
    r"explore our collection",
    r"perfect gift",
    r"limited edition",
    r"exclusive collection",
    r"must-have piece",
    r"perfect addition to your home",
    r"ideal for your home",
    r"elevate your home decor",

    # Marketing fluff
    r"premium quality",
    r"carefully curated",
    r"beautiful artwork",
    r"stunning artwork",
    r"unique masterpiece",

    # Generic filler
    r"rich cultural heritage",
    r"timeless elegance",
    r"traditional craftsmanship",
    r"symbol of indian heritage",
    r"deeply rooted in tradition",
    r"reflects the vibrant culture",
    r"celebrates indian culture",

    # UI junk
    r"read more",
    r"click here",
    r"view products",
    r"free shipping",
    r"add to cart",
    r"wishlist",
    r"learn more",
]


#faq and section heading patterns to identify and remove low value sentences that are 
# likely to be headings or FAQ sections rather than semantic content
SECTION_PATTERNS = [

    r"^faqs?$",
    r"^frequently asked questions$",
    r"^history of",
    r"^types of",
    r"^materials and methods",
    r"^materials and method",
    r"^buying .* online",
    r"^why is",
    r"^how old is",
    r"^which god",
    r"^conclusion",
    r"^significance",
    r"^themes in",
]

#important entity keywords to preserve sentences that contain key information about the art forms,
#  artists, materials, techniques, and cultural context
IMPORTANT_ENTITY_WORDS = [

    "madhubani",
    "pattachitra",
    "gond",
    "warli",
    "kalighat",

    "artist",
    "artisan",
    "craft",

    "bihar",
    "odisha",
    "india",
    "mithila",

    "painting",
    "folk art",
    "traditional art",

    "canvas",
    "paper",
    "fabric",
    "wood",

    "natural colors",
    "natural pigments",

    "handmade",
    "handcrafted",
]


#remove HTML tags and extract text content, which is common in descriptions and
#  can interfere with sentence processing
def remove_html(text):

    soup = BeautifulSoup(text, "html.parser")

    return soup.get_text(" ")


#normalize text by removing unicode characters, fixing merged headings, converting colons to sentences,
#  and cleaning up spaces and punctuation to create a cleaner input for sentence processing and LLM summarization
def normalize_text(text):

    # Remove unicode / emojis
    text = re.sub(
        r'[^\x00-\x7F]+',
        ' ',
        text
    )

    # Fix merged headings
    text = re.sub(
        r'([a-z])([A-Z])',
        r'\1. \2',
        text
    )

    # Convert colon sections into sentences
    text = re.sub(
        r":\s+",
        ". ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove repeated punctuation
    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    # Remove repeated commas
    text = re.sub(
        r",\s*,+",
        ",",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([.,!?])",
        r"\1",
        text
    )

    return text.strip()


#detect if a sentence contains important entity keywords to help preserve sentences with key 
#information about the art forms, artists, materials, techniques, and cultural context during the cleaning process
def contains_important_entity(sentence):

    sentence_lower = sentence.lower()

    return any(
        keyword in sentence_lower
        for keyword in IMPORTANT_ENTITY_WORDS
    )

#FAQ and section heading detection to identify and remove low value sentences
#that are likely to be headings or FAQ sections rather than semantic content,
# which can help improve the quality of the cleaned descriptions and the resulting LLM summaries
def is_section_or_faq(sentence):

    sentence_lower = sentence.lower().strip()

    for pattern in SECTION_PATTERNS:

        if re.search(pattern, sentence_lower):
            return True

    return False


#heading detection to identify and remove low value sentences that are likely to be headings
# rather than semantic content, which can help improve the quality of the cleaned descriptions
# and the resulting LLM summaries
def is_heading(sentence):

    # Small title-like lines
    if len(sentence.split()) <= 8:

        if sentence.istitle():
            return True

    return False

#low value sentence detection based on patterns, length, and presence of important entities
# to filter out sentences that do not contribute meaningful semantic content about the art forms,
# artists, materials, techniques, and cultural context
def is_low_value_sentence(sentence):

    sentence_lower = sentence.lower()

    # Tiny sentence
    if len(sentence.split()) < MIN_SENTENCE_WORDS:
        return True

    # Very low alphabetic content
    if len(re.findall(r"[a-zA-Z]", sentence)) < 15:
        return True

    # Remove headings / FAQ
    if is_heading(sentence):
        return True

    if is_section_or_faq(sentence):
        return True

    # Preserve semantic sentences
    if contains_important_entity(sentence):
        return False

    # Remove SEO fluff
    for pattern in LOW_VALUE_PATTERNS:

        if re.search(
            pattern,
            sentence_lower,
            flags=re.IGNORECASE
        ):
            return True

    return False


#remove duplicate sentences while preserving order to help reduce redundancy and
# improve the quality of the cleaned descriptions and the resulting LLM summaries
def deduplicate_sentences(sentences):

    seen = set()

    unique = []

    for sentence in sentences:

        normalized = sentence.lower().strip()

        if normalized not in seen:

            unique.append(sentence)

            seen.add(normalized)

    return unique


#remove keyword stuffing by identifying and filtering out sentences that contain
# excessively repeated phrases, which can help improve the quality of the cleaned descriptions
# and the resulting LLM summaries by reducing noise and redundancy
def remove_keyword_stuffing(sentences):

    phrase_counter = Counter()

    cleaned = []

    for sentence in sentences:

        normalized = sentence.lower()

        words = normalized.split()

        phrases = [

            " ".join(words[i:i+3])

            for i in range(len(words)-2)
        ]

        repeated = False

        for phrase in phrases:

            phrase_counter[phrase] += 1 #counts occurrences of each 3-word phrase across sentences

            # Phrase repeated too much
            #if a 3-word phrase appears more than 3 times across sentences,
            # it is considered excessively repeated and indicates keyword stuffing
            if phrase_counter[phrase] > 3: 
                repeated = True

        if not repeated:
            cleaned.append(sentence) #if sentence does not contain excessively repeated phrases, it is kept in the cleaned list

    return cleaned

#sentence scoring based on presence of important entities, key information about the art forms,
# artists, materials, techniques, and cultural context, as well as penalizing noisy repetition
# and huge sentences to help prioritize high value sentences during the cleaning process and 
# improve the quality of the resulting LLM summaries
def score_sentence(sentence):

    score = 0

    sentence_lower = sentence.lower()

    # Important semantic entities
    if contains_important_entity(sentence):
        score += 5

    # Artist/location/material info
    important_info = [

        "artist",
        "artisan",
        "bihar",
        "odisha",
        "india",
        "mithila",

        "canvas",
        "paper",
        "fabric",
        "wood",

        "natural pigments",
        "natural colors",

        "folk art",
        "tribal art",
    ]

    if any(
        word in sentence_lower
        for word in important_info
    ):
        score += 4

    # Penalize noisy repetition
    repetitive_patterns = [

        "radha krishna",
        "lord jagannath",
        "mythological",
        "buy online",
    ]

    if any(
        word in sentence_lower
        for word in repetitive_patterns
    ):
        score -= 2

    # Penalize huge noisy sentences
    if len(sentence.split()) > 40:
        score -= 1

    return score


#main cleaning function that applies all the steps to clean the descriptions by removing
# low value sentences, SEO fluff, and preserving important semantic content about the art forms,
# artists, materials, techniques, and cultural context
def clean_description(text):

    # Handle nulls
    if pd.isna(text):
        return text

    text = str(text)

    
    text = remove_html(text)

    text = normalize_text(text)

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    cleaned_sentences = []
    #filter out low value sentences based on patterns, length, and presence of important entities
    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if is_low_value_sentence(sentence):
            continue

        cleaned_sentences.append(sentence)

    #remove duplicate sentences while preserving order to help reduce redundancy and
    # improve the quality of the cleaned descriptions and the resulting LLM summaries
    cleaned_sentences = deduplicate_sentences(
        cleaned_sentences
    )

    #remove keyword stuffing by identifying and filtering out sentences that contain
    # excessively repeated phrases, which can help improve the quality of the cleaned descriptions
    cleaned_sentences = remove_keyword_stuffing(
        cleaned_sentences
    )

    #sentence scoring based on presence of important entities, key information about the art forms,
    scored_sentences = []

    for index, sentence in enumerate(cleaned_sentences):

        score = score_sentence(sentence)

        scored_sentences.append(
            (score, sentence, index)
        )

    #select top sentences based on scores to help prioritize high value sentences during the cleaning process and
    top_sentences = sorted(
        scored_sentences,
        key=lambda x: x[0],
        reverse=True
    )[:MAX_SENTENCES]
    #sort top sentences back to original order to maintain coherence in the cleaned descriptions
    top_sentences = sorted(
        top_sentences,
        key=lambda x: x[2]
    )

    # extract sentences from scored tuples to create the final cleaned description
    best_sentences = [

        sentence
        for score, sentence, index
        in top_sentences
    ]

    # join best sentences to create the final cleaned description text
    cleaned_text = " ".join(best_sentences)

    # final normalization to clean up any remaining issues and create a cleaner input for LLM summarization
    cleaned_text = normalize_text(
        cleaned_text
    )

    return cleaned_text


print("\nCleaning descriptions...\n")

df["cleaned_description"] = df[
    DESCRIPTION_COLUMN
].apply(clean_description)


for index, row in df.iterrows():

    print("=" * 120)
    print(f"ROW: {index}")
    print("=" * 120)

    print("\nORIGINAL:\n")
    print(str(row[DESCRIPTION_COLUMN])[:2000])

    print("\n" + "-" * 120)

    print("\nCLEANED:\n")
    print(str(row["cleaned_description"])[:2000])

    print("\n\n")


#save intermediate cleaned descriptions to a new CSV file to allow for manual review and further processing before LLM summarization
OUTPUT_FILE = "cleaned_descriptions.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nCleaned file saved as: "
    f"{OUTPUT_FILE}"
)