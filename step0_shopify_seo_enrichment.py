import csv
import requests
from urllib.parse import unquote, urlparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import os

# load environment variables
load_dotenv()

SHOP = os.getenv("SHOP")

CLIENT_ID = os.getenv("CLIENT_ID")

CLIENT_SECRET = os.getenv("CLIENT_SECRET")

API_VERSION = "2026-04"

# input/output files
INPUT_CSV = "seo_agent_scoring.csv"

OUTPUT_CSV = "enriched_sheet.csv"

NOT_FOUND_CSV = "not_found.csv"

# number of parallel workers
MAX_WORKERS = 10

# graphql query for collections
GET_COLLECTION_QUERY = """
query GetCollection($query: String!) {
  collections(first: 1, query: $query) {
    edges {
      node {
        id
        handle
        title
        description
      }
    }
  }
}
"""

# graphql query for pages
GET_PAGE_QUERY = """
query GetPage($query: String!) {
  pages(first: 1, query: $query) {
    edges {
      node {
        id
        handle
        title
        bodySummary
      }
    }
  }
}
"""

# graphql query for articles
GET_ARTICLE_QUERY = """
query GetArticle($query: String!) {
  articles(first: 20, query: $query) {
    edges {
      node {
        id
        handle
        title
        body
        blog {
          handle
        }
      }
    }
  }
}
"""


# authenticate with shopify
def get_access_token():

    print("\nAUTHENTICATING WITH SHOPIFY...\n")

    url = (
        f"https://{SHOP}"
        f"/admin/oauth/access_token"
    )

    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
    }

    response = requests.post(
        url,
        json=payload,
        timeout=20,
    )

    if response.status_code != 200:
        print(f"AUTH FAILED (HTTP {response.status_code})")
        print(response.text)
        return None

    data = response.json()

    token = data.get("access_token")

    if not token:
        print("ACCESS TOKEN NOT FOUND")
        print(data)
        return None

    print("AUTH SUCCESSFUL\n")

    return token


# execute graphql query
def execute_graphql_query(access_token, query, variables):

    endpoint = (
        f"https://{SHOP}"
        f"/admin/api/{API_VERSION}"
        f"/graphql.json"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }

    payload = {
        "query": query,
        "variables": variables,
    }

    response = requests.post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=30,
    )

    if response.status_code != 200:
        print(f"HTTP ERROR {response.status_code}")
        print(response.text)
        return None

    data = response.json()

    if "errors" in data:
        print("GRAPHQL ERRORS:")
        print(data["errors"])
        return None

    return data


# clean html content
def clean_html(html_content):

    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    return soup.get_text(separator=" ", strip=True)


# extract type and handle from path
def extract_handle(raw_path: str):

    if not raw_path:
        return None, None, None

    path = unquote(raw_path.strip())

    if path.startswith("http"):
        path = urlparse(path).path

    path = path.split("?")[0].split("#")[0].strip()

    parts = [p for p in path.split("/") if p]

    if not parts:
        return None, None, None

    resource_type = parts[0]

    if resource_type == "collections":

        if len(parts) >= 3 and parts[1] == "all":
            return "collection", parts[2], None

        elif len(parts) >= 2:
            return "collection", parts[1], None

    elif resource_type == "pages":

        if len(parts) >= 2:
            return "page", parts[1], None

    elif resource_type == "blogs":

        if len(parts) >= 3:
            return "article", parts[2], parts[1]

    return None, None, None


# fetch collection data
def fetch_collection_by_handle(access_token, handle):

    response = execute_graphql_query(
        access_token,
        GET_COLLECTION_QUERY,
        {"query": f"handle:{handle}"},
    )

    if not response:
        return None

    edges = response["data"]["collections"]["edges"]

    if not edges:
        return None

    node = edges[0]["node"]

    return {
        "id": node.get("id", ""),
        "title": node.get("title", ""),
        "description": node.get("description", ""),
    }


# fetch page data
def fetch_page_by_handle(access_token, handle):

    response = execute_graphql_query(
        access_token,
        GET_PAGE_QUERY,
        {"query": f"handle:{handle}"},
    )

    if not response:
        return None

    edges = response["data"]["pages"]["edges"]

    if not edges:
        return None

    node = edges[0]["node"]

    return {
        "id": node.get("id", ""),
        "title": node.get("title", ""),
        "description": node.get("bodySummary", ""),
    }


# fetch article data
def fetch_article_by_handle(access_token, article_handle, blog_handle):

    response = execute_graphql_query(
        access_token,
        GET_ARTICLE_QUERY,
        {"query": f"handle:{article_handle}"},
    )

    if not response:
        return None

    edges = response["data"]["articles"]["edges"]

    if not edges:
        return None

    for edge in edges:

        node = edge["node"]

        node_blog = node["blog"]["handle"]

        if blog_handle and node_blog.lower() != blog_handle.lower():
            continue

        return {
            "id": node.get("id", ""),
            "title": node.get("title", ""),
            "description": node.get("body", ""),
        }

    return None


# process single row
def process_row(access_token, row):

    raw_path = row.get("Landing page path", "").strip()

    row["id"] = ""

    row["title"] = ""

    row["description"] = ""

    row["summary"] = ""

    if not raw_path:
        return row, "skipped", None

    typ, handle, blog_handle = extract_handle(raw_path)

    if not typ or not handle:
        return row, "skipped", None

    print(f"\nPROCESSING [{typ}]: {raw_path}")

    data = None

    if typ == "collection":
        data = fetch_collection_by_handle(access_token, handle)

    elif typ == "page":
        data = fetch_page_by_handle(access_token, handle)

    elif typ == "article":
        data = fetch_article_by_handle(
            access_token,
            handle,
            blog_handle
        )

    if data:

        raw_description = data.get("description", "")

        clean_description = clean_html(raw_description)

        row["id"] = data.get("id", "")

        row["title"] = data.get("title", "")

        row["description"] = clean_description

        row["summary"] = ""

        return row, "matched", None

    return row, "not_found", (typ, handle, raw_path)


# main function
def main():

    print("\n" + "=" * 60)

    print("SHOPIFY SEO ENRICHMENT STARTED")

    print("=" * 60)

    access_token = get_access_token()

    if not access_token:
        print("\nSTOPPING SCRIPT BECAUSE AUTH FAILED")
        return

    matched = 0

    skipped = 0

    not_found = []

    not_found_rows = []

    with open(INPUT_CSV, "r", encoding="utf-8") as infile:

        reader = csv.DictReader(infile)

        fieldnames = list(reader.fieldnames or [])

        for col in ["id", "title", "description", "summary"]:
            if col not in fieldnames:
                fieldnames.append(col)

        rows = list(reader)

    # parallel processing
    results = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_index = {
            executor.submit(process_row, access_token, row): i
            for i, row in enumerate(rows)
        }

        for future in as_completed(future_to_index):

            i = future_to_index[future]

            try:

                row, status, meta = future.result()

                results[i] = (row, status, meta)

            except Exception as e:

                print(f"ERROR on row {i}: {e}")

                results[i] = (rows[i], "skipped", None)

    # write enriched output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row, status, meta in results:

            if status == "matched":
                matched += 1

            elif status == "skipped":
                skipped += 1

            elif status == "not_found":
                not_found.append(meta)
                not_found_rows.append(row)
                continue

            writer.writerow(row)

    # write not found rows
    with open(NOT_FOUND_CSV, "w", newline="", encoding="utf-8") as nf:

        writer = csv.DictWriter(nf, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(not_found_rows)

    # final report
    print("\n" + "=" * 60)

    print(f"OUTPUT SAVED TO: {OUTPUT_CSV}")

    print(f"NOT FOUND SAVED TO: {NOT_FOUND_CSV}")

    print(f"\nMATCHED   : {matched}")

    print(f"SKIPPED   : {skipped}")

    print(f"NOT FOUND : {len(not_found)}")

    print("=" * 60)

    # unmatched details
    if not_found:

        print(
            "\n"
            f"{'TYPE':<12}"
            f"{'HANDLE':<55}"
            f"RAW PATH"
        )

        print("-" * 130)

        seen = set()

        for typ, handle, raw in sorted(not_found):

            key = (typ, handle)

            if key in seen:
                continue

            seen.add(key)

            print(
                f"{typ:<12}"
                f"{handle:<55}"
                f"{raw}"
            )


if __name__ == "__main__":

    main()