import requests
import time
import json


def recover_abstract(inv_index):
    inv_map = {}
    for word, positions in inv_index.items():
        for pos in positions:
            inv_map[pos] = word
    return ' '.join([inv_map[i] for i in range(len(inv_map))])


FIELD_ID = "C41008148"  # Computer science
PER_PAGE = 200
TARGET_NUM = 9000
OUTPUT_FILE = "openalex_9000.json"

papers = []
page = 1

while len(papers) < TARGET_NUM:
    print(f"Fetching page {page} (current total: {len(papers)})")
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"concepts.id:{FIELD_ID},has_abstract:true",
        "per-page": PER_PAGE,
        "page": page
    }

    try:
        r = requests.get(url, params=params, headers={"User-Agent": "OpenAlex-PaperFetcher/1.0"})
        r.raise_for_status()
        data = r.json()

        if "results" not in data:
            print("⚠️ No 'results' in response. Skipping this page.")
            page += 1
            time.sleep(1)
            continue

        results = data["results"]

        for item in results:
            try:
                abs_inv = item.get("abstract_inverted_index", {})
                if not abs_inv:
                    continue
                abstract = recover_abstract(abs_inv)

                papers.append({
                    "id": item["id"],
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "references": item.get("referenced_works", []),
                    "publication_year": item.get("publication_year")
                })

                if len(papers) >= TARGET_NUM:
                    break
            except Exception as e:
                print(f"Skipped one item due to error: {e}")
                continue

        page += 1
        time.sleep(1)

    except Exception as e:
        print(f"Error on page {page}: {e}")
        page += 1
        time.sleep(2)
        continue

print(f"Done! Total papers saved: {len(papers)}")

# Save as .json
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)
