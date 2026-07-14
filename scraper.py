"""
TASK 1: WEB SCRAPING
---------------------
Generic, reusable scraper. Swap the CONFIG block below to point at any
public site and it will collect the fields you define into a CSV.

Default config is wired to https://books.toscrape.com (a free, scraping-
friendly practice site) so the script runs successfully out of the box.
Replace the CONFIG values with your real target site/selectors.

Run: python3 scraper.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys

# ----------------------- CONFIG: edit this for your dataset -----------------------
CONFIG = {
    "base_url": "https://books.toscrape.com/catalogue/page-{}.html",
    "num_pages": 3,                 # how many pages to scrape
    "item_selector": "article.product_pod",   # CSS selector for each item/card
    "fields": {                     # field_name: (css_selector, attribute or "text")
        "title": ("h3 a", "title"),
        "price": ("p.price_color", "text"),
        "availability": ("p.instock.availability", "text"),
        "rating_class": ("p.star-rating", "class"),
    },
    "request_delay_seconds": 1,     # be polite to the server
    "output_csv": "scraped_data.csv",
}
# ------------------------------------------------------------------------------------

HEADERS = {"User-Agent": "Mozilla/5.0 (educational scraping project)"}


def scrape_page(url, item_selector, fields):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    items = soup.select(item_selector)

    records = []
    for item in items:
        record = {}
        for field_name, (selector, attr) in fields.items():
            el = item.select_one(selector)
            if el is None:
                record[field_name] = None
                continue
            if attr == "text":
                record[field_name] = el.get_text(strip=True)
            elif attr == "title":
                record[field_name] = el.get("title", el.get_text(strip=True))
            elif attr == "class":
                record[field_name] = " ".join(el.get("class", []))
            else:
                record[field_name] = el.get(attr)
        records.append(record)
    return records


def main():
    all_records = []
    for page_num in range(1, CONFIG["num_pages"] + 1):
        url = CONFIG["base_url"].format(page_num)
        print(f"Scraping page {page_num}: {url}")
        try:
            records = scrape_page(url, CONFIG["item_selector"], CONFIG["fields"])
        except requests.RequestException as e:
            print(f"  Failed to fetch {url}: {e}", file=sys.stderr)
            continue

        if not records:
            print(f"  No items found on page {page_num}, stopping.")
            break

        all_records.extend(records)
        time.sleep(CONFIG["request_delay_seconds"])

    df = pd.DataFrame(all_records)
    df.drop_duplicates(inplace=True)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"\nDone. Scraped {len(df)} records -> {CONFIG['output_csv']}")
    print(df.head())


if __name__ == "__main__":
    main()
