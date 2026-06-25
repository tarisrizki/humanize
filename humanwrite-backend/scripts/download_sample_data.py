import os
import json
import httpx
from bs4 import BeautifulSoup
from pathlib import Path
import xml.etree.ElementTree as ET

def fetch_arxiv_journals(limit=15):
    print(f"Fetching {limit} journals from arXiv...")
    url = f"http://export.arxiv.org/api/query?search_query=all:computer&start=0&max_results={limit}"
    headers = {'User-Agent': 'HumanWriteBot/1.0 (test@example.com)'}
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
    root = ET.fromstring(response.text)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    entries = root.findall('atom:entry', namespace)
    
    docs = []
    for entry in entries:
        title = entry.find('atom:title', namespace).text.strip()
        summary = entry.find('atom:summary', namespace).text.strip()
        docs.append(f"Title: {title}\n\nAbstract:\n{summary}")
    return docs

def fetch_devto_blogs(limit=15):
    print(f"Fetching {limit} blogs from dev.to...")
    url = f"https://dev.to/api/articles?per_page={limit}"
    headers = {'User-Agent': 'HumanWriteBot/1.0 (test@example.com)'}
    response = httpx.get(url, headers=headers)
    articles = response.json()
    
    docs = []
    for art in articles:
        # Dev.to API returns the description, let's try fetching the actual URL for more text
        art_url = art['url']
        try:
            art_resp = httpx.get(art_url, follow_redirects=True)
            soup = BeautifulSoup(art_resp.text, 'html.parser')
            # The body is usually in an article tag or div with class 'crayons-article__body'
            body = soup.find('div', class_='crayons-article__body')
            if body:
                text = body.get_text(separator='\n', strip=True)
                docs.append(f"Title: {art['title']}\n\n{text}")
            else:
                docs.append(f"Title: {art['title']}\n\n{art['description']}")
        except Exception as e:
            docs.append(f"Title: {art['title']}\n\n{art['description']}")
    return docs

def fetch_wikipedia(lang="en", limit=10):
    print(f"Fetching {limit} articles from Wikipedia ({lang})...")
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&exchars=3000&format=json&grnlimit={limit}"
    headers = {'User-Agent': 'HumanWriteBot/1.0 (test@example.com)'}
    response = httpx.get(url, headers=headers)
    data = response.json()
    
    docs = []
    if 'query' in data and 'pages' in data['query']:
        for page_id, page_info in data['query']['pages'].items():
            if 'extract' in page_info:
                docs.append(f"Title: {page_info['title']}\n\n{page_info['extract']}")
    return docs

def main():
    backend_dir = Path(__file__).resolve().parent.parent
    corpus_dir = backend_dir / "data" / "training_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    journals = fetch_arxiv_journals(limit=20)
    blogs = fetch_devto_blogs(limit=20)
    wiki_en = fetch_wikipedia("en", limit=20)
    wiki_id = fetch_wikipedia("id", limit=20)
    
    all_docs = []
    for idx, doc in enumerate(journals):
        all_docs.append(("journal", idx, doc))
    for idx, doc in enumerate(blogs):
        all_docs.append(("blog", idx, doc))
    for idx, doc in enumerate(wiki_en):
        all_docs.append(("wiki_en", idx, doc))
    for idx, doc in enumerate(wiki_id):
        all_docs.append(("wiki_id", idx, doc))
        
    all_docs = all_docs[:50]
        
    for category, idx, text in all_docs:
        filename = f"{category}_{idx+1}.txt"
        filepath = corpus_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
            
    print(f"\nSuccessfully downloaded and saved {len(all_docs)} documents to {corpus_dir}")

if __name__ == '__main__':
    main()
