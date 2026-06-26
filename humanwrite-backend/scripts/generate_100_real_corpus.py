import os
import random
import urllib.request
import re

def write_corpus(category_folder, texts, limit=100):
    os.makedirs(category_folder, exist_ok=True)
    count = 0
    # Keep the first 3 files that were already carefully crafted
    existing_count = 3
    for i in range(1, 4):
        if not os.path.exists(os.path.join(category_folder, f"real_{i}.txt")):
            existing_count = i - 1
            break
            
    start_index = 4
    for text in texts:
        if count >= (limit - 3):
            break
        # Skip very short texts
        if len(text.split()) < 20:
            continue
        filepath = os.path.join(category_folder, f"real_{start_index + count}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        count += 1
    print(f"Wrote {count} new files to {category_folder} (Total: {count + 3})")

def parse_wiki_dumps():
    filepaths = [
        'data/training_corpus/raw_html_dumps/wiki_sains.txt',
        'data/training_corpus/raw_html_dumps/wiki_bumi.txt',
        'data/training_corpus/raw_html_dumps/wiki_ai.txt'
    ]
    paragraphs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Super aggressive HTML stripping
        clean = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
        clean = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
        clean = re.sub(r'<[^>]+>', '\n', clean)
        
        lines = clean.split('\n')
        for line in lines:
            line = line.strip()
            # Filter valid paragraphs
            if len(line.split()) > 25 and '{' not in line and '}' not in line and 'px|' not in line and not line.startswith('Title:'):
                paragraphs.append(line)
    random.shuffle(paragraphs)
    return paragraphs

def get_smsa_texts():
    url = 'https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        res = urllib.request.urlopen(req)
        content = res.read().decode('utf-8')
        lines = content.split('\n')
        texts = [line.split('\t')[0].strip() for line in lines if '\t' in line and len(line.split('\t')[0].split()) > 20]
        random.shuffle(texts)
        return texts
    except:
        return []

if __name__ == '__main__':
    print("Parsing local wiki dumps...")
    wiki_texts = parse_wiki_dumps()
    
    print("Fetching SmSA dataset...")
    smsa_texts = get_smsa_texts()
    
    # Split the datasets to fill the categories
    # Akademik and Profesional from Wikipedia
    akademik_texts = wiki_texts[:150]
    profesional_texts = wiki_texts[150:300]
    
    # Kreatif and Populer from SmSA (reviews/comments)
    populer_texts = smsa_texts[:150]
    kreatif_texts = smsa_texts[150:300]
    
    # Fallbacks in case wiki wasn't enough
    if len(akademik_texts) < 97:
        akademik_texts += smsa_texts[300:400]
    if len(profesional_texts) < 97:
        profesional_texts += smsa_texts[400:500]
    
    write_corpus('data/corpus_akademik', akademik_texts, 100)
    write_corpus('data/corpus_profesional', profesional_texts, 100)
    write_corpus('data/corpus_populer', populer_texts, 100)
    write_corpus('data/corpus_kreatif', kreatif_texts, 100)
    
    print("Done!")
