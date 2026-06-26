import os
import urllib.request
import re

def download_and_save(url, output_path):
    print(f"Downloading {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        content = res.read().decode('utf-8')
        
        # Clean up CSV/TSV to just get the text column
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            parts = re.split(r'\t|,', line)
            if parts and len(parts[0]) > 20:
                clean_lines.append(parts[0].strip())
                
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(clean_lines))
        print(f"Saved {len(clean_lines)} lines to {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def clean_wiki_html(input_path, output_path):
    if not os.path.exists(input_path):
        return
    print(f"Cleaning {input_path}...")
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Strip HTML tags aggressively
    clean = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
    clean = re.sub(r'<script.*?</script>', '', clean, flags=re.DOTALL)
    clean = re.sub(r'<[^>]+>', '\n', clean)
    
    lines = clean.split('\n')
    valid_lines = []
    for line in lines:
        line = line.strip()
        if len(line.split()) > 15 and '{' not in line and '}' not in line and 'px|' not in line:
            valid_lines.append(line)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_lines))
    print(f"Saved {len(valid_lines)} clean paragraphs to {output_path}")

if __name__ == '__main__':
    os.makedirs('data/training_corpus/indo_nlu_bulk', exist_ok=True)
    os.makedirs('data/training_corpus/id_articles', exist_ok=True)
    
    urls = [
        ('https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv', 'data/training_corpus/indo_nlu_bulk/smsa_full.txt'),
        ('https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/emot_emotion-twitter/train_preprocess.csv', 'data/training_corpus/indo_nlu_bulk/emot_full.txt'),
        ('https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/casa_absa-prosa/train_preprocess.csv', 'data/training_corpus/indo_nlu_bulk/casa_full.txt')
    ]
    
    for url, path in urls:
        download_and_save(url, path)
        
    wiki_files = [
        ('data/training_corpus/raw_html_dumps/wiki_sains.txt', 'data/training_corpus/id_articles/wiki_sains_clean.txt'),
        ('data/training_corpus/raw_html_dumps/wiki_bumi.txt', 'data/training_corpus/id_articles/wiki_bumi_clean.txt'),
        ('data/training_corpus/raw_html_dumps/wiki_ai.txt', 'data/training_corpus/id_articles/wiki_ai_clean.txt')
    ]
    
    for in_path, out_path in wiki_files:
        clean_wiki_html(in_path, out_path)
    
    print("Done enriching training_corpus!")
