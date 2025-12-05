
import os
import re
from typing import Dict, List, Tuple
import bisect

dataset_path = "/Users/shenjiajun/.cache/kagglehub/datasets/michaelarman/poemsdataset/versions/1"

import kagglehub

# Download latest version
path = kagglehub.dataset_download("rtatman/english-word-frequency")
# read the path


def parse_filename(filename: str) -> Tuple[str, str, bool]:
    """
    Parse the poem filename to extract title, author, and translation status.
    
    Args:
        filename: The filename (without .txt extension)

    Returns:
        Tuple[title, author, is_translation]
    """
    if filename.endswith('.txt'):
        filename = filename[:-4]
    
    is_translation = 'Translation' in filename
    
    if 'by' in filename:
        parts = filename.split('by')
        author = parts[-1] 
        title_part = 'by'.join(parts[:-1]) 
    else:
        author = "Unknown"
        title_part = filename
    
    title_part = re.sub(r'^[A-Za-z-]+Poems\d*', '', title_part)
    
    title_part = re.sub(r'Translation(Poem)?', '', title_part)
    
    title_part = re.sub(r'Poem$', '', title_part)
    
    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title_part.strip())
    title = re.sub(r'\s+', ' ', title).strip()
    
    author = re.sub(r'([a-z])([A-Z])', r'\1 \2', author.strip())
    author = re.sub(r'\s+', ' ', author).strip()
    
    if not title:
        title = "Untitled"
    
    return title, author, is_translation


def load_poems_from_forms(forms_list: List[str] = None, verbose: bool = True) -> List[Dict[str, any]]:
    """
    Load poems from the forms directory.    
    
    Args:
        forms_list: List of poem forms to load. If None, load all forms.
        verbose: Whether to show processing progress.

    Returns:
        List[Dict]: A list of dictionaries containing poem information, each dictionary includes:
            - title: Poem title
            - content: Poem content
            - author: Author
            - is_translation: Whether it is a translation
            - form: Poem form
            - filename: Original filename
    """
    poems = []
    forms_path = os.path.join(dataset_path, 'forms')
    
    if not os.path.exists(forms_path):
        print(f"Forms directory does not exist: {forms_path}")
        return poems
    
    if forms_list is None:
        available_forms = [d for d in os.listdir(forms_path) 
                          if os.path.isdir(os.path.join(forms_path, d))]
    else:
        available_forms = [f for f in forms_list 
                          if os.path.isdir(os.path.join(forms_path, f))]
    
    if verbose:
        print(f"Preparing to load {len(available_forms)} forms")
    
    for form_dir in available_forms:
        form_path = os.path.join(forms_path, form_dir)
        
        if verbose:
            print(f"Processing {form_dir}")
        
        try:
            txt_files = [f for f in os.listdir(form_path) if f.endswith('.txt')]
        except OSError as e:
            if verbose:
                print(f"Cannot read directory {form_path}: {e}")
            continue
        
        for txt_file in txt_files:
            try:
                title, author, is_translation = parse_filename(txt_file)
                
                file_path = os.path.join(form_path, txt_file)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
                
                poem = {
                    'title': title,
                    'content': content,
                    'author': author,
                    'is_translation': is_translation,
                    'form': form_dir,
                    'filename': txt_file
                }
                
                poems.append(poem)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {txt_file}: {e}")
                continue
    
    if verbose:
        print(f"Total {len(poems)} poems loaded")
    return poems


def remove_copyright_notices(poem_content: str) -> str:
    lines = poem_content.split('\n')
    filtered_lines = [line for line in lines if '©' not in line]
    return '\n'.join(filtered_lines)

sep_line = ' - - - '
def remove_separator_lines(poem_content: str) -> str:

    lines = poem_content.split('\n')
    filtered_lines = []
    for line in lines:
        if sep_line not in line:
            filtered_lines.append(line)
        else:
            return '\n'.join(filtered_lines), True # stop at the first separator line
    return '\n'.join(filtered_lines), False



def load_words_set(file_path: str) -> list:
    words = []
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    words.append(word)
    elif file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        # conly count > 100000
        words = df[df['count'] > 100000]['word'].astype(str).str.lower().tolist()
        
    return words

def is_english_word_(words, word: str) -> bool:
    # word should be sorted in the words list
    index = bisect.bisect_left(words, word)
    return index < len(words) and words[index] == word

def lemmatize_word(word: str) -> list:
    # A simple lemmatizer for common suffixes
    suffixes = ['ing', 'ed', 'ly', 'es', 's', 'er', 'est']
    all_lematizations = [word]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            all_lematizations.append(word[:-len(suffix)])
    return all_lematizations

def is_english_word(words, word: str) -> bool:
    lemmas = lemmatize_word(word)
    for lemma in lemmas:
        if is_english_word_(words, lemma):
            return True
    return False

def english_word_ratio(words, text: str) -> float:
    words_in_text = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words_in_text:
        return 0.0
    english_word_count = sum(1 for word in words_in_text if is_english_word(words, word.lower()))
    return english_word_count / len(words_in_text)

def filter_english_poems(words, poems: List[Dict[str, any]]) -> List[Dict[str, any]]:
    # with more than 95% English letters
    original_poem_count = len(poems)
    english_poems = []
    for poem in poems:
        content = poem['content']
        if english_word_ratio(words, content) >= 0.8:
            english_poems.append(poem)
    print(f"Filtered {len(english_poems)} English poems out of {original_poem_count} total poems.")
    return english_poems

def english_word_ratio_histogram(poems: List[Dict[str, any]], bins: int = 10) -> Dict[float, int]:
    import matplotlib.pyplot as plt
    histogram = {}
    examples = {}
    for poem in poems:
        ratio = english_word_ratio(poem['content'])
        ratio_rounded = round(ratio, 2)
        histogram[ratio_rounded] = histogram.get(ratio_rounded, 0) + 1
        examples[ratio_rounded] = poem['content']
    ratios = sorted(histogram.keys())
    counts = [histogram[r] for r in ratios]

    present_ratios = [0.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]

    for i, r in enumerate(present_ratios):
        if i % 1 == 0:
            p = examples[r]
            lines_p = p.split('\n')
            if len(lines_p) > 5:
                p = '<br/>'.join(lines_p[:5]) + '<br/>...'
            print(f"| {p}")
            #print("-----")
    plt.bar(ratios, counts)
    plt.xlabel("English Word Ratio")
    plt.ylabel("Number of Poems")
    plt.title("English Word Ratio Histogram")
    plt.show()


def english_word_ratio_histogram_texts(texts: List[str], bins: int = 10) -> Dict[float, int]:
    import matplotlib.pyplot as plt
    histogram = {}
    examples = {}
    for text in texts:
        ratio = english_word_ratio(text)
        ratio_rounded = round(ratio, 2)
        if ratio_rounded < 0.5:
            print(text)
        histogram[ratio_rounded] = histogram.get(ratio_rounded, 0) + 1
        examples[ratio_rounded] = text
    ratios = sorted(histogram.keys())
    counts = [histogram[r] for r in ratios]

    present_ratios = [0.0, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]

    for i, r in enumerate(present_ratios):
        if i % 1 == 0:
            p = examples.get(r, "")
            lines_p = p.split('\n')
            if len(lines_p) > 5:
                p = '<br/>'.join(lines_p[:5]) + '<br/>...'
            print(f"| {p}")
            #print("-----")
    plt.bar(ratios, counts)
    plt.xlabel("English Word Ratio")
    plt.ylabel("Number of Poems")
    plt.title("English Word Ratio Histogram")
    plt.show()

def poems():
    words = load_words_set("unigram_freq.csv")
    words.sort()
    poems = load_poems_from_forms(verbose=False)

    # english_word_ratio_histogram(poems)

    count = 0
    sep_count = 0
    for poem in poems:
        if '©' in poem['content']:
            poem['content'] = remove_copyright_notices(poem['content'])
            count += 1
        poem['content'], has_sep = remove_separator_lines(poem['content'])
        if has_sep:
            sep_count += 1
    print(f"Removed copyright notices from {count} poems.")
    print(f"Found and removed separator line from {sep_count} poems.")

    poems = filter_english_poems(words, poems)

    def save_csv(poems: List[Dict[str, any]], output_file: str):
        import pandas as pd
        df = pd.DataFrame(poems)
        df.to_csv(output_file, index=False)
        print(f"Saved cleaned poems to {output_file}")
        
    save_csv(poems, "cleaned_poems__________.csv")


if __name__ == "__main__":
    poems()