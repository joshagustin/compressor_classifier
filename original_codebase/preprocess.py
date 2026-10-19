import string
import nltk.tokenize as tokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from configs import PreprocessingConfig
from typing import Union
import time

snowball = SnowballStemmer("english")

def preprocess(
    texts: list,
    config: Union[PreprocessingConfig, list]
):
    start = time.time()

    single = not isinstance(config, (list, tuple))
    configs = [config] if single else list(config)

    base = None
    if any(c.normalize for c in configs):
        base = tokenize_texts(texts)
        base = lowercase_tokens(base)
        base = remove_punctuation_tokens(base)

    # Cache token lists by (stopwords_removed, stemmed) so shared steps run once
    cache = {(False, False): base}

    def get_tokens(stop: bool, stem: bool):
        key = (stop, stem)
        if key not in cache:
            if stem:
                cache[key] = stem_tokens(get_tokens(stop, False))
            else:
                cache[key] = remove_stopword_tokens(base)
        return cache[key]

    results = []
    for c in configs:
        if c.normalize:
            results.append(detokenize(get_tokens(c.remove_stopwords, c.stem)))
        else:
            results.append(texts)

    print(f"preprocessed in: {time.time() - start}")
    return results[0] if single else results

def tokenize_texts(texts: list) -> list:
    return [tokenizer.word_tokenize(text) for text in texts] 

def lowercase_tokens(texts: list) -> list:
    return [
        [token.lower() for token in text] 
        for text in texts
    ]
    
def remove_punctuation_tokens(texts: list) -> list:
    return [
        [token for token in text if token not in string.punctuation]
        for text in texts
    ]

def remove_stopword_tokens(texts: list) -> list:
    words = set(stopwords.words("english"))
    return [
        [token for token in text if token not in words]
        for text in texts
    ]

def stem_tokens(texts: list) -> list:
    return [
        [snowball.stem(token) for token in text]
        for text in texts
    ]

def detokenize(texts: list) -> list:
    return [" ".join(text) for text in texts]

if __name__ == "__main__":
    text = ["The quick! brown.", "Fox jumps!", "Over the\n lazy dog."]
    text = tokenize_texts(text)
    print(text)
    text = lowercase_tokens(text)
    print(text)
    text = remove_punctuation_tokens(text)
    print(text)

    text = remove_stopword_tokens(text)
    print(text)
    text = stem_tokens(text)
    print(detokenize(text))