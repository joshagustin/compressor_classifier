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
    text = ["From: lerxst@wam.umd.edu (where's my thing)\nSubject: WHAT car is this!?\nNntp-Posting-Host: rac3.wam.umd.edu\nOrganization: University of Maryland, College Park\nLines: 15\n\n I was wondering if anyone out there could enlighten me on this car I saw\nthe other day. It was a 2-door sports car, looked to be from the late 60s/\nearly 70s. It was called a Bricklin. The doors were really small. In addition,\nthe front bumper was separate from the rest of the body. This is \nall I know. If anyone can tellme a model name, engine specs, years\nof production, where this car is made, history, or whatever info you\nhave on this funky looking car, please e-mail.\n\nThanks,\n- IL\n   ---- brought to you by your neighborhood Lerxst ----\n\n\n\n\n"]
    print(text[0])
    text = tokenize_texts(text)
    text = lowercase_tokens(text)
    text = remove_punctuation_tokens(text)
    print(text[0])
    text = remove_stopword_tokens(text)
    print(text[0])
    text = stem_tokens(text)
    print(text[0])
    print(detokenize(text)[0])