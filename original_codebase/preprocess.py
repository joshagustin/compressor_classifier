import string
import nltk.tokenize as tokenizer
import time

def preprocess(
    texts: list,
    tokenize: bool = True,
    lowercase: bool = True,
    remove_punctuation: bool = True
):
    start = time.time()

    if tokenize:
        texts = tokenize_texts(texts)
    if lowercase:
        texts = lowercase_tokens(texts)
    if remove_punctuation:
        texts = remove_punctuation_tokens(texts)

    if tokenize:
        texts = detokenize(texts)

    print(f"preprocessed in: {time.time() - start}")
    return texts

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
    print(detokenize(text))