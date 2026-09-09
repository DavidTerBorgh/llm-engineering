import tiktoken
enc = tiktoken.get_encoding("o200k_base")
text = "The patient was admitted to hospital yesterday."
ids = enc.encode(text)
print("tokenizer loaded, vocabulary size:", enc.n_vocab)
print("characters:", len(text))
print("tokens: ", len(ids))
print("pieces: ", [enc.decode([i]) for i in ids])
