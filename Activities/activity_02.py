## Activity 2 — Tokenization Practical

# Tokenize a sentence, then try a few different words and phrases.

# - Is one word always one token? No common words are single tokens, but long words or complex words maybe split into subwords and those subwords get converted into tokens.
# - Why are some words split? For efficiency.
# - What happens to punctuation? They are also converted to tokens.
# - Why does the model need token IDs? Cuz machines only processes numbers not words.

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained("bert-base-uncased")

texts = [
    "Replace me by any text you'd like.",
    "Tokenization decomposes uncommon words into subword units!",
    "unbelievably supercalifragilisticexpialidocious"
]

for text in texts:
    print("=" * 60)
    print(f"Original Text: '{text}'")
    
    # 1. Tokenize into subword strings
    tokens = tokenizer.tokenize(text)
    print(f"Tokens:        {tokens}")
    
    # 2. Convert tokens to numerical IDs with special tokens ([CLS], [SEP])
    encoded_input = tokenizer(text, return_tensors='pt')
    input_ids = encoded_input['input_ids'][0].tolist()
    print(f"Token IDs:     {input_ids}")
    
    # 3. Decode IDs back to text
    decoded_text = tokenizer.decode(input_ids)
    print(f"Decoded Text:  '{decoded_text}'")
    
    # 4. Pass through model
    output = model(**encoded_input)
    print(f"Model Output Last Hidden State Shape: {output.last_hidden_state.shape}")

print("=" * 60)
