"""BytePair encoding"""
import importlib
import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print("intergers",integers)
string = tokenizer.decode(integers)
print("string,",string)

text = "Akwirw ier"
ids = tokenizer.encode(text)
print(ids)
for i in ids:
    print(i,'\t',tokenizer.decode([i]))



with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))