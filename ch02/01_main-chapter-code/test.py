# from importlib.metadata import version
# import os
# import urllib.request
# import re
#
# print("torch version:", version("torch"))
# print("tiktoken version:", version("tiktoken"))
#
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
#
# print("Total number of character:", len(raw_text))
# print(raw_text[:99])
#
# """tokenizer text"""
# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(preprocessed[:30])
# print(len(preprocessed))
#
# """Creating a vocabulary"""
# # 所有的词汇表
# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
#
# print(vocab_size)  # 1130，则词汇表大小为 1130
#
# vocab = {token: integer for integer, token in enumerate(all_words)}
#
# # 查看前 50个
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break
#
#
# class SimpleTokenizerV1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i: s for s, i in vocab.items()}
#
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#
#         preprocessed = [
#             item.strip() for item in preprocessed if item.strip()
#         ]
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return text
#
#
# # 实例化
# tokenizer = SimpleTokenizerV1(vocab)
#
# text = """"It's the last he painted, you know,"
#            Mrs. Gisburn said with pardonable pride."""
# # encode
# ids = tokenizer.encode(text)
# print(ids)
# # decode
# print(tokenizer.decode(ids))
#
#
#
# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i: s for s, i in vocab.items()}
#
#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [
#             item if item in self.str_to_int
#             else "<|unk|>" for item in preprocessed
#         ]
#
#         ids = [self.str_to_int[s] for s in preprocessed]
#         return ids
#
#     def decode(self, ids):
#         text = " ".join([self.int_to_str[i] for i in ids])
#         # Replace spaces before the specified punctuations
#         text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
#         return text
#
# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"]) #
# vocab = {token:integer for integer,token in enumerate(all_tokens)}
#
# tokenizer = SimpleTokenizerV2(vocab)
#
# text1 = "Hello, do you like tea haha?"
# text2 = "In the sunlit terraces of the palace."
#
# text = " <|endoftext|> ".join((text1, text2))
# print(f"merged text: {text}")
#
#
# ids = tokenizer.encode(text)
# print(f"encoded text:{ids}")
# decoded_text = tokenizer.decode(ids)
# print(f"decodec text:{decoded_text}")


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