import tiktoken

sample_encoder = tiktoken.get_encoding("gpt2") # 使用gpt2 tokenizer
sample_text = "hello,world!"

result = sample_encoder.encode(sample_text)
print(result)