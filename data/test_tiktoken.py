import tiktoken

sample_encoder = tiktoken.get_encoding("gpt2") # 使用gpt2 tokenizer
# sample_encoder = tiktoken.get_encoding("cl100k_base")

# sample_text = "hello world!"
# sample_text = "我觉得我是"
sample_text = '<|endoftext|>'

result = sample_encoder.encode(sample_text)
print(result)

for i,x in enumerate(result):
    print(sample_encoder.decode(result[i:i+1]))

print(sample_encoder.decode(result))