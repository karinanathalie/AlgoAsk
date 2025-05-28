from llama_cpp import Llama

llm = Llama(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

response = llm("Q: What is 2 + 2?\nA:")
print(response["choices"][0]["text"])