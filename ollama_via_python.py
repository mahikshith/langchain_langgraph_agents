# pip install ollama 

# usisng python to call ollama model without using ollama endpoints 

# use faiss-cpu or langchain-astra db for vector store for now 

import ollama 

# response = ollama.list() 

# print(response) 

model = "llama3.2"

# olllama chat  

response = ollama.chat(model = model,
                       messages = [
                           {"role": "user", "content": "How to negotaite a raise with HR, explain in 5 sentences"}],stream = False )

# print(response["message"]["content"])

 # using ollama generate 

ans = ollama.generate(model = model,
                       prompt = "How to negotaite a raise with HR, explain in 5 sentences with bullet points",
                       stream = False)

print(ans["response"]) 

