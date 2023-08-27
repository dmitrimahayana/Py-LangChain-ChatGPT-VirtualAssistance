import time

from MyCustomLLM import MyCustomLLM

model_folder_path = "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/"
model_name = "llama-2-7b-chat.ggmlv3.q4_0.bin"

llm = MyCustomLLM(
    model_folder_path=model_folder_path,
    model_name=model_name,
    allow_download=True
)

time.sleep(1)
while True:
    query = input('Enter your Query: ')
    if query == "exit":
        break
    response = llm(
        query
    )
    print("AI: "+response)
