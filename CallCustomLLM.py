import time

from MyCustomLLM import MyCustomLLM

model_folder_path = "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/"
model_name = "llama-2-7b-chat.ggmlv3.q4_0.bin"

llm = MyCustomLLM(
    model_folder_path=model_folder_path,
    model_name=model_name,
    allow_download=True
    # model_folder_path=model_folder_path,
)

time.sleep(1)
response = llm("My name is dmitri and i live in Jember. I have 2 daughters and they are really lovely. Their names are naisha and nafeesa")
print(response)
response = llm("Who is naisha and nafeesa?")
print(response)
response = llm("where do i live?")
print(response)

# while True:
#     query = input('Enter your Query: ')
#     if query == "exist":
#         break
#     response = llm(
#         query
#     )
#     print(response)
