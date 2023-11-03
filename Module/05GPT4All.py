from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationChain

local_path = (
    "C:/Users/dmitr/AppData/Local/nomic.ai/GPT4All/llama-2-7b-chat.ggmlv3.q4_0.bin"  # replace with your desired local file path
)

# Callbacks support token-wise streaming
# callbacks = [StreamingStdOutCallbackHandler()]

# If you want to use a custom model add the backend parameter
# Check https://docs.gpt4all.io/gpt4all_python.html for supported backends
llm = GPT4All(
    model=local_path,
    backend="llama",
    # backend="gptj",
    # callbacks=callbacks,
    verbose=True)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
# question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
# llm_chain.run(question)

# paragraph = "your name is dmitri and you live in Jember. \
# you have 2 daughters and they really lovely. \
# their names are naisha and nafeesa"
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
# memory.save_context({"input": "Hello"}, {"output": "What's up"})
# memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
# memory.save_context({"input": "Do you remember me?"}, {"output": f"{paragraph}"})
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
# response = conversation.predict(input="Who is my daughter?")
# print(response)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    # verbose=True
)
response = conversation.predict(input="My name is dmitri and i live in Jember. I have 2 daughters and they are really lovely. Their names are naisha and nafeesa")
print(response)
response = conversation.predict(input="Who is naisha and nafeesa?")
print(response)
response = conversation.predict(input="where do i live?")
print(response)
response = conversation.predict(input="what is the synonyms of classic wool peacoat?")
print(response)