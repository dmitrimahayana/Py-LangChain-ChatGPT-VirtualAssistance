from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(

    model_id="bigscience/bloom-1b7",
    task="text-generation",
    device=0,
    model_kwargs={"max_length": 100},
)

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
response = conversation.predict(input="Hello, my name is Andrea and i live at jember")
print(response)
response = conversation.predict(input="What is 1+1?")
print(response)
response = conversation.predict(input="where do i live?")
print(response)

# paragraph = "your name is dmitri and you live in Jember. \
# you have 2 daughters and they really lovely. \
# their names are naisha and nafeesa"
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
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
# response = conversation.predict(input="what is 2 + 3?")
# print(response)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)
# chain = prompt | llm
# question = "What is electroencephalography?"
#
# print(chain.invoke({"question": question}))