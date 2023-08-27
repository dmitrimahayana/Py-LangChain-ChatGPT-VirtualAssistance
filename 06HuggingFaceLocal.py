from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(

    model_id="bigscience/bloom-1b7",
    task="text-generation",
    device=0,
    # model_kwargs={"temperature": 0, "max_length": 64},
)

paragraph = "your name is dmitri and you live in Jember. \
you have 2 daughters and they really lovely. \
their names are naisha and nafeesa"
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
memory.save_context({"input": "Do you remember me?"}, {"output": f"{paragraph}"})
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
response = conversation.predict(input="Who is my daughter?")
print(response)

# template = """Question: {question}
# Answer: Let's think step by step."""
# prompt = PromptTemplate.from_template(template)
# chain = prompt | llm
# question = "What is electroencephalography?"
#
# print(chain.invoke({"question": question}))