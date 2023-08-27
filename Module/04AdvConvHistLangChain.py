# OpenAI key from environment
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import os

paragraph = "your name is dmitri and you live in Jember. \
you have 2 daughters and they really lovely. \
their names are naisha and nafeesa"

llm = OpenAI(
    openai_api_key=os.environ.get('CHATGPT_API_KEY')
)
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