# OpenAI key from environment
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os

llm = OpenAI(openai_api_key=os.environ.get('CHATGPT_API_KEY'))
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
