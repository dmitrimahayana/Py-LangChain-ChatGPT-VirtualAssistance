from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os


def define_model():
    # LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=os.environ.get('CHATGPT_API_KEY'),
    )

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                f"You are a formal assistant that expert in fashion taxonomy. "
                f"You know how to find the trending keywords for fashion products. "
                f"Remove any explanation or description. Please use comma as separator of the result and no jokes answer."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory)
    return conversation


def find_keyword(conversation, product_name):
    # Get OpenAI Response
    response = conversation(
        {
            "question": f"what are the trending synonyms of {product_name}?"
        })
    list_keyword = response['text']
    print("trending keywords of ", product_name, ": ", list_keyword)
    return list_keyword


if __name__ == "__main__":
    conv = define_model()
    find_keyword(conv, "puffer coat")
    find_keyword(conv, "faux fur coat")
    find_keyword(conv, "blazer coat")
