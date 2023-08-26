from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

llm = HuggingFacePipeline.from_model_id(

    model_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)
chain = prompt | llm
question = "What is electroencephalography?"

print(chain.invoke({"question": question}))