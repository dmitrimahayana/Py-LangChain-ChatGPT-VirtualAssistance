import openai, os

openai.api_key = os.environ.get('CHATGPT_API_KEY')
messages = [
    {'role': 'system', 'content': 'You are friendly chatbot.'},
    {'role': 'user', 'content': 'Hi, my name is Andrea'},
    {'role': 'assistant', 'content': "Nice to meet you, Andrea! How can I assist you today?"},
    {'role': 'user', 'content': 'So who is my name?'}
]


def call_chatgpt_with_memory(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message["content"]


response = call_chatgpt_with_memory(messages)
print(response)
