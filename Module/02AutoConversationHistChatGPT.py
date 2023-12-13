import openai
import os

openai.api_key = os.environ.get('CHATGPT_API_KEY')
messages = [
    {'role': 'system', 'content': 'You are friendly chatbot.'},
    {'role': 'user', 'content': 'Hi, my name is Andrea'},
    {'role': 'assistant', 'content': "Nice to meet you, Andrea! How can I assist you today?"}]
context = messages


def call_chatgpt_with_memory(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message["content"]


def chatgpt_conversation(prompt):
    context.append({'role': 'user', 'content': f"{prompt}"})
    response = call_chatgpt_with_memory(context)
    context.append({'role': 'assistant',
                    'content': f"{response}"})
    return response


response = chatgpt_conversation("Hi bot you remember my name right?")
print(response)