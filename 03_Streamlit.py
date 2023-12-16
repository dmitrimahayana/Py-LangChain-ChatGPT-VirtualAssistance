import streamlit as st
import requests
import io
from PIL import Image
from IPython.display import display

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": "Bearer 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"}


def query_stabilitydiff(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


with st.sidebar:
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot - Text to Image")
st.caption("🚀 A Streamlit chatbot powered by Stable Diffusion")
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "What kind of image that I need to draw?"}]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

st.chat_message("assistant").write("What kind of image that I need to draw?")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if prompt := st.chat_input():
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    # client = OpenAI(api_key=openai_api_key)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    image_bytes = query_stabilitydiff({
        "inputs": prompt,
    })
    image = Image.open(io.BytesIO(image_bytes))
    msg = f'here is your image related to "{prompt}"'
    st.session_state.messages.append({"role": "assistant", "content": msg, "prompt": prompt, "image": image})
    # st.session_state.messages.append({"role": "assistant", "content": image})
    # st.chat_message("assistant").write(msg)
    # st.chat_message("assistant").image(image, caption=prompt, use_column_width=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").write(prompt)
        elif message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        if 'image' in message:
            st.chat_message("assistant").image(message["image"], caption=message["prompt"], use_column_width=True)