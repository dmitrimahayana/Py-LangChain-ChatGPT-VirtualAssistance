import os
from pydantic import Field
from typing import List, Mapping, Optional, Any
from langchain.llms.base import LLM
from gpt4all import GPT4All

"""
A custom LLM class that integrates gpt4all models

Arguments:

model_folder_path: (str) Folder path where the model lies
model_name: (str) The name of the model to use (<model name>.bin)
allow_download: (bool) whether to download the model or not

backend: (str) The backend of the model (Supported backends: llama/gptj)
n_threads: (str) The number of threads to use
n_predict: (str) The maximum numbers of tokens to generate
temp: (str) Temperature to use for sampling
top_p: (float) The top-p value to use for sampling
top_k: (float) The top k values use for sampling
n_batch: (int) Batch size for prompt processing
repeat_last_n: (int) Last n number of tokens to penalize
repeat_penalty: (float) The penalty to apply repeated tokens

"""
model_folder_path: str = 'D:\\04 Model\\LLMA'
model_name: str = 'gpt4all'
allow_download: bool = True

# # all the optional arguments

backend: Optional[str] = 'llama'
temp: Optional[float] = 0.7
top_p: Optional[float] = 0.1
top_k: Optional[int] = 40
n_batch: Optional[int] = 8
n_threads: Optional[int] = 4
n_predict: Optional[int] = 256
max_tokens: Optional[int] = 200
repeat_last_n: Optional[int] = 64
repeat_penalty: Optional[float] = 1.18

# initialize the model
gpt4_model_instance: Any = None


model_folder_path: str = model_folder_path
model_name = model_name
allow_download = allow_download

# trigger auto download
self.auto_download()

self.gpt4_model_instance = GPT4All(
    model_name=self.model_name,
    model_path=self.model_folder_path,
)


def auto_download(self) -> None:
    """
    This method will download the model to the specified path
    reference: python.langchain.com/docs/modules/model_io/models/llms/integrations/gpt4all
    """

    # import all the required dependencies
    import requests
    from tqdm import tqdm

    # see whether the model name has .bin or not

    model_name = (
        f"{self.model_name}.bin"
        if not self.model_name.endswith(".bin")
        else self.model_name
    )

    download_path = os.path.join(self.model_folder_path, model_name)

    if not os.path.exists(download_path):
        if self.allow_download:

            # send a GET request to the URL to download the file.
            # Stream it while downloading, since the file is large

            try:
                url = f'http://gpt4all.io/models/{model_name}'

                response = requests.get(url, stream=True)
                # open the file in binary mode and write the contents of the response
                # in chunks.

                with open(download_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8912)):
                        if chunk: f.write(chunk)

            except Exception as e:
                print(f"=> Download Failed. Error: {e}")
                return

            print(f"=> Model: {self.model_name} downloaded sucessfully 🥳")

        else:
            print(
                f"Model: {self.model_name} does not exists in {self.model_folder_path}",
                "Please either download the model by allow_download = True else change the path"
            )
