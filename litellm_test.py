from langchain_litellm import ChatLiteLLM
from rich import print
llm = ChatLiteLLM(model_name='ollama/llama3.2', temperature=0)


r =llm.invoke("Hello, how are you?")
print(r)