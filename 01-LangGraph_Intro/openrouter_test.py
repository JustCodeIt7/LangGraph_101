# %%
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv
from dotenv import load_dotenv
from rich import print
#%%
load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=['question'])
model_name = 'google/gemini-2.0-flash-001'
llm = ChatOpenAI(
    openai_api_key=getenv('OPENROUTER_API_KEY', 'KEY_IF_NOT_SET'),
    openai_api_base=getenv('OPENROUTER_API_BASE', 'https://openrouter.ai/api/v1'),
    model_name=model_name
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = 'What NFL team won the Super Bowl in the year Justin Bieber was born?'

response = llm_chain.invoke(question)
print(response)
# %%


# %%
