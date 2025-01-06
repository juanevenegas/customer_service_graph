from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import random

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")