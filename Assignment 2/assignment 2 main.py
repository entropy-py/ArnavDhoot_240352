import requests
import weaviate
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq  # if you're using the langchain_groq integration
import os
from dotenv import load_dotenv
load_dotenv() 
url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/88bd99222d46f763957c2873e48bbba2f3e4b36a/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
from langchain_cohere import CohereEmbeddings
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-v3.0"
)
vectorstore = FAISS.from_documents(docs, embeddings)

groq_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.5,
    max_tokens=200,
    groq_api_key=os.environ["GROQ_API_KEY"],
)
retriever=vectorstore.as_retriever()


rag_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    chain_type="stuff",  # Stuff the context into the prompt
    retriever=retriever,
    return_source_documents=False  # Optional: set to True to also get the source docs
)
#Summarize the main points of Biden's state of union address 2023 speec

query = input("Enter question- ")
terminate='exit'
while(query!=terminate):
  context = retriever.invoke(query)
  resp = rag_chain.invoke(query)
  print("\nAnswer:")
  print(resp['result'])
  query = input("Enter question- ")
