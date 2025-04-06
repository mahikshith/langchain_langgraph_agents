from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama

from langchain.schema.runnable import RunnablePassthrough  
from langchain.retrievers.multi_query import MultiQueryRetriever  

MODEL = "llama3.2"
file_path = "./langgraph_presentation.pdf"

# Load the PDF file
loader = PyMuPDFLoader(file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

texts = text_splitter.split_documents(documents)

embed_model = "nomic-embed-text"

embedding_model = OllamaEmbeddings(model=embed_model)
vector_store = FAISS.from_documents(texts, embedding_model)

# Define the LLM
llm = ChatOllama(model=MODEL)

# Prompt for multi-query retriever
query_prompt = PromptTemplate(

    template="""You are a helpful assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",

    input_variables=["question"],
)

retriever = MultiQueryRetriever.from_llm(
    vector_store.as_retriever(),
    prompt=query_prompt,
    llm=llm,
)

# RAG  template


template = """You are a helpful assistant. You will be provided with a question and some context, {context}.
Your task is to answer the question, {question} based on the context provided. \
If the context does not contain enough information to answer the question, \
say "I couldn't find that information in the provided document."""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}  
    | prompt
    | llm  
)

ans = chain.invoke(input="What is langgraph?")
print(ans.content) 