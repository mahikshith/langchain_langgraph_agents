# 1. reading the pdf using PyMuPDF
# 2. splitting the text into chunks using langchain text splitter 
# 3. embedding the text using langchain embedding model
# 4, storing the embeddings in a vector store using langchain vector store
# 5. using langchain retriever to retrieve the relevant chunks from the vector store 
# 6. using ollama to generate the answer using the retrieved chunks as context  

# step 1 : 

#from langchain.document_loaders import PyMuPDFLoader  -- depricated 

from langchain_community.document_loaders import PyMuPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter 


# from langchain.embeddings import OllamaEmbeddings  -- depricated

from langchain_community.embeddings import OllamaEmbeddings

# from langchain.vectorstores import FAISS  -- depricated 

from langchain_community.vectorstores import FAISS

MODEL = "llama3.2" 

file_path =r"C:\Users\mahik\Documents\Git_serious\langchain_langgraph_agents\langgraph_presentation.pdf" 

file_path = "./langgraph_presentation.pdf"

# load the pdf file using PyMuPDFLoader
loader = PyMuPDFLoader(file_path)
documents = loader.load()

# print(f"Number of documents loaded: {len(documents)}") 

# print(documents[3].page_content)

# split the text into chunks using langchain text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20) 
texts = text_splitter.split_documents(documents)

# ollama embedding model using faiss vector store 
embed_model = "nomic-embed-text"

embedding_model = OllamaEmbeddings(model=embed_model)
vector_store = FAISS.from_documents(texts, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

