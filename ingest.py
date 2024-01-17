from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

class LLMVectorDB:
  def __init__(self, data_path, chromadb_path, faissdb_path, model_name, device='cuda'):
        self.data_path = data_path
        self.chromadb_path = chromadb_path
        self.faissdb_path = faissdb_path
        self.model_name = model_name
        self.device = device

  # Create vector database
  def chroma_db(self):
      loader = DirectoryLoader(self.data_path,
                              glob='*.pdf',
                              loader_cls=PyPDFLoader)

      documents = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                                    chunk_overlap=128)
      texts = text_splitter.split_documents(documents)
      print(texts)

      embeddings = HuggingFaceEmbeddings(model_name=self.model_name,
                                        model_kwargs={'device': 'cuda'})

      db = Chroma.from_documents(texts, embeddings, persist_directory=self.chromadb_path)
      db.persist()
      
  def faiss_db(self):
      loader = DirectoryLoader(self.data_path,
                              glob='*.pdf',
                              loader_cls=PyPDFLoader)

      documents = loader.load()
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
      texts = text_splitter.split_documents(documents)

      embeddings = HuggingFaceEmbeddings(model_name=self.model_name,
                                        model_kwargs={'device': 'cuda'})

      db = FAISS.from_documents(texts, embeddings)
      db.save_local(self.faissdb_path)

if __name__ == "__main__":
  llm_vector_db = LLMVectorDB(
       
        data_path='/content/drive/MyDrive/LLM/Fabric/docs',
        faissdb_path='/content/drive/MyDrive/LLM/VectorDB/FaissDB',
        chromadb_path='/content/drive/MyDrive/LLM/VectorDB/ChromaDB',        
        model_name='sentence-transformers/all-MiniLM-L6-v2' ) 
             
  llm_vector_db.chroma_db()

