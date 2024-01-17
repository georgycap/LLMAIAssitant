from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from chromadb import PersistentClient
#from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.llms import LlamaCpp
from langchain.llms import CTransformers

CHROMA_DB_PATH = '/content/drive/MyDrive/LLM/VectorDB/ChromaDB'
#DB_FAISS_PATH = 'med_journal_faissdb/med_faissdb'
LLM_MODEL_PATH = '/content/drive/MyDrive/LLM/Llama Model/llama-2-7b-chat.ggmlv3.q8_0.bin'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():

  n_gpu_layers = 48  # Metal set to 1 is enough.
  n_batch = 224  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

  # Make sure the model path is correct for your system!
  '''llm = LlamaCpp(
    model_path="/content/drive/MyDrive/LLMModel/llama-2-13b-chat.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_threads=8,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
  )'''

# Load the locally downloaded model here
  llm = CTransformers(
      model = LLM_MODEL_PATH,
      model_type="llama",
      max_new_tokens = 512,
      temperature = 0.5
  )
      
  
  return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function = embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi John, Welcome to your Chroma DB Medical Assistant. How can I help you?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    srcs = []
    for i in sources:
      srcs.append(i.metadata)
    #mtdata = []
    if sources:
        #answer += f"\nSources:" + str(sources)              
        answer = f"\nAnswer : " + str(answer)
        await cl.Message(content=answer).send()
        count = 0
        for item in srcs:
          count += 1
          await cl.Message(content=f"\nSource"+str(count)+"\n\nPage No:"+ str(item['page'])+ "\nLink:" + str(item['source'])).send()
      
        
    else:
        answer = "\nNo sources found"
        await cl.Message(content=answer).send()


