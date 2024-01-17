from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from chromadb import PersistentClient
#from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
#import openai




OpenAI_API_KEY = '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'
os.environ["OPENAI_API_KEY"] = OpenAI_API_KEY

'''
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key = OpenAI_API_KEY,
)
'''

CHROMA_DB = 'med_journal_chromadb/med_chromadb'

custom_prompt_template = """Use the following pieces of information to answer the user's question. 
Must show both ICD-11 codes and ICD-10 codes for the medical terms with the answer. 
The ICD codes should map with the medical terms in the answer.
If you don't know the ICD-11 just say "ICD-11 unavailable".
Your response should be atleast 5 sentences.
If the question is out of the context you should generate a response based on it from the web, considering the user's query and the context of the conversation.
Be transparent and inform the user that the answer was not found in the documents and don't show sources.
Show a disclaimer that it is a curated content from web.

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
'''def load_llm():
  llm = OpenAI(openai_api_key=OpenAI_API_KEY,   
        model="text-davinci-003"

        
  )'''

def load_llm():
    llm = ChatOpenAI(temperature=0.5, model_name='gpt-3.5-turbo-16k')
    return llm

'''
def load_llm():
  llm = OpenAI(openai_api_key=OpenAI_API_KEY,   
     messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo"
  )
  return llm
'''
#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cuda'})
    db = Chroma(persist_directory=CHROMA_DB, embedding_function = embeddings)
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
    msg = cl.Message(content="Starting the Open AI bot...")
    await msg.send()
    msg.content = "Hi John, Welcome to your AI Assistant. Ask me your queries."
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
    src_content = ''
    src_content  += "Source reference(s) : "+str(len(sources))

    for i in sources:
      src_content  += "\n\nPage No : "+ str(i.metadata['page'])+ "\nLink : " + str(i.metadata['source'])  
    if sources:                   
      answer = f"\nAnswer : " + str(answer)
      await cl.Message(content=answer).send()	
      if "I'm sorry" not in answer:
        await cl.Message(content=src_content).send()
      
      #if src_content != None:
        #await cl.Message(content=src_content).send()
      

    
    else:
        answer = "\nNo sources found"
        await cl.Message(content=answer).send()
 


 
 

