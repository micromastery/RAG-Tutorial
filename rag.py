from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain_core.prompts.chat import (
    ChatPromptTemplate
)
from langchain_core.messages import HumanMessage, SystemMessage


import os
os.environ['OPENAI_API_KEY'] = '<OPENAI_API_KEY>'

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = CSVLoader('Customer_Support_Tickets_Closed.csv', encoding="utf8").load()
# print(raw_documents[0])
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Save the vector store to disk
db.save_local('faiss_db')

# Load the vector store from disk
db = FAISS.load_local('faiss_db', OpenAIEmbeddings(), allow_dangerous_deserialization = True)

product_purchased = input("Enter the product purchased: ")
query = input("Enter your query: ")
k = 3
similar_results = db.similarity_search(f'Product Purchased: {product_purchased}\nTicket Description: {query}', k=k)

contents = [doc.page_content for doc in similar_results]

print(contents)

model = ChatOpenAI(model_name="gpt-3.5-turbo")

template_messages = [
    SystemMessage( 
                content =  f"""You are an assistant for helping to find solutions to customer support ticket by doing the following 
1/ Guess the Ticket Type
2/ Guess the Ticket Subject

You need to guess these values based on the similar tickets that were resolved in the past. You will be provided with the last {k} similar tickets that were resolved.""" 
                ),
    HumanMessage(
                content = f"""Below is the information that the customer provided in the ticket.
Product Purchased: {product_purchased}
Ticket Description: {query}

Here is the last {k} similar tickets that were resolved:
{contents}

Please write the probable ticket type, ticket subject"""
                )]

prompt = ChatPromptTemplate.from_messages(template_messages)

llm_chain = LLMChain(prompt=prompt, llm=model)

print(llm_chain.run({}))


