from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
# new entries
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

import streamlit as st

vector_store_container = {"instance": None}

load_dotenv()


# --------------------- LLM ----------------------


llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')


# --------------------- Tools --------------------


search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()



@tool
def search_uploaded_pdf(query: str) -> str:
    """Use this tool to search the uploaded PDF document for relevant information."""
    
    # 1. Check our shared backend container instead of st.session_state
    if vector_store_container["instance"] is None:
        return "No PDF document has been uploaded yet. Ask the user to upload a PDF file first."

    # 2. Grab the FAISS vector store instance
    vector_store = vector_store_container["instance"]
    
    # 3. Query the retriever for the top 3 chunks
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query)

    # 4. Handle empty retrieval results
    if not retrieved_docs:
        return "No relevant information found in the uploaded PDF document."

    # 5. Extract and join the text chunks into a single clean string
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 6. Return ONLY the raw context text string back to the agent
    return context_text


# --------------------- Tools binding --------------------
tools = [search_tool, get_stock_price, calculator, search_uploaded_pdf]
llm_with_tools = llm.bind_tools(tools)


# --------------------- State --------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]



# --------------------- Nodes --------------------


def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm_with_tools.invoke(messages)

    # response store state
    return {'messages': [response]}


tool_node = ToolNode(tools)


# --------------------- Checkpointer -------------

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=connection)



# --------------------- Graph --------------------

graph = StateGraph(ChatState)
# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, 'chat_node')

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# --------------------- Helper Func --------------

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return (list(all_threads))

# for message_chunk, metadata in  chatbot.stream( # this returns a generator 
#    {'messages': [HumanMessage(content='What is the recipe to make pasta')]},
#    config = {'configurable': {'thread_id': 'thread_1'}},
#    stream_mode= 'messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)

# while True:

#     user_message = input("type here: ")
#     print('User Message: ', user_message)
#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break
    
#     config = {'configurable': {'thread_id': thread_id}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]}, config=config)

#     print('AI: ', response['messages'][-1].content)
