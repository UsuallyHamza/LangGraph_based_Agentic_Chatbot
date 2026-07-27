import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads, vector_store_container
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------Utility functions-------------------------------------- 

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']

# Configure the page
st.set_page_config(
    page_title="Gemini AI Assistant", 
    page_icon="🤖", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)

# ----------------------Session Setup-------------------------------------- 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# ----------------------Sidebar UI-------------------------------------- 
st.sidebar.title('LangGraph chatbot')

if st.sidebar.button("New Chat"):
    reset_chat()

# --- DOCUMENT CONTEXT SECTION ---
st.sidebar.header("Document Context")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to chat with it", type="pdf")

# Handle PDF Upload, Processing, and Cleanup
# Handle PDF Upload, Processing, and Syncing
if uploaded_file is not None:
    # 1. New file uploaded -> Process it
    if 'current_pdf_name' not in st.session_state or st.session_state['current_pdf_name'] != uploaded_file.name:
        with st.sidebar.status("Processing Document...", expanded=True) as status:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                st.write("📄 Loading PDF...")
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                st.write("✂️ Splitting text...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_documents(documents)
                
                st.write("🧠 Creating embeddings...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Update BOTH session state and the backend container reference
                st.session_state['vector_store'] = vector_store
                st.session_state['current_pdf_name'] = uploaded_file.name 
                vector_store_container["instance"] = vector_store
                
                os.remove(tmp_file_path)
                status.update(label="✅ PDF loaded into memory!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ Error processing PDF", state="error")
                st.sidebar.error(str(e))
    # 2. File already processed and active
    elif 'vector_store' in st.session_state:
        # Keep container synced on script reruns
        vector_store_container["instance"] = st.session_state['vector_store']
        st.sidebar.success(f"✅ Active PDF: `{uploaded_file.name}`")
else:
    # 3. Clear state if the file is removed
    if 'vector_store' in st.session_state:
        st.session_state.pop('vector_store', None)
        st.session_state.pop('current_pdf_name', None)
        vector_store_container["instance"] = None

# --- PAST CONVERSATIONS SECTION ---
st.sidebar.header("Past Conversations")

for thread_id in st.session_state['chat_threads'][::]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
                content_text = str(msg.content)
            else:
                role = 'assistant'
                if isinstance(msg.content, list) and len(msg.content) > 0:
                    content_text = msg.content[0].get('text', '')
                else:
                    content_text = str(msg.content)
            
            temp_messages.append({'role': role, 'content': content_text})
        
        st.session_state['message_history'] = temp_messages


# --- GEMINI STYLE LANDING PAGE ---
if len(st.session_state['message_history']) == 0:
    for _ in range(4):
        st.write("")
    
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 3rem; font-weight: 500; color: #E3E6E8; margin-bottom: 10px;">
                Meet your personal AI assistant
            </h1>
            <p style="font-size: 1.2rem; color: #888888;">
                Powered by LangGraph & Gemini
            </p>
            <p style="font-size: 1.2rem; color: #888888;">
                Made by M Hamza
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )


# ----------------------Main UI-------------------------------------- 

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content']) 

# --- CHAT INPUT ---
user_input = st.chat_input('Ask Anything...')

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        status_holder = {"box": None}

        def stream_content():
            for message_chunk, metadata in chatbot.stream( 
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Executing `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Executing `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and status_holder["box"] is not None:
                    status_holder["box"].update(
                        label="✅ Tool execution complete!",
                        state="complete",
                        expanded=False,
                    )

                content = message_chunk.content
                
                if isinstance(content, list) and len(content) > 0:
                    yield content[0].get('text', '')
                elif isinstance(content, str) and isinstance(message_chunk, AIMessage):
                    yield content

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Finished processing",
                    state="complete",
                    expanded=False,
                )

        full_response = st.write_stream(stream_content())

    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})