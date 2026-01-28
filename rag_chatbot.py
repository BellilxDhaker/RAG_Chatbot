import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from utils import load_models, process_pdf, init_memory, get_relevant_chunks_with_scores
import time

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")

# Load models once
if "models_loaded" not in st.session_state:
    with st.spinner("Loading models..."):
        st.session_state.llm, st.session_state.embeddings = load_models()
    st.session_state.models_loaded = True

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = process_pdf(uploaded_file, st.session_state.embeddings)
            st.session_state.memory = init_memory()
            st.session_state.messages = []
            st.session_state.current_file = uploaded_file.name
        st.success(f"✅ PDF processed successfully!")

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" in st.session_state:
    
    # Enhanced prompt that uses conversation history
    qa_prompt_template = """You are a helpful AI assistant. Answer questions based on the context and remember the conversation history.

When the user refers to "he", "she", "his", "her", "their", "it", "the person", etc., look at the chat history to understand who they're talking about.

Be direct and concise. Give only the specific information asked for.

Context from document:
{context}

Previous conversation:
{chat_history}

Current question: {question}

Answer (be brief and specific):"""

    QA_PROMPT = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create QA chain with memory
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Get top 10 chunks to find the right person
        ),
        memory=st.session_state.memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show only the most relevant source with chunk
            if msg["role"] == "assistant" and "source" in msg and msg["source"]:
                source = msg["source"]
                with st.expander(f"📄 Source: Page {source['page']}", expanded=False):
                    st.text(source['chunk'])

    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response
                    result = qa_chain.invoke({"question": prompt})
                    response = result["answer"]
                    st.markdown(response)
                    
                    # Get only the most relevant source
                    source_data = None
                    if "source_documents" in result and result["source_documents"]:
                        # Get top chunks and find the one most relevant to the answer
                        docs_with_scores = get_relevant_chunks_with_scores(
                            st.session_state.vectorstore, 
                            prompt, 
                            k=10  # Get top 10 to search through
                        )
                        
                        # Find the best matching chunk - prefer one that contains answer keywords
                        best_doc = None
                        best_score = -1
                        answer_lower = response.lower()
                        
                        for doc, similarity_score in docs_with_scores:
                            chunk_lower = doc.page_content.lower()
                            # Check if response content appears in chunk
                            if any(keyword in chunk_lower for keyword in answer_lower.split() if len(keyword) > 3):
                                if similarity_score > best_score:
                                    best_score = similarity_score
                                    best_doc = doc
                        
                        # Fallback to top result if no keyword match
                        if best_doc is None and docs_with_scores:
                            best_doc, best_score = docs_with_scores[0]
                        
                        if best_doc:
                            page_num = best_doc.metadata.get("page", "Unknown")
                            chunk_text = best_doc.page_content.strip()
                            
                            source_data = {
                                "page": page_num,
                                "chunk": chunk_text,
                                "score": best_score
                            }
                            
                            # Display source with chunk
                            with st.expander(f"📄 Source: Page {page_num}", expanded=False):
                                st.text(chunk_text)
                
                except Exception as e:
                    response = f"⚠️ Error: {str(e)}"
                    st.error(response)
                    source_data = None

        # Save message with source
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "source": source_data,
            "timestamp": time.time()
        })

else:
    st.info("👆 Please upload a PDF to start chatting!")