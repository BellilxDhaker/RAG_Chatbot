import os
import tempfile
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional, Any
import streamlit as st

load_dotenv()


class FastTfidfEmbeddings(Embeddings):
    """Lightning-fast TF-IDF embeddings with proper fitting and normalization"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=384,
            ngram_range=(1, 2),  # Use unigrams and bigrams for better matching
            min_df=1,  # Include terms that appear at least once
            max_df=0.95,  # Exclude very common terms
            norm='l2',  # L2 normalization for better similarity scores
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Use sublinear term frequency scaling
        )
        self.fitted = False
        self._all_texts = []
    
    def fit(self, texts: List[str]):
        """Fit the vectorizer on all texts at once"""
        if texts:
            self._all_texts = texts
            self.vectorizer.fit(texts)
            self.fitted = True
            print(f"✓ Embeddings fitted on {len(texts)} text chunks")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not self.fitted:
            # If not fitted, fit on these texts
            self.fit(texts)
        
        embeddings = self.vectorizer.transform(texts).toarray()
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if not self.fitted:
            # This shouldn't happen, but handle it gracefully
            self.vectorizer.fit([text])
            self.fitted = True
        
        embedding = self.vectorizer.transform([text]).toarray()[0]
        return embedding.tolist()
    
    def __call__(self, text: str) -> List[float]:
        """Make the object callable for FAISS compatibility"""
        return self.embed_query(text)


class SimpleLLM(LLM):
    """Direct Gemini API wrapper with dynamic model selection and improved error handling"""
    
    model_name: str = ""
    api_key: str = ""
    temperature: float = 0.1  # Lower temperature for more factual responses
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "GOOGLE_API_KEY" in st.secrets:
            self.api_key = st.secrets["GOOGLE_API_KEY"]
        else:
            self.api_key = os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found! Please set it in .env or Streamlit secrets.")
        # Dynamically select available model
        self.model_name = self._get_available_model()
        print(f"✓ Using model: {self.model_name}")
    
    def _get_available_model(self) -> str:
        """Fetch and return first available model that supports generateContent"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            
            print(f"📋 Found {len(models)} available models")
            
            # Prefer specific models in order
            preferred_models = [
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-pro"
            ]
            
            # First, try to find preferred models
            for preferred in preferred_models:
                for model in models:
                    model_id = model.get("name", "").split("/")[-1]
                    methods = model.get("supportedGenerationMethods", [])
                    
                    if preferred in model_id and "generateContent" in methods:
                        print(f"✓ Selected preferred model: {model_id}")
                        return model_id
            
            # If no preferred model found, use first available
            for model in models:
                model_id = model.get("name", "").split("/")[-1]
                methods = model.get("supportedGenerationMethods", [])
                
                if "generateContent" in methods:
                    print(f"✓ Available model: {model_id}")
                    return model_id
            
            raise ValueError("No models available for generateContent")
        
        except Exception as e:
            print(f"⚠️ Error fetching models: {e}")
            # Fallback to gemini-pro
            return "gemini-pro"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Gemini using v1 REST API"""
        try:
            # Correct format: models/{model-name}:generateContent
            url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent?key={self.api_key}"
            
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048,
                }
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    if len(candidate["content"]["parts"]) > 0:
                        return candidate["content"]["parts"][0].get("text", "No response")
            
            return "Could not generate response"
            
        except requests.exceptions.Timeout:
            return "⚠️ Request timeout - please try again"
        except requests.exceptions.HTTPError as e:
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
            except:
                error_msg = e.response.text
            return f"⚠️ {error_msg}"
        except Exception as e:
            return f"⚠️ {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "google_generative_ai_v1"


def load_models():
    """Load LLM (Gemini) and embeddings"""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    
    print("✓ Loading Gemini model...")
    llm = SimpleLLM()
    
    print("✓ Initializing embeddings model...")
    embeddings = FastTfidfEmbeddings()
    
    return llm, embeddings


def process_pdf(uploaded_file, embeddings):
    """Process PDF with enhanced chunking and metadata tracking"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        pdf_path = tmp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"📄 Loaded {len(documents)} pages from PDF")

        # Enhanced text splitter for better chunk quality
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks to isolate individual person data
            chunk_overlap=50,  # Minimal overlap to avoid duplication
            separators=["Person ", "\n\n", "\n", ". ", " ", ""],  # Split on "Person X:" first
            length_function=len,
            add_start_index=True  # Track position in original document
        )
        chunks = splitter.split_documents(documents)
        
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Enhance metadata for better tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "source": uploaded_file.name,
                "char_count": len(chunk.page_content),
                "word_count": len(chunk.page_content.split())
            })
        
        # CRITICAL: Fit embeddings on ALL chunks first
        all_texts = [chunk.page_content for chunk in chunks]
        embeddings.fit(all_texts)
        
        # Now create vectorstore with properly fitted embeddings
        print(f"🔍 Creating vector store...")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        print(f"✅ PDF processed successfully! ({len(chunks)} chunks indexed)")
        return vectorstore
        
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def init_memory():
    """Initialize conversation memory"""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )


def get_relevant_chunks_with_scores(vectorstore, query, k=4):
    """
    Retrieve chunks with actual similarity scores from FAISS
    Returns list of (document, score) tuples
    
    For TF-IDF vectors, FAISS returns cosine distance (0-2 range)
    where 0 = identical, 2 = completely different
    Convert to similarity percentage: similarity = (2 - distance) / 2
    """
    try:
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        # Convert cosine distance to similarity percentage
        # Cosine distance range: 0 (identical) to 2 (opposite)
        # Convert to 0-1 similarity: (2 - distance) / 2
        converted_scores = []
        for doc, distance in docs_and_scores:
            # Cosine similarity = 1 - cosine_distance/2
            similarity = max(0.0, min(1.0, (2.0 - distance) / 2.0))
            converted_scores.append((doc, similarity))
        
        return converted_scores
    except Exception as e:
        print(f"⚠️ Error retrieving chunks with scores: {e}")
        print(f"Error details: {type(e).__name__}")
        # Fallback to regular search without scores
        docs = vectorstore.similarity_search(query, k=k)
        # Return with estimated scores based on order
        return [(doc, max(0.3, 1.0 - (i * 0.15))) for i, doc in enumerate(docs)]