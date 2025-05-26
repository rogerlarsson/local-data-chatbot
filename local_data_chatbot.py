import os
import json
import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API key from environment, with fallback option
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Will be None if not set in .env or environment

# Download required NLTK data for unstructured library
# The unstructured library uses NLTK for text tokenization when processing markdown files
# This ensures the punkt_tab tokenizer is available to avoid runtime errors
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def load_config(config_path="config.json"):
    """Load configuration from JSON file with fallback defaults"""
    default_config = {
        "app_name": "Local Data ChatBot",
        "data_directory": "./data",
        "vector_db_directory": "./chroma_db",
        "prompt": "./prompt.txt",
        "text_splitter": {
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        "ui": {
            "gradio_server_name": "0.0.0.0",
            "gradio_server_port": 7860,
            "gradio_share_public_link": False
        },
        "embeddings": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"},
            "encode_kwargs": {"normalize_embeddings": True}
        },
        "anthropic": {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    }
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using default configuration")
        return default_config
    except json.JSONDecodeError:
        print(f"Error parsing {config_path}, using default configuration")
        return default_config

def load_prompt(prompt_path):
    """Load prompt from file with fallback default"""
    default_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, 
say that you don't have enough information to answer the question.

Context: {context}

Question: {question}

Answer: """
    
    try:
        with open(prompt_path, 'r') as f:
            prompt = f.read()
        print(f"Prompt loaded from {prompt_path}")
        return prompt
    except FileNotFoundError:
        print(f"Prompt file {prompt_path} not found, using default prompt")
        return default_prompt
    except Exception as e:
        print(f"Error loading prompt from {prompt_path}: {e}, using default prompt")
        return default_prompt

class LocalDataChatBot:
    def __init__(self, config, anthropic_api_key=None):
        """
        Initialize the Local Data ChatBot
        
        Args:
            config: Configuration dictionary loaded from JSON
            anthropic_api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        self.config = config
        self.data_path = config["data_directory"]
        self.persist_directory = config["vector_db_directory"]
        
        # Set up Anthropic API key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        
        # Initialize components
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history = []
        
        # Build the knowledge base
        self._load_and_process_documents()
        self._setup_conversation_chain()
    
    def _load_and_process_documents(self):
        """Load markdown and PDF files and create vector embeddings"""
        print("Loading documents...")
        
        # Load markdown files
        md_documents = self._load_markdown_documents()
        
        # Load PDF files
        pdf_documents = self._load_pdf_documents()
        
        # Combine all documents
        all_documents = md_documents + pdf_documents
        
        if not all_documents:
            raise ValueError(f"No documents found in {self.data_path}")
        
        print(f"Loaded {len(md_documents)} markdown documents and {len(pdf_documents)} PDF documents")
        
        # Split documents into chunks
        # Text Splitter Configuration:
        # chunk_size: Controls how large each text chunk is (default: 1000 characters)
        # chunk_overlap: Controls overlap between chunks to maintain context (default: 200 characters)
        # Benefits:
        # Fine-tuning for different content:
        # Smaller chunks for precise answers
        # Larger chunks for more context
        # Adjust overlap based on document structure
        # Easy experimentation: Users can test different chunking strategies without modifying code
        # Document-type optimization:
        # Technical docs might need larger chunks
        # FAQ-style content might work better with smaller chunks
        text_splitter_config = self.config["text_splitter"]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_splitter_config["chunk_size"],
            chunk_overlap=text_splitter_config["chunk_overlap"],
            length_function=len
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"Created {len(chunks)} text chunks")
        
        # Create HuggingFace embeddings
        print("Initializing HuggingFace embeddings...")
        embeddings_config = self.config["embeddings"]
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_config["model_name"],
            model_kwargs=embeddings_config["model_kwargs"],
            encode_kwargs=embeddings_config["encode_kwargs"]
        )
        
        # Create or load Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist the database
        self.vectorstore.persist()
        print("ChromaDB vector store created and persisted successfully")

    def _load_markdown_documents(self):
        """Load all markdown files from the directory (recursively including all subdirectories to any depth)"""
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        return loader.load()

    def _load_pdf_documents(self):
        """Load all PDF files from the directory (recursively including all subdirectories to any depth)"""
        try:
            from langchain.document_loaders import DirectoryLoader, PyPDFLoader
            
            loader = DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            return loader.load()
        except ImportError:
            print("Warning: PyPDF2 not installed. PDF files will be skipped.")
            print("To enable PDF support, install PyPDF2: pip install PyPDF2")
            return []
    
    def _setup_conversation_chain(self):
        """Set up the conversational retrieval chain"""
        # Initialize the Anthropic language model
        # If you want to use local models instead of Anthropic, you can replace ChatAnthropic with:
        # from langchain.llms import Ollama
        # or
        # from langchain.chat_models import ChatOllama
        # llm = ChatOllama(model="llama2")  # Requires Ollama installed locally
        anthropic_config = self.config["anthropic"]
        llm = ChatAnthropic(
            model=anthropic_config["model"],
            temperature=anthropic_config["temperature"],
            max_tokens=anthropic_config["max_tokens"]
        )
        
        # Create memory for conversation history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Custom prompt
        prompt_text = load_prompt(self.config["prompt"])
        prompt = PromptTemplate(
            template=prompt_text,
            input_variables=["context", "question"]
        )
        
        # Create the conversational retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
    
    def chat(self, message, history):
        """Process chat message and return response"""
        try:
            # Get response from the chain
            result = self.conversation_chain({"question": message})
            answer = result["answer"]
            
            # Format source information
            sources = []
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "Unknown")
                sources.append(os.path.basename(source))
            
            if sources:
                unique_sources = list(set(sources))
                answer += f"\n\n*Sources: {', '.join(unique_sources)}*"
            
            return answer
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_chain.memory.clear()
        return []

def create_gradio_interface(chatbot, config):
    """Create and return Gradio interface"""
    
    app_name = config["app_name"]
    
    with gr.Blocks(title=app_name, theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# 📚 {app_name}")
        gr.Markdown("Ask questions about your markdown documents!")
        
        chatbot_component = gr.Chatbot(
            height="70vh",  # Use 70% of viewport height
            show_label=False,
            container=True
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question about your documents...",
                container=False,
                scale=7
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear", scale=1)
        
        # Handle message submission
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            # Get bot response
            bot_response = chatbot.chat(message, history)
            
            # Update history
            history.append((message, bot_response))
            return history, ""
        
        def clear_chat():
            chatbot.clear_history()
            return []
        
        # Event handlers
        submit_btn.click(
            respond,
            inputs=[msg, chatbot_component],
            outputs=[chatbot_component, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot_component],
            outputs=[chatbot_component, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot_component]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    try:
        # Initialize the chatbot
        print("Initializing Local Data ChatBot...")
        bot = LocalDataChatBot(
            config=config,
            anthropic_api_key=ANTHROPIC_API_KEY
        )
        
        # Create and launch Gradio interface
        interface = create_gradio_interface(bot, config)
        print("Launching Gradio interface...")
        
        gradio_config = config["ui"]
        interface.launch(
            share=gradio_config["gradio_share_public_link"],  # Set to True to create a public link for sharing/demo
            # When share=True, Gradio creates a temporary public URL (e.g., https://xyz123.gradio.live)
            # This allows others to access your locally running app over the internet
            # Perfect for demos when running the app on your laptop but want others to test it
            # Note: Public links expire after 72 hours and are rate-limited
            server_name=gradio_config["gradio_server_name"],
            server_port=gradio_config["gradio_server_port"]
        )
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set your ANTHROPIC_API_KEY environment variable")
        print("2. Created a './data' directory with markdown files (or updated config.json)")
        print("3. Installed required packages: pip3 install -r requirements.txt")