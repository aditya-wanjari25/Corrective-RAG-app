import streamlit as st
import os
import json
import time
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Core imports (make sure these are installed)
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import DuckDuckGoSearchResults
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set OpenAI API key (make sure you have this in your .env file)
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# Import your helper functions (adjust path as needed)
try:
    import sys
    sys.path.append('RAG_TECHNIQUES')
    from helper_functions import encode_pdf
except ImportError:
    st.error("⚠️ Helper functions not found. Please ensure RAG_TECHNIQUES is accessible.")

# Page configuration
st.set_page_config(
    page_title="🧠 Enhanced CRAG Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .user-message {
        background-color: #667eea;
        border-left-color: #667eea;
    }
    
    .assistant-message {
        background-color: #1f77b4;
        border-left-color: #1f77b4;
    }
    
    .system-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    
    .status-ready { background-color: #00d084; }
    .status-error { background-color: #ff6b6b; }
    .status-warning { background-color: #feca57; }
</style>
""", unsafe_allow_html=True)

# Pydantic models (same as notebook)
class RetrievalEvaluatorInput(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of the document to the query. Score should be between 0 and 1.")

class KnowledgeRefinementInput(BaseModel):
    key_points: str = Field(..., description="The key information extracted from the document.")

class QueryRewriterInput(BaseModel):
    query: str = Field(..., description="The rewritten query optimized for web search.")

class QueryExpansionInput(BaseModel):
    expanded_query: str = Field(..., description="The query expanded with conversational context.")

# Enhanced CRAG System (streamlined for Streamlit)
class StreamlitCRAGSystem:
    def __init__(self, llm, faiss_index, search_tool, max_token_limit: int = 2000):
        self.llm = llm
        self.faiss_index = faiss_index
        self.search = search_tool
        
        # Initialize memory
        self.memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        
        # Conversation context for analytics
        self.conversation_context = []
        self.query_analytics = []
    
    def add_to_memory(self, human_input: str, ai_response: str):
        """Add conversation to memory and analytics"""
        self.memory.chat_memory.add_user_message(human_input)
        self.memory.chat_memory.add_ai_message(ai_response)
        
        # Add to context
        self.conversation_context.append({
            'human': human_input,
            'ai': ai_response,
            'timestamp': datetime.now()
        })
        
        # Analytics
        self.query_analytics.append({
            'query': human_input,
            'response_length': len(ai_response),
            'timestamp': datetime.now()
        })
        
        # Keep manageable size
        if len(self.conversation_context) > 10:
            self.conversation_context.pop(0)
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context"""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for turn in self.conversation_context[-3:]:
            context_parts.append(f"Human: {turn['human']}")
            ai_response = turn['ai'][:300] + "..." if len(turn['ai']) > 300 else turn['ai']
            context_parts.append(f"Assistant: {ai_response}")
        
        return "\n".join(context_parts)
    
    def expand_query_with_context(self, query: str) -> str:
        """Expand query with context"""
        if not self.conversation_context:
            return query
        
        context = self.get_conversation_context()
        
        expansion_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""Given the conversation history and current query, expand the query to include relevant context.

Conversation History:
{context}

Current Query: {query}

Rules:
- If query uses pronouns (it, that, this, they), replace with specific terms from context
- If query is vague, add relevant context
- If query is already specific, return as-is
- Keep expansion concise and relevant

Expanded Query:"""
        )
        
        try:
            chain = expansion_prompt | self.llm.with_structured_output(QueryExpansionInput)
            result = chain.invoke({"context": context, "query": query})
            expanded_query = result.expanded_query.strip()
            
            if len(expanded_query) > len(query) * 3 or not expanded_query:
                return query
            
            return expanded_query
        except:
            return query
    
    def retrieval_evaluator(self, query: str, document: str) -> float:
        """Evaluate document relevance"""
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="On a scale from 0 to 1, how relevant is this document to the query?\n\nQuery: {query}\n\nDocument: {document}\n\nRelevance score:"
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(RetrievalEvaluatorInput)
            result = chain.invoke({"query": query, "document": document})
            return result.relevance_score
        except:
            return 0.5
    
    def knowledge_refinement(self, document: str) -> List[str]:
        """Extract key information"""
        prompt = PromptTemplate(
            input_variables=["document"],
            template="Extract the key information from the following document in bullet points:\n\n{document}\n\nKey points:"
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(KnowledgeRefinementInput)
            result = chain.invoke({"document": document})
            key_points = result.key_points
            return [point.strip() for point in key_points.split('\n') if point.strip()]
        except:
            return [document[:500] + "..."]
    
    def rewrite_query_for_search(self, query: str) -> str:
        """Rewrite query for web search"""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="Rewrite this query to be more effective for web search:\n\nOriginal: {query}\n\nWeb-optimized query:"
        )
        
        try:
            chain = prompt | self.llm.with_structured_output(QueryRewriterInput)
            result = chain.invoke({"query": query})
            return result.query.strip()
        except:
            return query
    
    def parse_search_results(self, results_string: str) -> List[Tuple[str, str]]:
        """Parse search results"""
        try:
            results = json.loads(results_string)
            return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results]
        except:
            return []
    
    def retrieve_documents(self, query: str, k: int = 3) -> List[str]:
        """Retrieve documents from FAISS"""
        docs = self.faiss_index.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def evaluate_documents(self, query: str, documents: List[str]) -> List[float]:
        """Evaluate document relevance"""
        return [self.retrieval_evaluator(query, doc) for doc in documents]
    
    def perform_web_search(self, query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Perform web search"""
        rewritten_query = self.rewrite_query_for_search(query)
        
        try:
            web_results = self.search.run(rewritten_query)
            web_knowledge = self.knowledge_refinement(web_results)
            sources = self.parse_search_results(web_results)
            return web_knowledge, sources
        except Exception as e:
            st.warning(f"Web search failed: {e}")
            return [f"Unable to retrieve web information for: {query}"], []
    
    def generate_response_with_context(self, query: str, knowledge: str, sources: List[Tuple[str, str]]) -> str:
        """Generate response with context"""
        try:
            memory_content = self.memory.buffer
        except:
            memory_content = self.get_conversation_context()
        
        response_prompt = PromptTemplate(
            input_variables=["memory", "query", "knowledge", "sources"],
            template="""Based on the conversation history, knowledge, and sources, answer the current query.

Conversation History:
{memory}

Current Query: {query}

Knowledge: {knowledge}

Sources: {sources}

Instructions:
- Consider conversation context when answering
- If the query refers to something mentioned earlier, use that context
- Provide a complete, helpful answer
- Include source references at the end if available
- Be conversational and natural

Answer:"""
        )
        
        try:
            sources_text = "\n".join([f"{title}: {link}" if link else title for title, link in sources])
            
            response = self.llm.invoke(
                response_prompt.format(
                    memory=memory_content,
                    query=query,
                    knowledge=knowledge,
                    sources=sources_text
                )
            )
            return response.content
        except Exception as e:
            return f"I apologize, but I encountered an error: {e}"
    
    def process_query_with_progress(self, query: str, progress_bar, status_text):
        """Process query with Streamlit progress updates"""
        
        # Step 1: Query expansion
        status_text.text("🔍 Analyzing query and expanding with context...")
        progress_bar.progress(10)
        expanded_query = self.expand_query_with_context(query)
        
        # Step 2: Document retrieval
        status_text.text("📚 Retrieving relevant documents...")
        progress_bar.progress(30)
        retrieved_docs = self.retrieve_documents(expanded_query)
        
        # Step 3: Evaluation
        status_text.text("🧮 Evaluating document relevance...")
        progress_bar.progress(50)
        eval_scores = self.evaluate_documents(query, retrieved_docs)
        
        # Step 4: Decision making
        status_text.text("🤖 Determining best information source...")
        progress_bar.progress(70)
        
        max_score = max(eval_scores) if eval_scores else 0
        sources = []
        
        if max_score > 0.7:
            action = "CORRECT"
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            final_knowledge = best_doc
            sources.append(("Retrieved Document", "Local Knowledge Base"))
            
        elif max_score < 0.3:
            action = "INCORRECT"
            status_text.text("🌐 Searching the web for better information...")
            final_knowledge, sources = self.perform_web_search(expanded_query)
            final_knowledge = "\n".join(final_knowledge) if isinstance(final_knowledge, list) else final_knowledge
            
        else:
            action = "AMBIGUOUS"
            status_text.text("🔄 Combining local and web sources...")
            best_doc = retrieved_docs[eval_scores.index(max_score)]
            retrieved_knowledge = self.knowledge_refinement(best_doc)
            web_knowledge, web_sources = self.perform_web_search(expanded_query)
            
            final_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
            sources = [("Retrieved Document", "Local Knowledge Base")] + web_sources
        
        # Step 5: Response generation
        status_text.text("✍️ Generating response...")
        progress_bar.progress(90)
        response = self.generate_response_with_context(query, final_knowledge, sources)
        
        # Final step
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        # Add to memory
        self.add_to_memory(query, response)
        
        return {
            'response': response,
            'action': action,
            'max_score': max_score,
            'eval_scores': eval_scores,
            'sources': sources,
            'expanded_query': expanded_query
        }

# Session state initialization
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'crag_system' not in st.session_state:
        st.session_state.crag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

# System initialization
@st.cache_resource
def initialize_system(openai_key: str, pdf_path: str):
    """Initialize the CRAG system (cached for performance)"""
    try:
        # Set environment
        os.environ["OPENAI_API_KEY"] = openai_key
        
        # Initialize components
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)
        search = DuckDuckGoSearchResults()
        
        # Create vector store
        if pdf_path and os.path.exists(pdf_path):
            vectorstore = encode_pdf(pdf_path)
        else:
            raise FileNotFoundError("PDF file not found")
        
        # Create CRAG system
        crag_system = StreamlitCRAGSystem(
            llm=llm,
            faiss_index=vectorstore,
            search_tool=search,
            max_token_limit=2000
        )
        
        return crag_system, "System initialized successfully! 🎉"
    
    except Exception as e:
        return None, f"Initialization failed: {str(e)}"

# Main app
def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Enhanced CRAG Assistant</h1>
        <p>Corrective Retrieval-Augmented Generation with Memory</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        openai_key = os.getenv('OPENAI_API_KEY')
        # openai_key = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     help="Enter your OpenAI API key"
        # )
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF to create the knowledge base"
        )
        
        # PDF path input (alternative)
        pdf_path = st.text_input(
            "Or enter PDF path",
            help="Full path to your PDF file"
        )
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            if not openai_key:
                st.error("Please provide OpenAI API key")
            elif not (uploaded_file or pdf_path):
                st.error("Please upload a PDF or provide path")
            else:
                with st.spinner("Initializing system..."):
                    # Handle uploaded file
                    if uploaded_file:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        pdf_path = temp_path
                    
                    # Initialize system
                    system, message = initialize_system(openai_key, pdf_path)
                    
                    if system:
                        st.session_state.crag_system = system
                        st.session_state.system_initialized = True
                        st.success(message)
                    else:
                        st.error(message)
        
        # System status
        st.markdown("---")
        st.header("📊 System Status")
        
        if st.session_state.system_initialized:
            st.markdown('<span class="status-indicator status-ready"></span>**Ready**', unsafe_allow_html=True)
            
            if st.session_state.crag_system:
                system = st.session_state.crag_system
                st.metric("Conversations", len(system.conversation_context))
                st.metric("Memory Turns", len(system.query_analytics))
        else:
            st.markdown('<span class="status-indicator status-error"></span>**Not Initialized**', unsafe_allow_html=True)
        
        # Clear memory button
        if st.session_state.system_initialized:
            st.markdown("---")
            if st.button("🧹 Clear Memory"):
                if st.session_state.crag_system:
                    st.session_state.crag_system.memory.clear()
                    st.session_state.crag_system.conversation_context = []
                    st.session_state.crag_system.query_analytics = []
                st.session_state.chat_history = []
                st.success("Memory cleared!")
                st.rerun()
    
    # Main content area
    if not st.session_state.system_initialized:
        st.info("👈 Please configure and initialize the system using the sidebar")
        
        # Demo section
        st.markdown("## 🌟 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🧠 Memory Management**
            - Conversational context
            - Smart query expansion
            - History summarization
            """)
        
        with col2:
            st.markdown("""
            **🔍 Intelligent Retrieval**
            - Document relevance scoring
            - Web search fallback
            - Source combination
            """)
        
        with col3:
            st.markdown("""
            **📊 Analytics**
            - Performance metrics
            - Conversation insights
            - Real-time monitoring
            """)
        
        return
    
    # Chat interface
    system = st.session_state.crag_system
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analytics", "🔧 Debug"])
    
    with tab1:
        st.header("Chat with your documents")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🧑 You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show metadata if available
                    if 'metadata' in message:
                        with st.expander("🔍 Query Details", expanded=False):
                            meta = message['metadata']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Action:** {meta['action']}")
                                st.write(f"**Max Score:** {meta['max_score']:.2f}")
                            
                            with col2:
                                if meta['expanded_query'] != message['original_query']:
                                    st.write(f"**Expanded Query:** {meta['expanded_query']}")
                                
                                if meta['sources']:
                                    st.write("**Sources:**")
                                    for title, link in meta['sources']:
                                        if link:
                                            st.write(f"- [{title}]({link})")
                                        else:
                                            st.write(f"- {title}")
        
        # Chat input
        st.markdown("---")
        
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask a question:",
                    placeholder="What would you like to know?",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send 🚀", type="primary")
        
        if submit_button and user_input:
            # Add user message to chat
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Process query with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                result = system.process_query_with_progress(
                    user_input, progress_bar, status_text
                )
                
                # Add assistant response
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': result['response'],
                    'timestamp': datetime.now(),
                    'original_query': user_input,
                    'metadata': result
                })
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {e}")
                progress_bar.empty()
                status_text.empty()
    
    with tab2:
        st.header("📊 System Analytics")
        
        if system.query_analytics:
            # Create analytics dataframe
            df = pd.DataFrame(system.query_analytics)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Queries",
                    len(df)
                )
            
            with col2:
                avg_response_length = df['response_length'].mean()
                st.metric(
                    "Avg Response Length",
                    f"{avg_response_length:.0f} chars"
                )
            
            with col3:
                st.metric(
                    "Active Memory Turns",
                    len(system.conversation_context)
                )
            
            with col4:
                recent_queries = len(df[df['timestamp'] > datetime.now().replace(hour=0, minute=0, second=0)])
                st.metric(
                    "Queries Today",
                    recent_queries
                )
            
            # Query frequency chart
            st.subheader("Query Activity Over Time")
            
            if len(df) > 1:
                # Resample by hour
                df_hourly = df.set_index('timestamp').resample('1H').size().reset_index()
                df_hourly.columns = ['timestamp', 'count']
                
                fig = px.line(
                    df_hourly,
                    x='timestamp',
                    y='count',
                    title='Queries per Hour',
                    labels={'count': 'Number of Queries', 'timestamp': 'Time'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Response length distribution
            st.subheader("Response Length Distribution")
            fig = px.histogram(
                df,
                x='response_length',
                title='Distribution of Response Lengths',
                labels={'response_length': 'Response Length (characters)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No analytics data available yet. Start chatting to see insights!")
    
    with tab3:
        st.header("🔧 Debug Information")
        
        # System information
        st.subheader("System State")
        
        if system:
            debug_info = {
                "Memory Buffer Length": len(system.memory.buffer) if hasattr(system.memory, 'buffer') else 0,
                "Conversation Context Turns": len(system.conversation_context),
                "Query Analytics Count": len(system.query_analytics),
                "FAISS Index Type": type(system.faiss_index).__name__,
                "LLM Model": system.llm.model_name if hasattr(system.llm, 'model_name') else "Unknown"
            }
            
            for key, value in debug_info.items():
                st.write(f"**{key}:** {value}")
        
        # Recent conversation context
        if system and system.conversation_context:
            st.subheader("Recent Conversation Context")
            
            with st.expander("View Raw Context"):
                context = system.get_conversation_context()
                st.text_area("Context", context, height=200)
        
        # Memory buffer
        if system:
            st.subheader("Memory Buffer")
            
            with st.expander("View Memory Buffer"):
                try:
                    buffer_content = system.memory.buffer
                    st.text_area("Buffer", buffer_content, height=200)
                except:
                    st.write("Memory buffer not accessible")
        
        # Session state
        st.subheader("Session State")
        
        with st.expander("View Session State"):
            session_state_dict = {
                key: str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                for key, value in st.session_state.items()
                if key != 'crag_system'  # Exclude the system object for readability
            }
            st.json(session_state_dict)

if __name__ == "__main__":
    main()