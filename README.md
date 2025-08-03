# 🧠 Enhanced CRAG Assistant

An intelligent conversational assistant built with **Corrective Retrieval-Augmented Generation (CRAG)**. Upload a PDF, ask questions, and get context-aware answers powered by OpenAI, FAISS, and web search fallback via DuckDuckGo.

## 🚀 Features
- ✅ Memory-aware conversation using LangChain
- 📄 PDF ingestion with semantic search (FAISS)
- 🔍 Query expansion & document relevance scoring
- 🌐 Web search fallback for missing context
- 📊 Real-time analytics and usage insights
- 🎨 Interactive UI with Streamlit and Plotly

## 🛠 Setup

1. **Install dependencies**

```bash
pip install streamlit langchain langchain-openai langchain-community faiss-cpu python-dotenv plotly pandas
```

2. **Setup Env File**

```bash
OPENAI_API_KEY=your_openai_key_here
```

3. ** Run the script
```bash
streamlit run app.py
```
