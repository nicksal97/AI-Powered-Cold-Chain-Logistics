# AI-Powered Cold Chain Logistics Platform

An intelligent **AI-powered Cold Chain Logistics Assistant** designed to
support logistics operations by combining **Agentic AI,
Retrieval-Augmented Generation (RAG), knowledge retrieval, and
data-driven incident analysis**.

The system helps users analyze cold-chain incidents, retrieve
operational knowledge from SOP documents, and provide AI-assisted
recommendations for logistics decision support.

## 🚀 Key Features

-   **Agentic AI Assistant**
    -   Uses an AI agent workflow to reason over logistics problems.
    -   Provides contextual answers instead of simple document search.
-   **RAG-Based Knowledge Retrieval**
    -   Retrieves information from cold-chain SOPs, policies, and
        operational documents.
    -   Uses vector-based search for relevant knowledge grounding.
-   **Cold Chain Incident Support**
    -   Helps analyze temperature excursions and logistics issues.
    -   Provides recommended actions based on stored operational
        procedures.
-   **Enterprise Data Integration**
    -   Supports structured logistics datasets.
    -   Includes ingestion pipelines for operational data sources.
-   **Interactive User Interface**
    -   Streamlit-based interface for interacting with the AI assistant.

## 🏗️ Architecture Overview

    User
     |
     | Query
     v
    Streamlit UI
     |
     v
    AI Agent Orchestrator
     |
     +----------------+
     |                |
     v                v
    RAG Pipeline    Agent Tools
     |
     v
    Vector Database / Knowledge Base
     |
     v
    Cold Chain SOPs & Logistics Data

## 🛠️ Technology Stack

### AI / LLM

-   Python
-   LangChain
-   LangGraph
-   OpenAI API
-   Sentence Transformers

### Retrieval & Data

-   Pinecone Vector Database
-   FAISS
-   PDF/document processing
-   Pandas

### Backend / Tools

-   SQL Server integration
-   SQLAlchemy
-   PyODBC

### Frontend

-   Streamlit

### Deployment

-   Docker
-   GitHub Actions

## 📂 Project Structure

    cold-chain-logistics-FDE-Project/
    │
    ├── src/
    │   ├── orchestrator.py        # Agent workflow
    │   ├── agent_tools.py         # AI agent tools
    │   └── ui.py                  # Streamlit interface
    │
    ├── data/
    │   ├── policy/                # SOP documents
    │   ├── raw/                   # Logistics datasets
    │   └── source/                # Source data
    │
    ├── scripts/
    │   ├── Data ingestion scripts
    │
    ├── docs/
    │   └── Documentation
    │
    ├── requirements.txt
    └── README.md

## ⚙️ Installation

Clone the repository:

``` bash
git clone https://github.com/nicksal97/AI-Powered-Cold-Chain-Logistics.git
cd AI-Powered-Cold-Chain-Logistics
```

Create virtual environment:

``` bash
python -m venv venv
```

Activate environment:

Windows:

``` bash
venv\\Scripts\\activate
```

Linux/Mac:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## 🔑 Environment Variables

Create a `.env` file:

    OPENAI_API_KEY=your_key
    PINECONE_API_KEY=your_key

Never commit secrets to GitHub.

## ▶️ Run Application

Start the Streamlit application:

``` bash
streamlit run src/ui.py
```

## 📚 Use Cases

-   Cold-chain incident investigation
-   Logistics knowledge assistant
-   SOP compliance support
-   Supply-chain operational intelligence
-   AI-assisted decision support

## 📄 Documentation

Additional technical documents are available in:

    Misc/Materials/

including: - Technical Design Document - Business Presentation

## 👨‍💻 Author

Nikhil Salunkhe

## ⭐ Future Improvements

-   Real-time IoT temperature sensor integration
-   Automated alert generation
-   Multi-agent logistics planning
-   Advanced analytics dashboard
-   Production cloud deployment
