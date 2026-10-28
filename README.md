# Heathrow AI Customer Support Agent

A context-aware AI support agent built for a fictional furniture company to automate and scale customer inquiries. 

This project utilizes Retrieval-Augmented Generation (RAG) backed by a Chroma vector database to ground AI responses in proprietary company data. It integrates the Groq API for ultra-fast, low-latency LLM inference, leveraging historical support logs to ensure accurate, contextually relevant, and safe multi-turn conversational interactions.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM & AI:** Groq API 
* **Vector Database:** Chroma DB
* **Infrastructure:** Docker, Docker Compose

## 📁 Repository Structure
* `src/`: Contains the core application logic, RAG pipelines, and API integrations.
* `capabilities.json`: Configuration mapping for the agent's domain-specific scenarios.
* `Dockerfile` & `docker-compose.yml`: Containerization configurations for standardized deployment.
* `requirements.txt`: Python package dependencies.

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/balloneya/Heathrow-AI-Agent.git](https://github.com/balloneya/Heathrow-AI-Agent.git)
   cd Heathrow-AI-Agent
