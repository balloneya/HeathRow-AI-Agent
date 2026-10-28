import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 1. Setup Logging & App ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="Heathrow Support Agent", description="AI Customer Support for Heathrow Furniture")

# --- 2. Securely Load API Key (Docker/Server safe) ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logging.warning("GROQ_API_KEY environment variable is not set!")

# --- 3. Build/Load The HYBRID Knowledge Base ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_dir = "./heathrow_db"

# When the server boots, check if the database exists
if os.path.exists(persist_dir):
    logging.info("🧠 Loading pre-trained Heathrow database...")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    logging.info("⚙️ Building HYBRID brain: Training on Chat Logs + Official Policies...")
    
    # Notice the path points to the src/ folder where your CSV lives
    csv_path = "src/Heathrow_Support_Sample.csv"
    if os.path.exists(csv_path):
        loader = CSVLoader(file_path=csv_path, source_column="instruction")
        chat_logs = loader.load()
    else:
        logging.warning(f"Dataset not found at {csv_path}. Proceeding with official policies only.")
        chat_logs = []
    
    official_policies = [
        Document(page_content="Refund Policy: All refunds take exactly 5-7 business days to process and will appear on the customer's original payment method.", metadata={"source": "Official Policy"}),
        Document(page_content="Returns: Heathrow Furniture accepts returns on unassembled furniture within 30 days of delivery. A 15% restocking fee applies to large items.", metadata={"source": "Official Policy"}),
        Document(page_content="Damage Claims: Must be reported within 24 hours of delivery with photographic evidence.", metadata={"source": "Official Policy"}),
        Document(page_content="Shipping Timelines: Standard freight takes 2-3 weeks. Expedited takes 1 week.", metadata={"source": "Official Policy"})
    ]
    
    all_knowledge = chat_logs + official_policies
    vectorstore = Chroma.from_documents(all_knowledge, embeddings, persist_directory=persist_dir)
    logging.info("✅ Hybrid Training complete and saved to disk!")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4. Prompt Engineering (Strict Escalation & Brevity) ---
system_template = """
You are a polite, professional AI customer support agent for 'Heathrow Furniture'. 
You have access to a hybrid database containing official corporate policies AND past human support chats:
{context}

Chat History:
{chat_history}

Instructions:
1. Fact Priority: If the retrieved context contains an "Official Policy", you MUST use those exact facts and numbers (e.g., timelines, fees).
2. Tone & Brevity: Maintain a warm, conversational, and helpful tone, but strictly limit your response to 2 to 3 sentences. 
3. MANDATORY ESCALATION (HIGHEST PRIORITY): You must completely stop trying to help and reply EXACTLY with "ESCALATION_TRIGGERED: [Brief summary of the anger/threat]" if ANY of the following occur:
   - The user threatens legal action (e.g., "sue you", "lawyer").
   - The user uses profanity or highly abusive language.
   - The user explicitly demands a human, agent, or manager.
   - The retrieved context does not contain the answer.

User Query: {question}
Agent Response:
"""

prompt = PromptTemplate(template=system_template, input_variables=["context", "chat_history", "question"])

# --- 5. Initialize Llama 3.1 and Memory ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1) 
# Note: In a production server, memory usually needs to be tied to a session ID, 
# but we will use standard buffer memory for this standalone agent version.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, combine_docs_chain_kwargs={"prompt": prompt}
)

# --- 6. FastAPI Web Endpoints ---

class ChatRequest(BaseModel):
    user_query: str

@app.get("/")
async def health_check():
    """Simple endpoint to verify the agent is online."""
    return {"status": "online", "agent": "Heathrow Support"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """The main endpoint Nasiko will send user messages to."""
    try:
        response = agent_chain.invoke({"question": request.user_query})
        answer = response['answer']
        
        # Intercept escalations before sending the response back to Nasiko
        if "ESCALATION_TRIGGERED" in answer:
            escalation_context = answer.replace("ESCALATION_TRIGGERED:", "").strip()
            logging.warning(f"Escalated: {escalation_context}")
            return {
                "response": "I want to make sure you get the best help possible with this. I am transferring this chat to a human agent who will be right with you.",
                "escalated": True,
                "context": escalation_context
            }
        
        # Standard successful response
        return {
            "response": answer,
            "escalated": False
        }
    except Exception as e:
        logging.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail="Internal agent error")