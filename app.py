from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize FastAPI app
app = FastAPI()

# Initialize LLM (using Hugging Face Endpoint)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=hf_api_key,
    max_new_tokens=150,
    temperature=0.7
)

# Initialize memory and conversation chain
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Pydantic model for request body
class ChatInput(BaseModel):
    user_input: str

# Chat endpoint
@app.post("/chat")
async def chat(input: ChatInput):
    response = conversation.predict(input=input.user_input)
    return {"response": response}

# Clear memory endpoint
@app.post("/clear-memory")
async def clear_memory():
    memory.clear()
    return {"status": "Conversation memory cleared"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000