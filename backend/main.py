from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import ollama
import json
from pathlib import Path

import config
from rag_engine import RAGEngine
from actions import ActionHandler

app = FastAPI(title="Local LLM Agent")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_engine = RAGEngine()
action_handler = ActionHandler()

# Conversation history
conversations = {}


class QueryRequest(BaseModel):
    query: str
    collection: str = config.DEFAULT_COLLECTION
    conversation_id: Optional[str] = None
    use_rag: bool = True


class ActionRequest(BaseModel):
    action: str
    parameters: dict


@app.get("/")
async def root():
    return {"message": "Local LLM Agent API", "status": "running"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = config.DEFAULT_COLLECTION
):
    """Upload and process a document"""
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )
    
    # Save file
    file_path = config.UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Process document
    result = rag_engine.process_document(file_path, collection)
    
    return result


@app.post("/query")
async def query_stream(request: QueryRequest):
    """Query the LLM with RAG context (streaming response)"""
    
    async def generate():
        try:
            # Get conversation history
            conv_id = request.conversation_id or "default"
            if conv_id not in conversations:
                conversations[conv_id] = []
            
            # Retrieve relevant context if RAG enabled
            context = ""
            sources = []
            if request.use_rag:
                docs = rag_engine.query(request.query, request.collection)
                if docs:
                    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs])
                    sources = [{"source": doc.metadata.get('source'), "content": doc.page_content[:200]} for doc in docs]
            
            # Build prompt
            system_prompt = """You are a helpful AI assistant with access to uploaded documents and tools.
When answering questions, use the provided context from documents when relevant.
If you need to use a tool, respond with a JSON object in this format:
{"action": "tool_name", "parameters": {"param": "value"}}

Available tools:
- web_search: Search the web for current information
- calculator: Perform mathematical calculations  
- get_time: Get current date and time

If you don't need a tool, just answer the question normally."""
            
            if context:
                system_prompt += f"\n\nContext from documents:\n{context}"
            
            # Add to conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversations[conv_id][-6:])  # Last 3 exchanges
            messages.append({"role": "user", "content": request.query})
            
            # Stream response from Ollama
            response_text = ""
            stream = ollama.chat(
                model=config.MODEL_NAME,
                messages=messages,
                stream=True,
                options={
                    "temperature": config.TEMPERATURE,
                    "num_predict": config.MAX_TOKENS
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    response_text += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            
            # Check if response is an action request
            if response_text.strip().startswith("{") and "action" in response_text:
                try:
                    action_data = json.loads(response_text)
                    if "action" in action_data and "parameters" in action_data:
                        # Execute action
                        action_result = action_handler.execute(
                            action_data["action"],
                            action_data["parameters"]
                        )
                        
                        # Send action result back to LLM
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": f"Tool result: {action_result}"})
                        
                        # Get final response
                        final_response = ""
                        stream = ollama.chat(
                            model=config.MODEL_NAME,
                            messages=messages,
                            stream=True
                        )
                        
                        for chunk in stream:
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                final_response += content
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        
                        response_text = final_response
                except json.JSONDecodeError:
                    pass
            
            # Update conversation history
            conversations[conv_id].append({"role": "user", "content": request.query})
            conversations[conv_id].append({"role": "assistant", "content": response_text})
            
            # Send sources
            if sources:
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/documents/{collection}")
async def list_documents(collection: str = config.DEFAULT_COLLECTION):
    """List all documents in a collection"""
    docs = rag_engine.list_documents(collection)
    return {"documents": docs, "collection": collection}


@app.delete("/documents/{collection}/{filename}")
async def delete_document(collection: str, filename: str):
    """Delete a document from collection"""
    success = rag_engine.delete_document(filename, collection)
    if success:
        return {"message": f"Deleted {filename}"}
    raise HTTPException(status_code=404, detail="Document not found")


@app.get("/actions")
async def list_actions():
    """List available actions/tools"""
    return {"actions": action_handler.get_available_actions()}


@app.post("/action")
async def execute_action(request: ActionRequest):
    """Execute an action directly"""
    result = action_handler.execute(request.action, request.parameters)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)