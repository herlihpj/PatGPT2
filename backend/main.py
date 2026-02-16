from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import ollama
import json
import re
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


def extract_json_action(text: str) -> Optional[dict]:
    """Extract JSON action from text, handling various formats"""
    # Try to find JSON in the text
    patterns = [
        r'\{[^}]*"action"[^}]*\}',  # Basic JSON object
        r'```json\s*(\{[^}]*"action"[^}]*\})\s*```',  # Markdown code block
        r'`(\{[^}]*"action"[^}]*\})`',  # Inline code
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                # Clean up common issues
                json_str = json_str.strip()
                action_data = json.loads(json_str)
                if "action" in action_data:
                    return action_data
            except json.JSONDecodeError:
                continue
    
    return None


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
            
            # Build improved system prompt
            system_prompt = """You are a helpful AI assistant with access to uploaded documents and tools.

When you need to use a tool, respond ONLY with a JSON object in this EXACT format (no additional text):
{"action": "tool_name", "parameters": {"param": "value"}}

Available tools:
1. web_search - Search the internet for current information
   Format: {"action": "web_search", "parameters": {"query": "your search query"}}
   Use when: User asks about current events, recent news, prices, weather, or anything requiring up-to-date information

2. calculator - Perform mathematical calculations
   Format: {"action": "calculator", "parameters": {"expression": "math expression"}}
   Use when: User asks for calculations

3. get_time - Get current date and time
   Format: {"action": "get_time", "parameters": {}}
   Use when: User asks about current time or date

IMPORTANT: 
- If you need to use a tool, respond with ONLY the JSON (nothing else)
- After the tool returns results, then provide a natural language response using those results
- If you don't need a tool, answer normally using your knowledge and provided context"""
            
            if context:
                system_prompt += f"\n\nContext from uploaded documents:\n{context}"
            
            # Add to conversation history
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversations[conv_id][-6:])  # Last 3 exchanges
            messages.append({"role": "user", "content": request.query})
            
            # Get initial response from Ollama
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
            
            # Check if response contains an action request
            action_data = extract_json_action(response_text)
            
            if action_data:
                # Execute the action
                yield f"data: {json.dumps({'type': 'action', 'action': action_data['action']})}\n\n"
                
                action_result = action_handler.execute(
                    action_data["action"],
                    action_data.get("parameters", {})
                )
                
                yield f"data: {json.dumps({'type': 'action_result', 'result': action_result[:200] + '...'})}\n\n"
                
                # Send results back to LLM for natural response
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": f"Here are the results from the {action_data['action']} tool:\n\n{action_result}\n\nPlease provide a natural, helpful response based on these results. Do not use any more tools."})
                
                # Get final natural language response
                final_response = ""
                stream = ollama.chat(
                    model=config.MODEL_NAME,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": 0.7,
                    }
                )
                
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        final_response += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                # Update conversation with final response
                conversations[conv_id].append({"role": "user", "content": request.query})
                conversations[conv_id].append({"role": "assistant", "content": final_response})
            else:
                # No action needed, update conversation with direct response
                conversations[conv_id].append({"role": "user", "content": request.query})
                conversations[conv_id].append({"role": "assistant", "content": response_text})
            
            # Send sources if available
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