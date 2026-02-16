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


def should_use_web_search(query: str) -> bool:
    """Smarter heuristic - only trigger for truly current/external info"""
    query_lower = query.lower()
    
    # ONLY trigger for clearly time-sensitive or external queries
    # Must have BOTH a trigger word AND context indicating external info needed
    
    # Strong indicators of needing web search
    strong_triggers = [
        'current price', 'price of', 'cost of', 'weather',
        'latest news', 'recent news', 'breaking news',
        'stock price', 'bitcoin price', 'crypto price',
        'what happened today', 'what is happening',
        'news today', 'today\'s news'
    ]
    
    # Explicit search requests
    explicit_search = [
        'search for', 'search the web', 'look up online',
        'find information about', 'google', 'web search'
    ]
    
    # Check for strong triggers
    if any(trigger in query_lower for trigger in strong_triggers):
        return True
    
    # Check for explicit search requests
    if any(phrase in query_lower for phrase in explicit_search):
        return True
    
    # Questions about documents, uploaded content, or general knowledge should NOT trigger
    if any(word in query_lower for word in ['document', 'uploaded', 'file', 'guidance', 'report', 'inventory', 'based on']):
        return False
    
    # Questions about processes, plans, recommendations should NOT trigger
    if any(word in query_lower for word in ['how do i', 'how should i', 'what should i', 'recommend', 'suggest', 'plan', 'strategy', 'path forward']):
        return False
    
    return False


@app.get("/")
async def root():
    return {"message": "Local LLM Agent API", "status": "running"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection: str = config.DEFAULT_COLLECTION
):
    """Upload and process a document"""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )
    
    file_path = config.UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    result = rag_engine.process_document(file_path, collection)
    return result


@app.post("/query")
async def query_stream(request: QueryRequest):
    """Query the LLM with RAG context (streaming response)"""
    
    async def generate():
        try:
            conv_id = request.conversation_id or "default"
            if conv_id not in conversations:
                conversations[conv_id] = []
            
            # Check if we should auto-trigger web search
            force_web_search = should_use_web_search(request.query)
            
            if force_web_search:
                # Auto-trigger web search for current info queries
                yield f"data: {json.dumps({'type': 'action', 'action': 'web_search'})}\n\n"
                
                # Extract search query from user question
                search_query = request.query
                # Clean up the query
                search_query = re.sub(r'^(can you |please |could you )', '', search_query, flags=re.IGNORECASE)
                search_query = re.sub(r'^(search|look up|find|tell me about) ', '', search_query, flags=re.IGNORECASE)
                
                # Execute search
                search_result = action_handler.execute("web_search", {"query": search_query, "max_results": 3})
                
                yield f"data: {json.dumps({'type': 'action_result', 'result': 'Search completed'})}\n\n"
                
                # Now ask LLM to format the results
                context = ""
                sources = []
                if request.use_rag:
                    docs = rag_engine.query(request.query, request.collection)
                    if docs:
                        context = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs])
                        sources = [{"source": doc.metadata.get('source'), "content": doc.page_content[:200]} for doc in docs]
                
                system_prompt = f"""You are a helpful AI assistant. Answer the user's question using the web search results provided.

Web Search Results:
{search_result}

"""
                if context:
                    system_prompt += f"\nAdditional Context from Documents:\n{context}"
                
                system_prompt += "\n\nProvide a clear, natural answer based on the search results. Do not mention that you performed a search."
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query}
                ]
                
                # Stream the formatted response
                response_text = ""
                stream = ollama.chat(
                    model=config.MODEL_NAME,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.7}
                )
                
                for chunk in stream:
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        response_text += content
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                conversations[conv_id].append({"role": "user", "content": request.query})
                conversations[conv_id].append({"role": "assistant", "content": response_text})
                
                if sources:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
                
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            
            # Normal flow for document-based and general queries
            context = ""
            sources = []
            if request.use_rag:
                docs = rag_engine.query(request.query, request.collection)
                if docs:
                    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}" for doc in docs])
                    sources = [{"source": doc.metadata.get('source'), "content": doc.page_content[:200]} for doc in docs]
            
            system_prompt = """You are a helpful AI assistant with expertise in analyzing documents and providing guidance.

When answering questions:
- Use information from uploaded documents when available
- Provide practical, actionable advice
- Structure your responses clearly
- Do NOT suggest using tools or actions unless explicitly asked to search the web
- Focus on answering the question directly based on available context

"""
            
            if context:
                system_prompt += f"\n\nContext from uploaded documents:\n{context}\n\nUse this context to answer the user's question."
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversations[conv_id][-6:])
            messages.append({"role": "user", "content": request.query})
            
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
            
            conversations[conv_id].append({"role": "user", "content": request.query})
            conversations[conv_id].append({"role": "assistant", "content": response_text})
            
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