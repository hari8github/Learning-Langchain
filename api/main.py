import asyncio
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Request model for JSON requests
class InvokeRequest(BaseModel):
    content: str

# Initialize application
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add a simple health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Simple test endpoint that doesn't use the agent
@app.post("/test")
async def test_endpoint(request: InvokeRequest):
    try:
        print(f"Test endpoint received: {request.content}")
        return {"message": f"Received: {request.content}", "status": "success"}
    except Exception as e:
        print(f"Error in test endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Simple streaming test
async def simple_stream_generator(content: str):
    try:
        yield f"<step><step_name>test</step_name>"
        yield '{"message": "This is a test"}'
        yield "</step>"
        yield f"<step><step_name>final_answer</step_name>"
        yield f'{{"answer": "You asked: {content}", "tools_used": ["test"]}}'
    except Exception as e:
        print(f"Error in stream generator: {e}")
        print(traceback.format_exc())
        yield f"Error: {str(e)}"

@app.post("/test-stream")
async def test_stream(request: InvokeRequest):
    try:
        print(f"Test stream endpoint received: {request.content}")
        return StreamingResponse(
            simple_stream_generator(request.content),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    except Exception as e:
        print(f"Error in test stream endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Now let's try to import and use the agent with proper error handling
@app.post("/invoke")
async def invoke(request: InvokeRequest):
    try:
        print(f"Invoke endpoint received: {request.content}")
        
        # Try to import the agent here to see if that's causing issues
        try:
            from agent import QueueCallbackHandler, agent_executor
            print("Successfully imported agent modules")
        except Exception as import_error:
            print(f"Error importing agent: {import_error}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Import error: {str(import_error)}")
        
        # Try to create the queue and streamer
        try:
            queue: asyncio.Queue = asyncio.Queue()
            streamer = QueueCallbackHandler(queue)
            print("Successfully created queue and streamer")
        except Exception as queue_error:
            print(f"Error creating queue/streamer: {queue_error}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Queue error: {str(queue_error)}")
        
        # Try to create the token generator
        async def token_generator(content: str, streamer: QueueCallbackHandler):
            try:
                print(f"Starting token generator for: {content}")
                task = asyncio.create_task(agent_executor.invoke(
                    input=content,
                    streamer=streamer,
                    verbose=True
                ))
                
                async for token in streamer:
                    try:
                        if token == "<<STEP_END>>":
                            yield "</step>"
                        elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                            if tool_name := tool_calls[0]["function"]["name"]:
                                yield f"<step><step_name>{tool_name}</step_name>"
                            if tool_args := tool_calls[0]["function"]["arguments"]:
                                yield tool_args
                    except Exception as token_error:
                        print(f"Error processing token: {token_error}")
                        continue
                
                await task
            except Exception as gen_error:
                print(f"Error in token generator: {gen_error}")
                print(traceback.format_exc())
                yield f"Error: {str(gen_error)}"
        
        return StreamingResponse(
            token_generator(request.content, streamer),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except Exception as e:
        print(f"Error in invoke endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)