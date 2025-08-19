import asyncio
import aiohttp
import os
from typing import Union, List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr

# Get API keys with error handling
try:
    # Fix: Use the correct environment variable name
    groq_api_key = os.environ["GROQ_API_KEY"]
    print(f"✅ GROQ API Key loaded (starts with: {groq_api_key[:10]}...)")
except KeyError:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it before running the application.")

try:
    serpapi_api_key = os.environ["SERPAPI_API_KEY"]
    print(f"✅ SerpAPI Key loaded (starts with: {serpapi_api_key[:10]}...)")
except KeyError:
    raise ValueError("SERPAPI_API_KEY environment variable is not set. Please set it before running the application.")

# Initialize LLM with proper error handling
try:
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.0,
        streaming=True,
        api_key=groq_api_key  # Pass the string directly, not SecretStr
    ).configurable_fields(
        callbacks=ConfigurableField(
            id="callbacks",
            name="callbacks",
            description="A list of callbacks to use for streaming."
        )
    )
    print("✅ ChatGroq LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChatGroq: {e}")
    raise

prompt = ChatPromptTemplate.from_messages([
    ("system", (
    "You are an AI assistant that must always use a tool to answer the user's CURRENT question. "
    "Never reply directly. Always call a tool. "
    "You have the following tools: search, add, subtract, multiply, exponentiate, and final_answer. "
    "Only use tools provided to you. "
    "Once the tool(s) give the needed result, use the final_answer tool with BOTH fields: "
    "'answer' (string) and 'tools_used' (list of tool names you used)."
)),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# we use the article object for parsing serpapi results later.

class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result.get("title", ""),
            source=result.get("source", ""),
            link=result.get("link", ""),
            snippet=result.get("snippet", "")
        )

# Tools definition
# note: we define all tools as async to simplify later code, but only the serpapi
# tool is actually async 

@tool
async def add(x: float, y: float) -> float:
    """Add two numbers together."""
    return x + y

@tool
async def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y

@tool
async def exponentiate(x: float, y: float) -> float:
    """Raise x to the power of y."""
    return x ** y

@tool 
async def subtract(x: float, y: float) -> float:
    """Subtract x from y."""
    return y - x

@tool
async def search(query: str) -> List[Article]:
    """Search the web using SerpAPI and return a list of articles."""
    params = {
        "api_key": serpapi_api_key,
        "engine": "google",
        "q": query
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search",
                params=params
            ) as response:
                if response.status != 200:
                    print(f"SerpAPI error: {response.status}")
                    return []

                results = await response.json()
                if "organic_results" not in results:
                    print(f"No organic results in SerpAPI response: {results}")
                    return []

                return [Article.from_serpapi_result(result) for result in results["organic_results"]]
    except Exception as e:
        print(f"Error in search tool: {e}")
        return []

@tool
async def final_answer(answer: str, tools_used: List[str]) -> Dict[str, Union[str, List[str]]]:
    """Provide the final answer to the user with a list of tools used."""
    return {"answer": answer, "tools_used": tools_used}

tools = [add, subtract, multiply, exponentiate, final_answer, search]

name2tool = {tool.name: tool.coroutine for tool in tools}

# Function to execute tool calls
async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    try:
        tool_call_data = tool_call.tool_calls[0]

        # Handle both formats
        if "function" in tool_call_data:
            tool_name = tool_call_data["function"].get("name")
            tool_args = tool_call_data["function"].get("arguments", {})
        elif "name" in tool_call_data and "args" in tool_call_data:
            tool_name = tool_call_data["name"]
            tool_args = tool_call_data["args"]
        else:
            raise ValueError("Tool name missing in tool_call")


        if not tool_name:
            raise ValueError("Tool name missing in tool_call")

        tool_out = await name2tool[tool_name](**tool_args)
        return ToolMessage(
            content=f"{tool_out}",
            tool_call_id=tool_call_data.get("id", "unknown")
        )
    except Exception as e:
        print(f"Error executing tool: {e}")
        return ToolMessage(
            content=f"Error: {str(e)}",
            tool_call_id=tool_call.tool_calls[0].get("id", "unknown")
        )


# streaming handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return

            if token_or_done:
                yield token_or_done

    async def on_llm_new_token(self, *args, **kwargs) -> None:
        try:
            chunk = kwargs.get("chunk")
            if chunk and chunk.message.additional_kwargs.get("tool_calls"):
                if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                    self.final_answer_seen = True

            self.queue.put_nowait(kwargs.get("chunk"))
        except Exception as e:
            print(f"Error in on_llm_new_token: {e}")

    async def on_llm_end(self, *args, **kwargs) -> None:
        try:
            if self.final_answer_seen:
                self.queue.put_nowait("<<DONE>>")
            else:
                self.queue.put_nowait("<<STEP_END>>")
        except Exception as e:
            print(f"Error in on_llm_end: {e}")

# agent executor
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        self.chat_history: List[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice=None)
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> Dict:
        try:
            count = 0
            final_answer: Optional[str] = None
            final_answer_call: Optional[Dict] = None
            agent_scratchpad: List[Union[AIMessage, ToolMessage]] = []

            async def stream(query: str) -> List[AIMessage]:
                try:
                    response = self.agent.with_config(
                        callbacks=[streamer]
                    )

                    outputs = []
                    async for token in response.astream({
                        "input": query, 
                        "chat_history": self.chat_history,
                        "agent_scratchpad": agent_scratchpad
                    }):
                        tool_calls = token.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            if tool_calls[0].get("id"):
                                outputs.append(token)
                            else:
                                if outputs:
                                    outputs[-1] += token

                    return [
                        AIMessage(
                            content=x.content,
                            tool_calls=x.tool_calls,
                        ) for x in outputs
                    ]
                except Exception as e:
                    print(f"Error in stream function: {e}")
                    return []

            while count < self.max_iterations:
                tool_calls = await stream(query=input)
                if not tool_calls:
                    break
                    
                tool_obs = await asyncio.gather(
                    *[execute_tool(tool_call) for tool_call in tool_calls]
                )

                id2tool_obs = {
                    tool_call.tool_calls[0]["id"]: tool_obs[i] 
                    for i, tool_call in enumerate(tool_calls)
                }
                
                for tool_call in tool_calls:
                    agent_scratchpad.extend([
                        tool_call, 
                        id2tool_obs[tool_call.tool_calls[0]["id"]]
                    ])

                count += 1
                found_final_answer = False
                for tool_call in tool_calls:
                    if tool_call.tool_calls[0].get("function", {}).get("name", "") == "final_answer":
                        final_answer_call = tool_call.tool_calls[0]
                        final_answer = final_answer_call["function"]["arguments"]["answer"]
                        found_final_answer = True
                        break

                if found_final_answer:
                    break

            self.chat_history.extend([
                HumanMessage(content=input),
                AIMessage(content=final_answer if final_answer else "No answer found")
            ])

            return final_answer_call if final_answer_call else {"answer": "No answer found", "tools_used": []}
        
        except Exception as e:
            print(f"Error in agent executor invoke: {e}")
            import traceback
            print(traceback.format_exc())
            return {"answer": f"Error: {str(e)}", "tools_used": []}

agent_executor = CustomAgentExecutor()
print("✅ Agent executor initialized successfully")

