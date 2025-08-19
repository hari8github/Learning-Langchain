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
    GROQ_API_KEY = SecretStr(os.environ["LANGCHAIN_API_KEY"])  # Using LANGCHAIN_API_KEY from your .env
except KeyError:
    raise ValueError("LANGCHAIN_API_KEY environment variable is not set. Please set it before running the application.")

try:
    SERPAPI_API_KEY = SecretStr(os.environ["SERPAPI_API_KEY"])
except KeyError:
    raise ValueError("SERPAPI_API_KEY environment variable is not set. Please set it before running the application.")

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.0,
    streaming=True,
    api_key=GROQ_API_KEY
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming."
    )
)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. When you have "
        "all the information you need, you MUST use the final_answer tool "
        "to provide a final answer to the user. Use tools to answer the "
        "user's CURRENT question, not previous questions."
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
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"]
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
async def serpapi(query: str) -> List[Article]:
    """Search the web using SerpAPI and return a list of articles."""
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://serpapi.com/search",
            params=params
        ) as response:
            results = await response.json()

    return [Article.from_serpapi_result(result) for result in results["organic_results"]]

@tool
async def final_answer(answer: str, tools_used: List[str]) -> Dict[str, Union[str, List[str]]]:
    """Provide the final answer to the user with a list of tools used."""
    return {"answer": answer, "tools_used": tools_used}

tools = [add, subtract, multiply, exponentiate, final_answer, serpapi]

name2tool = {tool.name: tool.coroutine for tool in tools}

# Function to execute tool calls
async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["function"]["name"]
    tool_args = tool_call.tool_calls[0]["function"]["arguments"]
    tool_out = await name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
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
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True

        self.queue.put_nowait(kwargs.get("chunk"))

    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

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
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> Dict:
        count = 0
        final_answer: Optional[str] = None
        final_answer_call: Optional[Dict] = None
        agent_scratchpad: List[Union[AIMessage, ToolMessage]] = []

        async def stream(query: str) -> List[AIMessage]:
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
                if tool_call.tool_calls[0]["function"]["name"] == "final_answer":
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

agent_executor = CustomAgentExecutor()