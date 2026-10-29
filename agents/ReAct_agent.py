from typing import (
    TypedDict,
    Annotated,
    Sequence,
)  # sequence is used to automatically manage state updates by adding new messages in chat history
from dotenv import load_dotenv
from langchain_core.messages import (
    ToolMessage,
)  # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import (
    BaseMessage,
)  # The foundational class for all message types in LangGraph
from langchain_core.messages import (
    SystemMessage,
)  # Message for providing instructions to the LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import (
    add_messages,
)  # this is a reducer function . it does not append the data evrytime but leverages using code in one line but updating the current state with the new state.
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage], add_messages
    ]  # annotated gives additional context  withoutn affecting the type itself


@tool  # decorater
def add(a: int, b: int):
    """This is an additon function that adds two integers together"""
    return a + b


tools = [add]
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    max_tokens=1024,
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [
        (
            "user",
            "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.",
        )
    ]
}
print_stream(app.stream(inputs, stream_mode="values"))
