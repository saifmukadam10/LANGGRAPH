import os
from typing import TypedDict, List
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY was not loaded; check the .env path and variable name."
    )


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b"),
    temperature=0.0,
    max_tokens=1024,
)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"AI : {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
app = graph.compile()

user_input = input("Enter")
app.invoke({"messages": [HumanMessage(content=user_input)]})
