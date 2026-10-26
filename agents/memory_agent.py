import os
from typing import TypedDict, List, Union
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY was not loaded; check the .env path and variable name."
    )

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_ID", "openai/gpt-oss-20b"),
    temperature=0.0,
    max_tokens=1024,
)


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
