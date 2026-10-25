import os
from typing import TypedDict, List, Union
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
