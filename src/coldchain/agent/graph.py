"""LangGraph ReAct-style agent: reasoner <-> tools, with durable per-thread memory.

The system prompt is applied at call time rather than stored in state, so prompt
changes take effect for existing conversations and are never duplicated in history.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from coldchain.agent.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    system_prompt: str = SYSTEM_PROMPT,
):
    model = llm.bind_tools(list(tools))
    system = SystemMessage(content=system_prompt)

    def reasoner(state: AgentState) -> dict:
        return {"messages": [model.invoke([system, *state["messages"]])]}

    g = StateGraph(AgentState)
    g.add_node("reasoner", reasoner)
    g.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    g.add_edge(START, "reasoner")
    g.add_conditional_edges("reasoner", tools_condition, {"tools": "tools", END: END})
    g.add_edge("tools", "reasoner")
    return g.compile(checkpointer=checkpointer)
