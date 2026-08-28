from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict



load_dotenv()


llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0)


# create calulator tool


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"



# Give Tool to LLM

llm_with_tools = llm.bind_tools([calculator])


# Define State 

# 5. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]


 # 6. Agent function
def agent(state: State):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }



# 7. Tool function
def tool_node(state: State):

    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]

    result = calculator.invoke(
        tool_call["args"]
    )

    return {
        "messages": [
            {
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call["id"]
            }
        ]
    }



# 8. Decide next step
def should_continue(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool"

    return "end"




# 9. Create graph
graph = StateGraph(State)


# 10. Add nodes
graph.add_node("agent", agent)
graph.add_node("tool", tool_node)


# 11. Connect START to Agent
graph.add_edge(
    START,
    "agent"
)


# 12. Conditional routing
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        "end": END
    }
)


# 14. Compile graph
app = graph.compile()


# 15. Get user question
question = input("\nAsk a question: ")


# 16. Start the graph
result = app.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    }
)


# 17. Print final answer
print("\nFinal Answer:")

print(
    result["messages"][-1].content
)