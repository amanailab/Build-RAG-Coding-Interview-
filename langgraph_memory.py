from dotenv import load_dotenv
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated, TypedDict


# 1. Load environment variables
load_dotenv()


# 2. Create LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 3. Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 4. Agent node
def agent(state: State):

    response = llm.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# 5. Create graph
graph = StateGraph(State)


# 6. Add node
graph.add_node("agent", agent)


# 7. Add edges
graph.add_edge(START, "agent")
graph.add_edge("agent", END)


# 8. Create memory
memory = MemorySaver()


# 9. Compile with memory
app = graph.compile(
    checkpointer=memory
)


# 10. Thread ID
config = {
    "configurable": {
        "thread_id": "user-1"
    }
}


# 11. Conversation loop
while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config
    )

    print(
        "\nAI:",
        result["messages"][-1].content
    )