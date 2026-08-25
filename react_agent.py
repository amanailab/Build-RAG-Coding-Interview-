from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool



load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    model = "openai/gpt-oss-20b",
    temperature = 0
)

# Tools

@tool
def calculator(expression:str) -> str:
    """A simple calculator tool that evaluates mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Bind Tool + LLM

llm_with_tools = llm.bind_tools(
    [calculator]
    )

question = input("\n Ask a questions:")

message = [
    ("user", question)
]

# Agent loop

while True:

    # Ask LLM What to do
    response = llm_with_tools.invoke(message)

    #Add LLM response to conversation
    message.append(response)

    # NO tool required

    if not response.tool_calls:
        print("Final Answer: ", response.content)
        break


    # Excute toool calls required 

    for tool_call in response.tool_calls:
        print("\n Tools Called")
        print(tool_call["name"])

        print("Arguments:")
        print(tool_call["args"])

        if tool_call["name"] == "calculator":
            result = calculator.invoke(tool_call["args"])
            print("Tool Result: ", result)

            # Send tool result back to LLM

            message.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call["id"]
            })

            