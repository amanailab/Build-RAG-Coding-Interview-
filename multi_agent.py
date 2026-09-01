import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool


# 1. Load API key
load_dotenv()


# 2. Create LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# 3. Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    try:
        return str(eval(expression))
    except:
        return "Invalid expression."


# 4. Weather tool
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""

    weather = {
        "delhi": "35°C and sunny",
        "mumbai": "30°C and cloudy",
        "agra": "34°C and sunny",
        "london": "18°C and rainy"
    }

    return weather.get(
        city.lower(),
        "Weather information not available."
    )


# 5. Word count tool
@tool
def word_count(text: str) -> str:
    """Count the number of words."""

    return str(len(text.split()))


# 6. Put all tools together
tools = [
    calculator,
    get_weather,
    word_count
]


# 7. Give tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# 8. Get user question
question = input("\nAsk a question: ")


# 9. Store conversation
messages = [
    ("user", question)
]


# 10. Agent loop
while True:

    response = llm_with_tools.invoke(messages)

    messages.append(response)

    # No tool needed
    if not response.tool_calls:

        print("\nFinal Answer:")
        print(response.content)

        break


    # Tool required
    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print("\nTool:", tool_name)
        print("Arguments:", tool_args)


        # Find requested tool
        for current_tool in tools:

            if current_tool.name == tool_name:

                result = current_tool.invoke(tool_args)

                print("Result:", result)


                # Send result back to LLM
                messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call["id"]
                    }
                )