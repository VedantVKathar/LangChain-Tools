import os
import requests
import json
from langchain_ollama import OllamaLLM
from langchain_opengradient import OpenGradientToolkit # Corrected import statement
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent

# -----------------------------------------------------------
# 1. TOOL DEFINITION (FUNCTION-BASED)
# -----------------------------------------------------------
@tool("OpenGradient Tool")
def opengradient_tool(query: str) -> str:
    """
    A tool that allows agents to interact with the OpenGradient API using natural language.
    This tool uses a LangChain OpenGradient Toolkit to perform tasks like
    retrieving and summarizing datasets.
    """
    try:
        print(f"Initializing OpenGradient Toolkit with query: '{query}'")
        
        # We will use the mistral model for the toolkit's LLM
        llm = OllamaLLM(model="mistral")
        
        # Instantiate the toolkit from the LLM
        opengradient_toolkit = OpenGradientToolkit.from_llm(llm=llm)
        
        if not opengradient_toolkit or not opengradient_toolkit.get_tools():
            return "Error: OpenGradient Toolkit could not be initialized or contains no tools."
            
        # The OpenGradient toolkit returns a list of tools.
        # We will iterate and find the most relevant tool for the query
        # For simplicity, we'll try to find a 'get_dataset' or 'summarize' tool
        available_tools = opengradient_toolkit.get_tools()
        tool_to_use = None
        
        # Simple logic to select a tool based on the query,
        # otherwise default to the first tool.
        if "summarize" in query.lower():
            tool_to_use = next((t for t in available_tools if "summarize" in t.name), available_tools[0])
        else:
            tool_to_use = next((t for t in available_tools if "dataset" in t.name), available_tools[0])

        print(f"Selected tool to use: {tool_to_use.name}")
        
        # Execute the query using the selected tool's run method
        return tool_to_use.run(query)
        
    except Exception as e:
        return f"Failed to run OpenGradient Tool: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
opengradient_analyst = Agent(
    role='OpenGradient Analyst',
    goal=dedent("""\
        Interact with the OpenGradient API to retrieve and analyze information about datasets.
        Your single tool is the 'OpenGradient Tool'. You MUST use this tool to accomplish your task.
        Your only job is to formulate a correct 'query' string for the tool and execute it.
        Example of a correct Action Input: `{"query": "get the dataset 'imagenet'"}`."""),
    backstory=dedent("""\
        You are an expert at using the OpenGradient API to find and summarize machine learning datasets.
        Your thought process must be focused on identifying the user's request and crafting a precise
        natural language query to pass to the tool.
        You MUST provide the Action and Action Input in the exact format required by the framework.
        Do NOT add any extra text or conversational filler to your 'Action' or 'Action Input' output."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[opengradient_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION
# -----------------------------------------------------------
opengradient_query_task = Task(
    description=dedent("""\
        Use the available tool to find the dataset named 'imagenet' and provide a summary of it.
        Your final output must be a concise summary of the dataset, including its size, purpose, and key characteristics.
        Rely solely on the output from the tool."""),
    expected_output=dedent("""\
        A clear, well-formatted summary of the 'imagenet' dataset.
        Example:
        - Name: ImageNet
        - Description: A large visual database designed for use in visual object recognition software research.
        - Size: 14 million images..."""),
    agent=opengradient_analyst
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
opengradient_crew = Crew(
    agents=[opengradient_analyst],
    tasks=[opengradient_query_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the OpenGradient API Crew")
    result = opengradient_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the OpenGradient API Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "Summarize the dataset named 'cifar10'."
    
    # Correctly call the .run() method on the tool object.
    tool_output = opengradient_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")