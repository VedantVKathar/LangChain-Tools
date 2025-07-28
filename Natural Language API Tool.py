import os
import requests
import json
from langchain_ollama import OllamaLLM
from langchain_community.agent_toolkits.nla.toolkit import NLAToolkit
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent

# -----------------------------------------------------------
# 1. TOOL DEFINITION
# -----------------------------------------------------------
@tool("Natural Language API Tool")
def natural_language_api_tool(query: str) -> str:
    """
    A tool that allows agents to interact with a specific API using natural language.
    This tool uses a LangChain Natural Language API (NLA) Toolkit
    to query an API described by an OpenAPI spec.
    """
    api_spec_url = "https://petstore3.swagger.io/api/v3/openapi.json"
    
    try:
        print(f"Fetching OpenAPI spec from: {api_spec_url} using NLAToolkit's built-in method.")
        
        # We will use the mistral model for this LLM instance
        llm = OllamaLLM(model="mistral")
        
        nla_toolkit = NLAToolkit.from_llm_and_url(
            llm=llm,
            open_api_url=api_spec_url
        )
        
        if not nla_toolkit or not nla_toolkit.get_tools():
            return "Error: NLA Toolkit could not be initialized or contains no tools."
            
        tool_to_use = nla_toolkit.get_tools()[0]
        
        return tool_to_use.run(query)
        
    except Exception as e:
        return f"Failed to load OpenAPI spec from {api_spec_url}: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
api_analyst_agent = Agent(
    role='API Analyst',
    goal=dedent("""\
        Interact with the Petstore API to get information about pets. Your single tool is the 'Natural Language API Tool'.
        You MUST use this tool to accomplish your task. Your only job is to formulate a correct 'query' string for the tool and execute it.
        You must output the correct JSON format for the tool's input.
        Example of a correct Action Input: `{"query": "find all available pets"}`."""),
    backstory=dedent("""\
        You are an expert at translating requests into precise, natural language queries for APIs.
        Your thought process must be focused on identifying the information needed and then crafting a single, concise natural language query.
        You MUST provide the Action and Action Input in the exact format required by the framework.
        Do NOT add any extra text, conversation, or full sentences to your 'Action' or 'Action Input' output, just the required tool name and a valid JSON object."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[natural_language_api_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION
# -----------------------------------------------------------
api_query_task = Task(
    description=dedent("""\
        Use the available tool to find all available pets in the store.
        Your final output must be a list of the pets that are available for purchase.
        Do not make any assumptions about the available pets; rely solely on the tool's output."""),
    expected_output=dedent("""\
        A clean, well-formatted list of all available pets, including their IDs and names.
        Example:
        - id: 123, name: 'doggie'
        - id: 456, name: 'kitty'"""),
    agent=api_analyst_agent
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
petstore_crew = Crew(
    agents=[api_analyst_agent],
    tasks=[api_query_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the Petstore API Crew")
    result = petstore_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the Petstore API Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "List all available pets."
    
    # CORRECTED LINE: Call the .run() method on the tool object.
    tool_output = natural_language_api_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")