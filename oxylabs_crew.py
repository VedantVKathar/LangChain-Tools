import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_oxylabs import OxylabsSearchAPIWrapper
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent
from typing import Optional

# This line loads the environment variables from the .env file
load_dotenv()

# -----------------------------------------------------------
# 1. TOOL DEFINITION (FUNCTION-BASED)
# -----------------------------------------------------------
@tool("Oxylabs Search Tool")
def oxylabs_search_tool(query: str) -> str:
    """
    A tool that performs a Google search using the Oxylabs Web Scraper API.
    The query should be a natural language string representing the search terms.
    """
    username = os.getenv("OXYLABS_USERNAME")
    password = os.getenv("OXYLABS_PASSWORD")
    
    if not username or not password:
        return "Error: OXYLABS_USERNAME or OXYLABS_PASSWORD environment variables not set."
    
    try:
        print(f"Initializing Oxylabs Search Tool with query: '{query}'")
        
        # The OxylabsSearchAPIWrapper uses the environment variables for authentication
        oxylabs_wrapper = OxylabsSearchAPIWrapper(
            oxylabs_username=username,
            oxylabs_password=password
        )
        
        return oxylabs_wrapper.run(query)
        
    except Exception as e:
        return f"Failed to run Oxylabs Search Tool: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
search_analyst_agent = Agent(
    role='Oxylabs Search Analyst',
    goal=dedent("""\
        Perform accurate and reliable Google searches using the Oxylabs Web Scraper API.
        Your single tool is the 'Oxylabs Search Tool'. You MUST use this tool to accomplish your task.
        Your only job is to formulate a correct 'query' string for the tool and execute it.
        Example of a correct Action Input: `{"query": "latest AI news"}`."""),
    backstory=dedent("""\
        You are a highly constrained AI expert at formulating search queries for web scraping services.
        Your thought process must be focused on identifying the information needed and then crafting a single, concise search query.
        You MUST provide the Action and Action Input in the exact format required by the framework.
        Do NOT add any extra text, conversation, or full sentences to your 'Action' or 'Action Input' output."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[oxylabs_search_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION - UPDATED
# -----------------------------------------------------------
search_task = Task(
    description=dedent("""\
        Use the available tool to find the most recent news headlines about artificial intelligence.
        Your final output must be a list of at least three headlines and their sources.
        Do not make any assumptions; rely solely on the tool's output."""),
    expected_output=dedent("""\
        A list of recent news headlines related to artificial intelligence.
        Example:
        - Headline 1: 'AI Breakthroughs at a Major Tech Conference' (Source: TechCrunch)
        - Headline 2: 'New AI Model Outperforms Competition' (Source: Wired)
        - Headline 3: 'Regulatory Scrutiny on AI Intensifies' (Source: The Wall Street Journal)"""),
    agent=search_analyst_agent
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
oxylabs_crew = Crew(
    agents=[search_analyst_agent],
    tasks=[search_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the Oxylabs Crew")
    result = oxylabs_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the Oxylabs Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "What is the latest news on CrewAI?"
    
    tool_output = oxylabs_search_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")