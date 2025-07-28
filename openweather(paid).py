import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent

# This line loads the environment variables from the .env file
load_dotenv()

# -----------------------------------------------------------
# 1. TOOL DEFINITION (FUNCTION-BASED)
# -----------------------------------------------------------
@tool("OpenWeatherMap Tool")
def openweathermap_tool(query: str) -> str:
    """
    A tool that fetches current weather information for a specific location.
    The query should be a natural language string specifying the location
    and the type of information needed, e.g., 'current temperature in London'.
    """
    # The API key is now retrieved from the environment variables loaded by dotenv
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "Error: OPENWEATHERMAP_API_KEY environment variable not set. Please check your .env file."

    # Explicitly check for the pyowm dependency
    try:
        import pyowm
    except ImportError:
        return "Failed to run OpenWeatherMap Tool: pyowm is not installed. Please install it with `pip install pyowm`"

    try:
        print(f"Initializing OpenWeatherMap Tool with query: '{query}'")
        
        # The OpenWeatherMapAPIWrapper requires the API key to be passed
        weather_api = OpenWeatherMapAPIWrapper(openweathermap_api_key=api_key)
        
        # The OpenWeatherMapAPIWrapper acts as a tool itself, with a single run method
        return weather_api.run(query)
        
    except Exception as e:
        return f"Failed to run OpenWeatherMap Tool: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
weather_analyst = Agent(
    role='Weather Analyst',
    goal=dedent("""\
        Find and report the current weather for specific locations using ONLY the 'OpenWeatherMap Tool'.
        Your sole purpose is to formulate a correct 'query' string for the tool and nothing else.
        Example of a correct Action Input: `{"query": "current temperature in New York"}`."""),
    backstory=dedent("""\
        You are a highly constrained, specialized AI. Your entire world is limited to a single tool: the 'OpenWeatherMap Tool'. 
        You cannot perform any other actions, commands, or installations. Your only function is to use the tool.
        Your thought process must be focused exclusively on identifying the location and weather metric from the user's request.
        You MUST provide the Action and Action Input in the exact, valid JSON format required by the framework.
        Do NOT add any extra text, conversation, or full sentences to your 'Action' or 'Action Input' output."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[openweathermap_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION
# -----------------------------------------------------------
check_weather_task = Task(
    description=dedent("""\
        Use the available tool to find the current weather in London, UK.
        Your final output must state the current temperature, humidity, and weather conditions.
        Rely solely on the output from the tool."""),
    expected_output=dedent("""\
        A concise summary of London's current weather.
        Example:
        - Location: London, UK
        - Temperature: 15°C
        - Humidity: 75%
        - Conditions: Cloudy"""),
    agent=weather_analyst
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
weather_crew = Crew(
    agents=[weather_analyst],
    tasks=[check_weather_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the OpenWeatherMap Crew")
    result = weather_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the OpenWeatherMap Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "What is the wind speed in Berlin, Germany?"
    
    tool_output = openweathermap_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")