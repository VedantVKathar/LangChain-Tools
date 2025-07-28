import os
import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_pandas_dataframe_agent
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent

# This line loads the environment variables from the .env file
load_dotenv()

# -----------------------------------------------------------
# 1. TOOL DEFINITION (FUNCTION-BASED)
# -----------------------------------------------------------
# Load the DataFrame from the specified CSV file
csv_file_path = r"C:\Users\maliv\OneDrive\Desktop\name,age,department,salary.csv" # Use raw string for path

try:
    df = pd.read_csv(csv_file_path)
    print(f"DataFrame loaded successfully from {csv_file_path}")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}. Please check the path and filename.")
    # Fallback to a hardcoded DataFrame if the file isn't found, for initial testing
    data = {
        'name': ['John', 'Jane', 'Doe', 'Alice', 'Bob'],
        'age': [25, 30, 35, 40, 45],
        'department': ['Sales', 'Marketing', 'Sales', 'Engineering', 'Marketing'],
        'salary': [50000, 60000, 70000, 80000, 90000]
    }
    df = pd.DataFrame(data)
    print("Using hardcoded DataFrame as fallback.")


# LLM for the pandas agent
pandas_llm = OllamaLLM(model="mistral:latest")

# Create the specialized pandas agent
pandas_agent = create_pandas_dataframe_agent(
    pandas_llm,
    df,
    verbose=True,
    allow_dangerous_code=True # This is crucial for allowing code execution
)

@tool("Pandas Data Analyst Tool")
def pandas_data_analyst_tool(query: str) -> str:
    """
    A tool that queries and manipulates a pandas DataFrame.
    The query should be a natural language string describing the operation.
    """
    try:
        print(f"Executing pandas query: '{query}'")
        # Instruct the internal agent to be concise in its final answer
        result = pandas_agent.run(f"{query}. Provide the final numerical answer directly, without any conversational text or explanation.")
        return str(result) # Ensure it's a string for tool output
    except Exception as e:
        return f"Failed to run Pandas Data Analyst Tool: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral:latest",
    base_url="http://localhost:11434"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
data_analyst_agent = Agent(
    role='Data Analyst',
    goal=dedent("""\
        Answer questions by analyzing a given pandas DataFrame.
        Your single tool is the 'Pandas Data Analyst Tool'.
        You MUST use this tool and ONLY this tool. Your only job is to formulate a correct 'query' string for the tool and nothing else.
        Example of a correct Action Input: `{"query": "calculate the average salary"}`."""),
    backstory=dedent("""\
        You are an expert at extracting insights from structured data using pandas.
        Your thought process must be focused on translating natural language questions into executable commands for the **Pandas Data Analyst Tool**.
        You MUST provide the Action and Action Input in the exact format required by the framework.
        Do NOT add any extra text or conversational filler to your 'Action' or 'Action Input' output. The Action must be exactly 'Pandas Data Analyst Tool'."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[pandas_data_analyst_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION
# -----------------------------------------------------------
analysis_task = Task(
    description=dedent("""\
        Use the available tool to find the average age of all employees in the DataFrame.
        Your final output must be the single numerical value for the average age.
        Rely solely on the output from the tool."""),
    expected_output=dedent("""\
        A clean, well-formatted summary of the average age.
        Example:
        - Average Age: 34.7"""),
    agent=data_analyst_agent
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
pandas_crew = Crew(
    agents=[data_analyst_agent],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the Pandas Crew")
    result = pandas_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the Pandas Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "What is the average age of employees in the 'Sales' department? Return only the number."
    
    tool_output = pandas_data_analyst_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")