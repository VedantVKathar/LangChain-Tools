import os
from langchain_ollama import OllamaLLM
from langchain_community.llms import OCIGenAI
from crewai.tools import tool
from crewai import Agent, Task, Crew, Process, LLM
from textwrap import dedent
from typing import Optional

# -----------------------------------------------------------
# 1. TOOL DEFINITION (FUNCTION-BASED)
# -----------------------------------------------------------
@tool("OciGenerateText Tool")
def oci_generate_text_tool(query: str) -> str:
    """
    A tool that generates text using the Oracle Cloud Infrastructure Generative AI service.
    This tool uses the OCIGenAI model to produce a text response based on a given query.
    The query should be a natural language string to prompt the model.
    """
    try:
        # The OCI SDK will automatically use the credentials configured in ~/.oci/config
        llm = OCIGenAI(
            model_id="cohere.command",  # Example model, can be changed
            service_endpoint="https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com", # Update with your endpoint
            compartment_id="YOUR_COMPARTMENT_OCID",  # IMPORTANT: Replace with your compartment's OCID
        )

        response = llm.invoke(query)
        return response
    
    except Exception as e:
        return f"Failed to run OciGenerateText Tool: {e}"

# -----------------------------------------------------------
# 2. LLM CONFIGURATION
# -----------------------------------------------------------
llm_config = LLM(
    model="ollama/mistral"
)

# -----------------------------------------------------------
# 3. AGENT DEFINITION
# -----------------------------------------------------------
oracle_ai_agent = Agent(
    role='Oracle AI Text Generator',
    goal=dedent("""\
        Generate high-quality text using the Oracle AI generative text model. Your single tool is the 'OciGenerateText Tool'.
        You MUST use this tool to accomplish your task. Your only job is to formulate a correct 'query' string for the tool and execute it.
        Example of a correct Action Input: `{"query": "write a short poem about the ocean"}`."""),
    backstory=dedent("""\
        You are a highly constrained AI expert at crafting prompts for Oracle's generative text models.
        Your thought process must be focused on identifying the text generation task and then crafting a concise natural language query for the tool.
        You MUST provide the Action and Action Input in the exact format required by the framework.
        Do NOT add any extra text or conversational filler to your 'Action' or 'Action Input' output."""),
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
    tools=[oci_generate_text_tool]
)

# -----------------------------------------------------------
# 4. TASK DEFINITION
# -----------------------------------------------------------
text_generation_task = Task(
    description=dedent("""\
        Use the available tool to write a short paragraph about the benefits of cloud computing.
        Your final output must be the generated text, clearly formatted.
        Rely solely on the output from the tool."""),
    expected_output=dedent("""\
        A well-structured paragraph about the benefits of cloud computing.
        Example:
        "Cloud computing offers many benefits, including scalability, cost-effectiveness, and enhanced security..." """),
    agent=oracle_ai_agent
)

# -----------------------------------------------------------
# 5. CREW DEFINITION
# -----------------------------------------------------------
oracle_crew = Crew(
    agents=[oracle_ai_agent],
    tasks=[text_generation_task],
    process=Process.sequential,
    verbose=True
)

# -----------------------------------------------------------
# 6. EXECUTION AND TESTING
# -----------------------------------------------------------
if __name__ == "__main__":
    print("## Starting the Oracle AI Crew")
    result = oracle_crew.kickoff()
    print("\n\n################################################")
    print("## Final Result of the Oracle AI Crew")
    print("################################################")
    print(result)

    print("\n\n################################################")
    print("## Direct Tool Testing")
    print("################################################")
    
    test_query = "Write a brief summary about artificial intelligence."
    
    tool_output = oci_generate_text_tool.run(test_query)
    
    print(f"Test Query: '{test_query}'")
    print(f"Tool Output: {tool_output}")