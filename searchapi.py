import os
from crewai import Agent, Task, Crew, LLM
from langchain_community.tools.searchapi.tool import SearchApiAPIWrapper
from crewai.tools import BaseTool
from pydantic import Field
from dotenv import load_dotenv
from typing import Type, Any
from langchain_google_genai import ChatGoogleGenerativeAI # Import the correct class

# Load environment variables
load_dotenv()

# Get API keys
GEMINI_API_KEY ="AIzaSyAVRyEWTJc8XejIP_uS113BT0WVRMiU4-0"
SEARCHAPI_KEY ="ZUQYTHAiysigNvWFcVAc2zvL"

class SearchTool(BaseTool):
    name: str = "search_web"
    description: str = "Search for latest news and summaries from the web"
    
    def _run(self, query: str) -> str:
        """Execute the search"""
        try:
            search_api = SearchApiAPIWrapper(searchapi_api_key=searchapi_key)
            result = search_api.run(query)
            return str(result)
        except Exception as e:
            return f"Search failed: {str(e)}"

def main():
    print("🚀 Starting with Google Gemini (FREE)...")
    
    try:
        # Use Google Gemini
        llm = LLM( # Correct instantiation of the LLM
            model="gemini/gemini-2.5-flash", # Use "gemini-pro" for free tier access via API key
            temperature=0.7
        )

        # Create the search tool
        search_tool = SearchTool()

        # Create agent
        agent = Agent(
            role="Web Search Expert",
            goal="Find and summarize the latest AI product launches",
            backstory="You are an expert at finding and summarizing AI product information.",
            tools=[search_tool],
            llm=llm,
            verbose=True
        )

        # Create task
        task = Task(
            description="Search for AI product launches in July 2025 and list the top 3.",
            expected_output="3 AI product announcements with descriptions.",
            agent=agent
        )

        # Create and run crew
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        result = crew.kickoff()
        print("\n\n✅ Final Result:\n", result)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()