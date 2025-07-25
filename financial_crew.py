import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool  # Correct import for @tool decorator
from dotenv import load_dotenv
from langchain_community.utilities.polygon import PolygonAPIWrapper
from langchain_community.tools.polygon import PolygonTickerNews, PolygonLastQuote

load_dotenv()

# --- SETUP ---
# Set your Polygon API Key in environment variable for security
POLYGON_API_KEY = "mRWfFNVFxHhY6yfvdTbAGUBwnIlCxVA3"  # Better to use env variable
os.environ["POLYGON_API_KEY"] = POLYGON_API_KEY

# Check if API key is set
if not os.environ.get("POLYGON_API_KEY"):
    raise ValueError("Polygon API key not found. Please set the POLYGON_API_KEY environment variable.")

# --- TOOL DEFINITION ---
# Create a single, shared instance of the API wrapper
polygon_api_wrapper = PolygonAPIWrapper()

# Create tool instances
polygon_news_tool = PolygonTickerNews(api_wrapper=polygon_api_wrapper)
polygon_quote_tool = PolygonLastQuote(api_wrapper=polygon_api_wrapper)

# Define custom wrapper functions with @tool decorator
@tool("Get Stock News")
def get_stock_news(ticker: str) -> str:
    """Fetches the most recent news articles for a given stock ticker."""
    try:
        return polygon_news_tool.run(ticker)
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"

@tool("Get Stock Quote")
def get_stock_quote(ticker: str) -> str:
    """Fetches the last trade price for a given stock ticker."""
    try:
        return polygon_quote_tool.run(ticker)
    except Exception as e:
        return f"Error fetching quote for {ticker}: {str(e)}"

# --- LLM SETUP ---
# You need to define your LLM here. Example with OpenAI:


# Make sure to set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.4,
    api_key=os.getenv("GEMINI_API_KEY"),
    max_retries=2,
)

# --- AGENT DEFINITION ---
financial_analyst = Agent(
    role='Senior Financial Analyst',
    goal='Provide the latest stock price and news for a given company ticker.',
    backstory=(
        "As a seasoned financial analyst with over 15 years of experience at a top Wall Street firm, "
        "you specialize in providing rapid, data-driven insights on publicly traded companies. "
        "You are known for your ability to quickly synthesize market data, including real-time prices and breaking news, "
        "to inform investment decisions."
    ),
    verbose=True,
    llm=llm,
    allow_delegation=False,
    tools=[get_stock_quote, get_stock_news]  # Use the decorated functions
)

# --- TASK DEFINITION ---
analysis_task = Task(
    description="Analyze the current financial status of NVIDIA (NVDA). "
                "First, get its most recent stock quote. "
                "Second, fetch the top 3 latest news headlines for it. "
                "Finally, compile this information into a brief, easy-to-read report.",
    expected_output="A concise report containing the latest stock quote for NVDA and a bulleted list of the top 3 recent news headlines.",
    agent=financial_analyst
)

# --- CREW DEFINITION ---
financial_crew = Crew(
    agents=[financial_analyst],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True
)

# --- EXECUTION ---
if __name__ == "__main__":
    print("🚀 Kicking off the Financial Analysis Crew...")
    try:
        result = financial_crew.kickoff()
        
        # Print the final result
        print("\n\n########################")
        print("## Final Report")
        print("########################\n")
        print(result)
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print("Please check your API keys and internet connection.")