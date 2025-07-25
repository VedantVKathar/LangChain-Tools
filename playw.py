import os
from crewai import Agent, Task, Crew, Process
from crewai import LLM

# CORRECTED IMPORTS:
# The PlaywrightBrowserTool is now in its own 'tool' module.
from langchain_community.tools.playwright.tool import PlaywrightBrowserTool 
from langchain_community.tools.playwright.utils import create_sync_playwright_browser

# 🚀 Set your Gemini API key as an environment variable
# For Google, the environment variable is typically GOOGLE_API_KEY
os.environ["GOOGLE_API_KEY"] = "AIzaSyA4U4TG5vIYZLXE9d7V0N7LtDbxpndu7mU"

# -----------------------------------------------------------
# 1. INITIALIZE THE GEMINI LLM
# -----------------------------------------------------------
llm = LLM(
    model="gemini-1.5-flash",
    temperature=0.4,
)

# -----------------------------------------------------------
# 2. SET UP THE PLAYWRIGHT TOOL
# -----------------------------------------------------------
sync_browser = create_sync_playwright_browser()
browser_tool = PlaywrightBrowserTool(
    sync_browser=sync_browser,
    browser="chromium"
)

# -----------------------------------------------------------
# 3. DEFINE AGENTS WITH THE GEMINI LLM
# -----------------------------------------------------------
web_navigator = Agent(
    role='Expert Web Navigator',
    goal='Navigate to a specific URL and extract requested information from the page.',
    backstory=(
        "You are a skilled web scraping expert, proficient in using programmatic "
        "browser tools to visit websites, inspect their content, and extract "
        "specific pieces of data like headlines, prices, or summaries."
    ),
    tools=[browser_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

analyst = Agent(
    role='Senior Data Analyst',
    goal='Analyze the extracted web content and provide a clear, concise summary.',
    backstory=(
        "With a keen eye for detail, you are an expert at sifting through raw text "
        "to find relevant insights and structuring messy information "
        "into a clean, human-readable report."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# -----------------------------------------------------------
# 4. DEFINE THE TASKS
# -----------------------------------------------------------
navigation_task = Task(
    description=(
        "Navigate to the URL 'https://web.archive.org/web/20240701000000*/https://www.reuters.com/technology/'. "
        "Find the main headline of the top story on the page and extract its text. "
        "Also, find the text of the first two sub-headlines you can see."
    ),
    expected_output="A string containing the text of the main headline and the two sub-headlines.",
    agent=web_navigator
)

analysis_task = Task(
    description=(
        "Take the extracted headlines and format them into a clear, bulleted list. "
        "The final report should have a title like 'Top Tech Headlines' "
        "followed by the bulleted list."
    ),
    expected_output="A well-formatted markdown report with a title and a bulleted list of the headlines.",
    context=[navigation_task],
    agent=analyst
)

# -----------------------------------------------------------
# 5. ASSEMBLE AND RUN THE CREW
# -----------------------------------------------------------
tech_news_crew = Crew(
    agents=[web_navigator, analyst],
    tasks=[navigation_task, analysis_task],
    process=Process.sequential,
    verbose=True
)

print("🚀 Kicking off the Crew with Gemini to test Playwright...")
result = tech_news_crew.kickoff()

print("\n\n########################")
print("✅ Crew execution finished!")
print("Final Result:")
print(result)
print("########################")

sync_browser.close()