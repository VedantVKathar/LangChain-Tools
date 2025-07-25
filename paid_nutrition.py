import os
from crewai import Agent, Task, Crew, Process, LLM # Import LLM from crewai
from crewai.tools import tool
from langchain_community.tools.passio_nutrition_ai import NutritionAI
from langchain_community.utilities.passio_nutrition_ai import NutritionAIAPI
from dotenv import load_dotenv
from langchain_core.utils import get_from_env
# The specific ChatGoogleGenerativeAI class is no longer needed

# Load environment variables from .env file
load_dotenv()

# --- Passio NutritionAI Tool Setup ---
# Retrieve the NutritionAI subscription key from environment variables
try:
    nutritionai_subscription_key = get_from_env(
        "nutritionai_subscription_key", "NUTRITIONAI_SUBSCRIPTION_KEY"
    )
except Exception as e:
    print(f"Error loading NutritionAI subscription key: {e}")
    print("Please ensure NUTRITIONAI_SUBSCRIPTION_KEY is set in your .env file.")
    nutritionai_subscription_key = None

# Initialize the NutritionAIAPI client and the Langchain tool
nutritionai_langchain_tool = None
if nutritionai_subscription_key:
    try:
        # Initialize NutritionAIAPI without arguments.
        # It automatically reads the NUTRITIONAI_SUBSCRIPTION_KEY from the environment.
        nutritionai_api_client = NutritionAIAPI()
        # CORRECTED: The NutritionAI tool now expects the client via the 'api_wrapper' parameter.
        nutritionai_langchain_tool = NutritionAI(api_wrapper=nutritionai_api_client)
    except Exception as e:
        print(f"Failed to initialize NutritionAI tool: {e}")
else:
    print("NutritionAI tool cannot be initialized without a subscription key.")


@tool("NutritionAI Search Tool")
def nutritionai_search_tool(query: str) -> str:
    """
    Searches Passio NutritionAI for detailed nutritional information about food items or ingredients.
    This tool is useful for retrieving data like calories, macronutrients, and micronutrients.

    Args:
        query (str): The food item or ingredient to search for (e.g., "apple", "100g chicken breast", "banana smoothie").

    Returns:
        str: A summary of the nutritional information from Passio NutritionAI.
             Returns an error message if the API key is missing or an error occurs during the search.
    """
    if not nutritionai_langchain_tool:
        return "NutritionAI tool is not initialized. Please provide a valid NUTRITIONAI_SUBSCRIPTION_KEY."
    try:
        # Invoke the original NutritionAI tool with the query
        result = nutritionai_langchain_tool.invoke(query)
        return result
    except Exception as e:
        return f"An error occurred while searching NutritionAI: {e}"

# --- Gemini LLM Setup ---
# Retrieve the Google API key from environment variables
try:
    google_api_key = get_from_env("google_api_key", "GOOGLE_API_KEY")
except Exception as e:
    print(f"Error loading Google API key: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file for Gemini LLM.")
    google_api_key = None

# Initialize the Gemini LLM using CrewAI's LLM class
gemini_llm = None
if google_api_key:
    # Using the crewai.LLM class as requested.
    # The model name is prefixed with 'gemini/' and the key is passed via the 'api_key' parameter.
    gemini_llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.7,
        api_key=google_api_key
    )
else:
    print("Gemini LLM cannot be initialized without a Google API key.")

# --- CrewAI Agent Definition ---
# Define the Nutrition Expert Agent
# Note: The agent will be None if the LLM failed to initialize
nutrition_expert_agent = None
if gemini_llm:
    nutrition_expert_agent = Agent(
        role='Nutrition Expert',
        goal='Provide detailed and accurate nutritional information for various food items.',
        backstory="""You are a highly knowledgeable nutrition expert with access to the Passio NutritionAI database.
                     Your expertise lies in breaking down food items into their core nutritional components,
                     including calories, macronutrients, and key micronutrients. You are precise and
                     always strive to deliver comprehensive and easy-to-understand nutrition facts.""",
        verbose=True,
        allow_delegation=False,
        tools=[nutritionai_search_tool],
        llm=gemini_llm # Assign the initialized LLM object
    )
else:
    print("Nutrition Expert Agent cannot be created without an initialized LLM.")


# --- CrewAI Task Definition ---
# Define the task for the Nutrition Expert Agent
nutrition_analysis_task = None
if nutrition_expert_agent:
    nutrition_analysis_task = Task(
        description="""Analyze the nutritional content of '{food_item}'.
                       Provide a detailed breakdown including:
                       - Total Calories
                       - Macronutrients (Protein, Carbs, Fats)
                       - Key Micronutrients (e.g., Vitamin C, Iron, Calcium - if available)
                       - Any other relevant nutritional facts.
                       Present the information clearly and concisely.""",
        expected_output="""A well-structured summary of the nutritional information for the specified food item,
                           including calories, macronutrients, and key micronutrients.
                           Example:
                           'For 1 large apple (approx. 223g):
                           - Calories: 116 kcal
                           - Protein: 0.6g
                           - Carbohydrates: 30.8g (Fiber: 5.4g, Sugars: 23.2g)
                           - Fats: 0.4g
                           - Vitamin C: 10.3mg (11% DV)
                           - Potassium: 235mg (5% DV)
                           - Other: Rich in antioxidants.'""",
        agent=nutrition_expert_agent,
        tools=[nutritionai_search_tool]
    )

# --- Crew Definition and Execution ---
# Form the Crew
nutrition_crew = None
if nutrition_expert_agent and nutrition_analysis_task:
    nutrition_crew = Crew(
        agents=[nutrition_expert_agent],
        tasks=[nutrition_analysis_task],
        process=Process.sequential,
        verbose=True
    )

# Main execution block
if __name__ == '__main__':
    if not nutrition_crew:
        print("\nSkipping CrewAI execution because the crew could not be created.")
        print("Please check your API keys and ensure all components are initialized correctly.")
    else:
        # Get input from the user for the food item
        food_item_input = input("Enter the food item you want to analyze (e.g., '1 large banana', '1 cup cooked rice'): ")

        # Kick off the crew with the user's input
        print(f"\n--- Starting Nutrition Analysis for: {food_item_input} ---\n")
        result = nutrition_crew.kickoff(inputs={'food_item': food_item_input})
        print("\n--- Nutrition Analysis Complete ---")
        print(result)
