import os
from crewai import Agent, Task, Crew, Process, LLM
# Corrected import for BaseTool
from crewai.tools import BaseTool

# --- Set up your LLM ---
# Make sure to set your GOOGLE_API_KEY environment variable
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# Instantiate the Gemini LLM using CrewAI's LLM class
# This uses LiteLLM in the background to call the model.
llm = LLM(
    model="gemini/gemini-2.5-flash", # You can use models like gemini-pro, etc.
    temperature=0.7,
)


# --- Tool Definition ---
# In a real scenario, you would import this from a library.
# For this example, we'll create a mock PaymanAITool that
# simulates the behavior of the real tool. It inherits from
# CrewAI's BaseTool for seamless integration.

class PaymanAITool(BaseTool):
    """
    A mock tool that simulates sending a payment.
    It's designed to be used by a CrewAI agent.
    """
    name: str = "send_payment"
    description: str = (
        "Use this tool to send a payment to a specified payee. "
        "The tool requires the payment amount and the payee's ID."
    )

    def _run(self, amount: float, payee_id: str) -> str:
        """
        Simulates the action of sending a payment.

        Args:
            amount (float): The monetary amount to be sent.
            payee_id (str): The unique identifier for the recipient.

        Returns:
            str: A confirmation message indicating the transaction was successful.
        """
        # In a real tool, this is where you would call the PaymanAI SDK/API
        print("--- MOCK TOOL EXECUTION ---")
        print(f"Attempting to send ${amount} to {payee_id}...")
        # Simulate a successful API call
        confirmation_details = f"Payment of ${amount:.2f} to '{payee_id}' processed successfully. Transaction ID: 8a7d6s5f."
        print(f"Success: {confirmation_details}")
        print("--------------------------")
        return confirmation_details

# Instantiate your tool
payment_tool = PaymanAITool()


# --- Agent Definition ---
# Create a specialized agent to handle payments.
payment_agent = Agent(
    role='Financial Transactions Specialist',
    goal=(
        'To accurately process payment requests by identifying the '
        'correct amount and payee, and then using the send_payment tool.'
    ),
    backstory=(
        'You are an advanced AI agent with expertise in financial operations. '
        'Your sole purpose is to execute payment commands with precision and '
        'reliability. You must use the tools available to you and report back '
        'the final confirmation message.'
    ),
    # The agent's "memory" is enabled to allow for a conversational flow
    # if the task requires follow-up actions or clarifications.
    memory=True,
    # Set verbose to True to see the agent's thought process
    verbose=True,
    # The list of tools this agent can use
    tools=[payment_tool],
    # Assign the LLM to the agent
    llm=llm
)


# --- Task Definition ---
# Define the specific task for the agent to perform.
payment_task = Task(
    description=(
        "Process the following payment request: 'Send $10 to payee123.' "
        "Your final answer must be the confirmation message from the tool."
    ),
    expected_output=(
        "A string containing the final confirmation message of the successful payment, "
        "including the transaction ID."
    ),
    # Assign the task to your agent
    agent=payment_agent,
)


# --- Crew Definition ---
# Assemble the agent and task into a crew.
payment_crew = Crew(
    agents=[payment_agent],
    tasks=[payment_task],
    process=Process.sequential, # Tasks will be executed one after another
    # Corrected verbose to be a boolean value
    verbose=True
)


# --- Execute the Crew ---
# The kickoff() method starts the crew's execution.
print("🚀 Starting Crew Execution...")
result = payment_crew.kickoff()

print("\n✅ Crew Execution Finished.")
print("\nFinal Result:")
print(result)
