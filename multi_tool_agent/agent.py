from google.adk.agents import Agent
from google.adk.agents import LlmAgent,ParallelAgent, SequentialAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
 # This import might not be strictly needed for the given functions, but keeping it as it was in your original code.

Model = "gemini-2.5-flash"



def get_add(a: int, b: int):
    return a + b


# def get_greet() -> str:
#     return "hai how are you"


def get_sub(a: int, b: int) -> int:
    return a - b


Add = LlmAgent(
    name="Add_agent",
    model=Model,
    description="Agent to do some addition",
    instruction=(
        """You are a helpful agent who can add the two values.
        You can give answer in:
        'result': your answer
       
        """
    ),
    tools=[get_add] , # Added this line to match the original code's structure
    output_key="Add_result"  # Added this line to match the original code's structure
)

sub = LlmAgent(
    name="sub_agent",
    model=Model,
    description="Agent to do some subtract",
    instruction=(
        """You are a helpful agent who can subtract the two values.
        """
    ),
    tools=[get_sub], # Added a comma here
  # Added this line to match the original code's structure
    output_key="sub_result"  # Added this line to match the original code's structure
   
)



gather_output =ParallelAgent(
    name="mathematical_agent",
    description="Agent to do some  Adding and Subtracting",
    sub_agents=[Add, sub] # Changed 'sub_agent' to 'sub_agents' (plural)
)


synthesizer = LlmAgent(
    name="Synthesizer",
    model=Model,
        planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            
            thinking_budget=1024,
        )
    ),
    instruction=("Combine results from {sub_result} and {Add_result}. in a same response"))

root_agent = SequentialAgent(
    name="FetchAndSynthesize",
    sub_agents=[gather_output, synthesizer] # Run parallel fetch, then synthesize
)



