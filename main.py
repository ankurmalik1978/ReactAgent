import json
from typing import List, Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools import Tool, tool
from langchain import HuggingFacePipeline
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish


# The core idea of agents is to use a language model to choose a sequence of actions to take.
# In chains, a sequence of actions is hardcoded (in code).
# In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.
# Description: We will create a custom function. Then we will wrap this function and make it available to the React app as a React Tool
# Tools are functions that an agent can invoke. They are used to perform actions that require computation or data processing.
# For this we will use a decorator called @tool. This decorator will wrap the function and make it available to the React app as a React Tool
# Its important to provide a description for the function. This description will be used to determine the right Tool for the task
@tool
def get_text_length(text: str) -> int:
    """Return the length of the input text."""

    print(f"get_text_length called with text: {text=}")
    # Remove any newlines, double quotes
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found in tools")


if __name__ == "__main__":
    print("Hello React Langchain!")

    # We will use a prompt
    # Get the prompt from Langchain Hub
    # Search for Langchain Hub  in Google and goto https://smith.langchain.com/hub
    # Search for react within agents and goto https://smith.langchain.com/hub/hwchase17/react
    # Copy the prompt and use it below

    # This will be the (chain of thought and few shot) prompt we will use and will help us in finding the right Tool for the task
    template = """
                    Answer the following questions as best you can. You have access to the following tools:

                    {tools}

                    Use the following format:

                    Question: the input question you must answer

                    Thought: you should always think about what to do

                    Action: the action to take, should be one of [{tool_names}]

                    Action Input: the input to the action

                    Observation: the result of the action

                    ... (this Thought/Action/Action Input/Observation can repeat N times)

                    Thought: I now know the final answer

                    Final Answer: the final answer to the original input question

                    Begin!

                    Question: {input}

                    Thought:
                """

    # We are creating set of Tools which will be used by the agent to find the right Tool for the task
    tools = [get_text_length]

    # Partial method will populate and plugin the placeholders in the template
    # We have "tools" and "tool_names" placeholders in the template
    # tool_names will be a commar seperated list of tool names
    # "input" placeholder will come dynamically when we run the prompt
    # Since LLMs depend on texts, we need to pass Tool Description as text. This is done using render_text_description method from langchain
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    model = "C:/Users/I038077/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"
    # model = "C:/Users/I038077/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473"

    # kwargs is for Keyword Arguments and is used for passign additional parameters to LLM
    # Here we are passing temperature and stop parameters to LLM
    # stop parameter is used to stop the generation of text when the text contains the specified string

    # Limiting Generation: max_new_tokens allows you to control the length of the generated text. This is crucial for managing response time, resource consumption, and preventing excessively long outputs.
    # Preventing Redundancy: By setting a reasonable limit, you can prevent the model from generating repetitive or irrelevant content that goes beyond the desired scope.

    hfgppl = HuggingFacePipeline.from_model_id(
        model_id=model,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        model_kwargs={"temperature": 0},  # {"stop":["\nObservation", "Observation"]}
    )

    # We will bind the stop parameter to the model
    # Adding "stop" parameter directly to the model did not work. We need to bind it to the model
    chat_model_with_stop = hfgppl.bind(stop=["\nObservation", "Observation"])

    # Pipe operator takes the output of the left side and passes it as input to the right side
    # So in the statement below we are taking return from the prompt execution which is a "PromptValue" and plugging into llm
    # But, we have not yet passed input into the Prompt. We will add that to the left
    # Input is known when the chain is executed. So we will pass it as a dictionary
    # We will implement a lambda function which takes the input disctioanry and fetches the "input" key
    # LLM has determined what is the right tool to perform the action
    # We need to now parse this text and extract tool name and input and make a call to the tool
    # LLM provides ReActSingleInputOutputParser to perform this parsing. We will add this to the agent pipe.

    agent = (
        {"input": lambda x: x["input"]}
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    # Return is a Union[AgentAction, AgentFinish]
    # This means that it could return either AgentAction or AgentFinish
    # AgentAction is a dataclass that represents the action an agent should take.
    # AgentAction consists of the name of the tool to execute and the input to pass to the tool.
    # This will be used to call the tool which LLM selected from its Reasoning
    # AgentFinish represents the final result from an agent, when it is ready to return to the user.
    # It contains a return_values key-value mapping, which contains the final agent output.
    # Usually, this contains an output key containing a string that is the agent's response.
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "What is the length in characters of the text DOG?"}
    )

    # If the return type is AgentAction, it will contain the name of the tool to execute and the input to pass to the tool
    # We check if the return type is AgentAction
    if isinstance(agent_step, AgentAction):
        # If it is AgentAction, we will get the tool name and input from the return value
        tool_name = agent_step.tool

        # From the tool name, we will get from the list of tools the tool which matches the name
        tool_to_use = find_tool_by_name(tools, tool_name)

        tool_input = agent_step.tool_input

        # We have the tool determined by LLM and the tool input
        # We will now execute the tool using "func" method
        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
    else:
        returnObj = agent_step.return_values["output"]
        print(f"Final Answer: {returnObj}")
