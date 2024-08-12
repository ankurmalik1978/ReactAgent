import json
from typing import List, Union
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools import Tool, tool
from langchain import HuggingFacePipeline
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
# This is a copy of main.py with an addition of ScratchPad
# For detailed comments please check main.py
# In "main.py" we have now executed the Agent once. We now have an option of either accepting the response or re-execute with the previous History.
# In LangChain, agent_scratchpad is a sequence of messages that contains the previous agent tool invocations and their corresponding outputs.
# It's used to inject the past conversation, which contains the agent's thought process so far.
# This allows the next thought-action-observation loop to access all thoughts and actions within the current agent executor chain, which enables continuity in agent actions
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


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


    # We have enhanced the prompt to include the ScratchPad
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

                    Thought: {agent_scratchpad}
                """

    tools = [get_text_length]

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    model = "C:/Users/I038077/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33"

    hfgppl = HuggingFacePipeline.from_model_id(
        model_id=model,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        model_kwargs={"temperature": 0},  # {"stop":["\nObservation", "Observation"]}
    )

    chat_model_with_stop = hfgppl.bind(stop=["\nObservation", "Observation"])

    # We introduce a list which will hold results from previous invocations 
    intermediate_steps = []

    # We enahnced the dictionary to include agent_scratchpad
    # We have to additionally parse using format_log_to_str since agent_scratchpad contains "agent_step" which is a tuple of AgentAction and observation strings amd LLM does not understand this
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )

    # During call the agent, we will additionally pass in the prompt agent_scratchpad and value will be intermediate_steps
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of the text DOG?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    print(f"Agent Step: {agent_step}")

    if isinstance(agent_step, AgentAction):

        tool_name = agent_step.tool

        tool_to_use = find_tool_by_name(tools, tool_name)

        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")

        # we append the agent_step and observation to the intermediate_steps list
        intermediate_steps.append(agent_step, str(observation))
    else:
        returnObj = agent_step.return_values["output"]
        print(f"Final Answer: {returnObj}")


    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length in characters of the text DOG?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    print(f"Agent Step: {agent_step}")

    if isinstance(agent_step, AgentAction):

        tool_name = agent_step.tool

        tool_to_use = find_tool_by_name(tools, tool_name)

        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append(agent_step, str(observation))
    else:
        returnObj = agent_step.return_values["output"]
        print(f"Final Answer: {returnObj}")
