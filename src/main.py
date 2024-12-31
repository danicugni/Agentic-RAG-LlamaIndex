from llama_index.agent.openai import OpenAIAgent
from utils import single_query_engine_tools, multiple_query_engine_tools

def main():
    individual_query_engine_tools = single_query_engine_tools()
    query_engine_tool = multiple_query_engine_tools()
    tools = individual_query_engine_tools + [query_engine_tool]
    
    agent = OpenAIAgent.from_tools(tools, verbose=True)

    # response = agent.chat("What were some of the biggest risk factors in 2020 for Uber?")
    # print(str(response))
    
    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = agent.chat(text_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    main()