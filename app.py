from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All

# Initialize the LLMs
llama = LlamaCpp(
    model_path="models/llama3_1", 
    temperature=0.7,
    max_tokens=2000,
    n_ctx=2048,
    verbose=True
)

gpt4all = GPT4All(
    model="models/gpt4all-j-v1.3-groovy",
    temperature=0.7,
    max_tokens=2000,
    verbose=True
)

# Create prompt templates
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me about {topic}."
)

# Create the chains
llama_chain = LLMChain(llm=llama, prompt=prompt)
gpt4all_chain = LLMChain(llm=gpt4all, prompt=prompt)

# Run both chains
llama_response = llama_chain.run("artificial intelligence")
gpt4all_response = gpt4all_chain.run("artificial intelligence")

print("Llama response:")
print(llama_response)
print("\nGPT4All response:")
print(gpt4all_response)
