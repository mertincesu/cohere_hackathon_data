from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = LlamaCpp(
    model_path="models/llama3_1",
    temperature=0.7,
    max_tokens=2000,
    n_ctx=2048,
    verbose=True
)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me about {topic}."
)

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run("artificial intelligence")
print(response)
