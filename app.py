from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
import os.path

# Initialize the LLMs
llama = None
gpt4all = None
llama_chain = None
gpt4all_chain = None

try:
    if not os.path.exists("models/llama3_1"):
        raise FileNotFoundError("Llama model file not found")
        
    llama = LlamaCpp(
        model_path="models/llama3_1", 
        temperature=0.7,
        max_tokens=2000,
        n_ctx=2048,
        verbose=True
    )
except Exception as e:
    print(f"Error loading Llama model: {str(e)}")

try:
    if not os.path.exists("models/gpt4all-j-v1.3-groovy"):
        raise FileNotFoundError("GPT4All model file not found")
        
    gpt4all = GPT4All(
        model="models/gpt4all-j-v1.3-groovy",
        temperature=0.7,
        max_tokens=2000,
        verbose=True
    )
except Exception as e:
    print(f"Error loading GPT4All model: {str(e)}")

# Create prompt templates
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me about {topic}."
)

# Create the chains if models loaded successfully
if llama is not None:
    llama_chain = LLMChain(llm=llama, prompt=prompt)
if gpt4all is not None:
    gpt4all_chain = LLMChain(llm=gpt4all, prompt=prompt)

while True:
    # Get user input for model selection
    print("\nSelect a model:")
    print("1. Llama")
    print("2. GPT4All")
    print("3. Exit")
    model_choice = input("Enter 1, 2, or 3: ")

    if model_choice == "3":
        print("Goodbye!")
        break

    # Get user input for topic
    topic = input("Enter a topic to learn about: ")

    # Run selected model
    if model_choice == "1":
        if llama_chain is not None:
            try:
                response = llama_chain.run(topic)
                print("\nLlama response:")
                print(response)
            except Exception as e:
                print(f"Error running Llama model: {str(e)}")
        else:
            print("Llama model is not available")
    elif model_choice == "2":
        if gpt4all_chain is not None:
            try:
                response = gpt4all_chain.run(topic)
                print("\nGPT4All response:")
                print(response)
            except Exception as e:
                print(f"Error running GPT4All model: {str(e)}")
        else:
            print("GPT4All model is not available")
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")
