from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os.path
import json
from datetime import datetime

# Initialize the LLMs and vector store components
llama = None
gpt4all = None
llama_chain = None
gpt4all_chain = None
embeddings = None
vector_store = None

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embeddings model: {str(e)}")

# Initialize vector store
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

try:
    vector_store = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
except Exception as e:
    print(f"Error initializing vector store: {str(e)}")

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
rag_prompt = PromptTemplate(
    input_variables=["context", "topic"],
    template="""You are a knowledgeable and helpful AI assistant. Using the provided context and your knowledge, please provide detailed, accurate, and well-structured information about {topic}.

Context information:
{context}

Your response should:
- Start with a clear introduction
- Include key concepts and definitions from the context
- Provide relevant examples where appropriate
- Be factual and informative
- End with a brief summary

Topic: {topic}
Response:"""
)

# Create the chains if models loaded successfully
if llama is not None:
    llama_chain = LLMChain(llm=llama, prompt=rag_prompt)
if gpt4all is not None:
    gpt4all_chain = LLMChain(llm=gpt4all, prompt=rag_prompt)

# Create responses directory if it doesn't exist
if not os.path.exists("responses"):
    os.makedirs("responses")

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

    # Get temperature input
    while True:
        try:
            temperature = float(input("Enter temperature (0.0 to 1.0): "))
            if 0.0 <= temperature <= 1.0:
                break
            else:
                print("Temperature must be between 0.0 and 1.0")
        except ValueError:
            print("Please enter a valid number")

    # Get user input for topic
    topic = input("Enter a topic to learn about: ")

    # Retrieve relevant documents from vector store
    if vector_store is not None:
        try:
            docs = vector_store.similarity_search(topic, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            context = ""
    else:
        context = ""

    # Run selected model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "llama" if model_choice == "1" else "gpt4all"
    
    if model_choice == "1":
        if llama_chain is not None:
            try:
                llama.temperature = temperature  # Update temperature
                response = llama_chain.run(context=context, topic=topic)
                print("\nLlama response:")
                print(response)
                
                # Save response
                response_data = {
                    "model": "Llama",
                    "topic": topic,
                    "temperature": temperature,
                    "context": context,
                    "response": response,
                    "timestamp": timestamp
                }
                with open(f"responses/{model_name}_{timestamp}.json", "w") as f:
                    json.dump(response_data, f, indent=4)
                print(f"\nResponse saved to responses/{model_name}_{timestamp}.json")
                
            except Exception as e:
                print(f"Error running Llama model: {str(e)}")
        else:
            print("Llama model is not available")
    elif model_choice == "2":
        if gpt4all_chain is not None:
            try:
                gpt4all.temperature = temperature  # Update temperature
                response = gpt4all_chain.run(context=context, topic=topic)
                print("\nGPT4All response:")
                print(response)
                
                # Save response
                response_data = {
                    "model": "GPT4All",
                    "topic": topic,
                    "temperature": temperature,
                    "context": context,
                    "response": response,
                    "timestamp": timestamp
                }
                with open(f"responses/{model_name}_{timestamp}.json", "w") as f:
                    json.dump(response_data, f, indent=4)
                print(f"\nResponse saved to responses/{model_name}_{timestamp}.json")
                
            except Exception as e:
                print(f"Error running GPT4All model: {str(e)}")
        else:
            print("GPT4All model is not available")
    else:
        print("Invalid selection. Please choose 1, 2, or 3.")
