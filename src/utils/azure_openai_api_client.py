import os
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def get_gpt_client(
    api_version: Optional[str] = None
) -> AzureOpenAI:
    """
    Create and return the Azure OpenAI client.
    
    Args:
        api_version: Optional override for API version
    """
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=api_version or "2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def get_gpt_response(
    query: str,
    context: str,
    additional_context: str = "",
    max_tokens: int = 50,
    temperature: float = 0.7
) -> str:
    """
    Get response from GPT model.
    
    Args:
        query: User's question
        context: Context for the question
        additional_context: Additional instructions for response formatting
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
    """
    try:
        client = get_gpt_client()
        
        system_message = "You are a helpful AI assistant. Using the provided context, answer the user's question. "
        if additional_context:
            system_message += additional_context
        system_message += " If the answer cannot be found in the context, say so. Always cite the source document."

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nPlease provide an answer based on the context above."}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating GPT response: {str(e)}"


# Only run the main function if this script is run directly
if __name__ == "__main__":
   
    # Test get_gpt_response function
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower."
    response = get_gpt_response(query, context)
    print("\nTest get_gpt_response:")
    print(f"Query: {query}")
    print(f"Response: {response}")