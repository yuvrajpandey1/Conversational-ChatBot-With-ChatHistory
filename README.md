
# Conversation Q&A ChatBot

Chat Bot by using Gemma2-9b-It and ChatGroq basically its a Q&A and we will Manage Session using Message History to get answer of past question which already we gave to bot and transalte in different languages.



# Documentation 

[Chat Models](https://python.langchain.com/v0.1/docs/modules/model_io/chat/)

Chat Models are a core component of LangChain.A chat model is a language model that uses chat messages as inputs and returns chat messages as outputs (as opposed to using plain text).

LangChain has integrations with many model providers (OpenAI, Cohere, Hugging Face, etc.) and exposes a standard interface to interact with all of these models.

[Messages](https://python.langchain.com/v0.1/docs/modules/model_io/chat/quick_start/)

The chat model interface is based around messages rather than raw text. The types of messages currently supported in LangChain are AIMessage, HumanMessage, SystemMessage, FunctionMessage and ChatMessage -- ChatMessage takes in an arbitrary role parameter. Most of the time, you'll just be dealing with HumanMessage, AIMessage, and SystemMessage.


[Custome Chat Model](https://python.langchain.com/v0.1/docs/modules/model_io/chat/custom_chat_model/)

Chat models take messages as inputs and return a message as output.

LangChain has a few built-in message types:

* SystemMessage: Used for priming AI behavior, usually passed in as the first of a sequence of input messages.  

* HumanMessage: Represents a message from a person interacting with the chat model.

* AIMessage: Represents a message from the chat model. This can be either text or a request to invoke a tool.

* FunctionMessage / ToolMessage: Message for passing the results of tool invocation back to the model.

* AIMessageChunk / HumanMessageChunk: Chunk variant of each type of message.











 








## Important Libraries Used

 - [ChatMessageHistory](https://python.langchain.com/v0.1/docs/modules/memory/chat_messages/)
 - [BaseChatMessageHistory](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)
- [RunnableWithMessageHistory](https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/)
 - [RunnablePassthrough](https://python.langchain.com/v0.1/docs/expression_language/primitives/passthrough/)
 - [PromptTemplate](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/)







## Plateform or Providers

 - [LangChain-OpenAI](https://python.langchain.com/docs/integrations/providers/openai/)
 - [LangChain Hub](https://smith.langchain.com/hub)

## Model

 - LLM - Llama3-8b-8192


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install langchain_groq
  pip install langchain-core


```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, ChatGroq

**Server:** Anaconda Navigator, Jupyter Notebook


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`GROQ_API_KEY`



## Examples
Messages that are passed in from a human to the model
```javascript
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant! Your name is Bob."
    ),
    HumanMessage(
        content="What is your name?"
    )
]

# Instantiate a chat model and invoke it with the messages
model = ...
print(model.invoke(messages))
```

## Managing memory outside of a chain

```javascript
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()

history.add_user_message("hi!")

history.add_ai_message("whats up?")
```
## Message history input and returns a message as output

```javascript
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model
```

