import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
)
# ChatPromptTemplate usage test 
non_formatted_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "please provide the result of : {input} into json in text format, no need .md format tag, without any other explaination"),
    ]
)

# for hw01 
# FewShotChatMessagePromptTemplate
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "以json格式，純文字形式列出全部結果：{input}, 不必說明"),
        ("ai", "{output}")
    ]
)

examples = [
    {"input": "2024年台灣10月紀念日有哪些?", 
     "output": """
     {
        "Result": [
          {
               "date": "2024-10-10",
               "name": "國慶日"
          }
        ]
    }
    """
    },
]

# A prompt template used to format each individual example.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# Add into ChatPromptTemplate for result
holiday_prompt = ChatPromptTemplate.from_messages(
    [
        few_shot_prompt,
        ('human', '{input}'),
    ]
)

def generate_hw01(question):
    prompt_value = holiday_prompt.format(input=question)
    return llm.invoke(prompt_value).content
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
