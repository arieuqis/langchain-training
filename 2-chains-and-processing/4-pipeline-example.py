from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n ``` {initial_text}```"
)

template_summarize = PromptTemplate(
    input_variables=["translated_text"],
    template="Summarize the following text:\n ``` {translated_text}```"
)

template_four_words_summary = PromptTemplate(
    input_variables=["summary"],
    template="Write a 4-word summary of the following text:\n ``` {summary}```"
)

llm = ChatOpenAI(model_name="gpt-5-mini", temperature=0)

# ============================================================
# APPROACH 1: Sequential chains (current implementation)
# ============================================================
# Each chain is complete (template + llm + parser)
# Output of one entire chain feeds directly into the next
# Limitation: Variable names must match exactly between chains
# IMPORTANT: This only works when each template has exactly ONE input_variable.
# LangChain automatically maps the string output to the single input variable.
# If a template expects multiple variables, this approach will fail and you must
# use APPROACH 2 (dictionary-based routing) for explicit mapping.

translate_chain = template_translate | llm | StrOutputParser()
summarize_chain = template_summarize | llm | StrOutputParser()
four_words_summary_chain = template_four_words_summary | llm | StrOutputParser()

pipeline_sequential = translate_chain | summarize_chain | four_words_summary_chain

# result = pipeline_sequential.invoke({"initial_text": "Hola, ¿cómo estás?"})
# print(result)

# ============================================================
# APPROACH 2: Dictionary-based routing (course approach)
# ============================================================
# More flexible: Use dictionary to map/rename outputs
# Allows explicit control over data flow between chains
# Can combine multiple chains into a single dictionary mapping

# Single translate chain
translate = template_translate | llm | StrOutputParser()

# Dictionary maps the output key to match the next template's input variable
# "translated_text" must match template_summarize's input_variables
pipeline_dict = {"translated_text": translate} | template_summarize | llm | StrOutputParser()

# For the 3-step pipeline, we can chain dictionaries:
pipeline_course = (
    {"translated_text": translate} 
    | template_summarize 
    | llm 
    | StrOutputParser()
)

# Or for the full 3-step example with dictionary mapping:
pipeline_full = (
    {"translated_text": translate} 
    | template_summarize 
    | llm 
    | StrOutputParser()
    | {"summary": lambda x: x}  # Pass through with key rename
    | template_four_words_summary 
    | llm 
    | StrOutputParser()
)

# Using the course approach
result = pipeline_course.invoke({"initial_text": "Hola, ¿cómo estás?"})
print(result)
