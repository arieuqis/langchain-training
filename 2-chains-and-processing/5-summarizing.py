from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

long_text = """
The Varginha UFO incident is one of the most famous UFO cases in Brazil and worldwide, occurring in January 1996 in the city of Varginha, Minas Gerais. The event involved multiple sightings of strange creatures and a UFO, witnessed by military personnel, firefighters, and civilians.

Timeline of Events:

January 20, 1996: Three young women (Liliane, Valquíria, and Katia) reported seeing a strange creature with red eyes and oily skin near a farm. They described it as approximately 1.6 meters tall with a small head and large eyes.

January 21, 1996: Military personnel and firefighters from the Varginha Fire Department were called to capture a creature. Witnesses reported seeing military trucks transporting something covered. The creature was reportedly taken to a local hospital and then transferred to a military facility.

January 22, 1996: A second creature was allegedly found dead near the same location. Witnesses described a strong ammonia-like odor in the area. Military personnel again secured the site and removed the body.

Following Days: Multiple witnesses reported seeing military vehicles and unusual activity in the region. Some witnesses claimed to have seen a UFO being transported on a truck covered by a tarp.

Investigation: The case was investigated by UFO researchers including Ubirajara Franco Rodrigues and Vitorio Pacaccini. They collected testimonies from military personnel, firefighters, and civilians who claimed to have seen the creatures and the military operation.

Official Response: Brazilian military authorities denied any knowledge of the incident. However, some military personnel later confirmed the events under anonymity.

The Varginha incident remains controversial, with believers claiming it's evidence of extraterrestrial contact and skeptics suggesting it was a case of misidentification or hoax. The case gained international attention and has been featured in numerous documentaries and investigations.
"""

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=70
)

# Create Document objects (like the course does)
docs = splitter.create_documents([long_text])

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)

# Combine all chunks into one text and summarize at once (like "stuff" chain type)
# Extract page_content from Document objects
combined_text = "\n\n".join([doc.page_content for doc in docs])

# Direct LLM invocation without explicit prompt (load_summarize_chain handled this internally)
result = llm.invoke(f"Summarize the following text:\n\n{combined_text}")
print("Summary:")
print(result.content)
