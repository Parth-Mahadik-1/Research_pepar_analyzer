import streamlit as st 
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO

from typing import Optional
from langchain.output_parsers import OutputFixingParser



# ------------------ Setup ------------------




st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper Summarizer (LLM)")
st.write("Summarize academic papers with technical depth and clarity.")

# ------------------ LLM ------------------


llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-V3.1",
    task = "text-generation",
    temperature = 0.7,
)

model = ChatHuggingFace(llm=llm)

class summarize(BaseModel):
    Overview: Optional[str] = Field(default="Insufficient information available")
    Key_insights: Optional[str] = Field(default="Insufficient information available")
    Mathematical_and_Technical: Optional[str] = Field(default="Insufficient information available")
    Analogies_and_Intuitive_Explanations: Optional[str] = Field(default="Insufficient information available")
    Critical_Perspective: Optional[str] = Field(default="Insufficient information available")
    Clarity_and_Accessibility: Optional[str] = Field(default="Insufficient information available")
    Important: Optional[str] = Field(default="Insufficient information available")

parser_summarize = PydanticOutputParser(pydantic_object=summarize)

fixing_parser = OutputFixingParser.from_llm(
    parser=parser_summarize,
    llm=model
)


prompt = PromptTemplate(

    template="""
You are an expert academic writer and technical communicator.

Please create a thorough and well-structured summary of the research paper titled "{paper_input}" using the following specifications:

Explanation Style: {style_input}  
Target Length: {length_input}

Guidelines for the Summary:
1. **Overview**  
   - Begin with a brief but clear statement of the paper’s main objective and its importance in the broader field.  
   - Identify the key research questions, hypotheses, or problems the authors set out to address.  
   - Describe the methodology, experimental setup, or theoretical framework in sufficient detail for an informed reader to understand the approach.

2. **Key_insights**  
   - Highlight the principal results, discoveries, or arguments.  
   - Explain why these findings matter and how they advance current knowledge or practice.

3. **Mathematical_and_Technical **  
   - Include any critical mathematical equations or algorithms when they are central to understanding the work.  
   - Present these equations in a clean, readable format and, where appropriate, illustrate the concepts with short, intuitive code snippets or pseudo-code to make them more approachable.

4. **Analogies_and_Intuitive_Explanations **  
   - When concepts are complex, use clear analogies or real-world comparisons to simplify them without losing accuracy.  
   - Make sure these analogies remain faithful to the technical content.

5. **Critical_Perspective**  
   - Note any limitations, open questions, or potential future directions the authors mention.  
   - Avoid adding unsupported speculation; stick to what can be inferred from the text.

6. **Clarity_and_Accessibility**  
   - Ensure the final summary reads smoothly and logically, with appropriate transitions between sections.  
   - Match the requested style ({style_input}) and stay close to the target length ({length_input}) while preserving accuracy and nuance.

Important:  
If the paper does not provide certain information—for example, specific equations, data details, or conclusions—state clearly: "Insufficient information available" instead of guessing or inventing details.

Your goal is to deliver a balanced, precise, and reader-friendly summary that captures both the technical depth and the broader significance of the research.
\n 

{formate_instruction}


"""
,
input_variables=["paper_input", "style_input", "length_input"],
partial_variables={"formate_instruction":parser_summarize.get_format_instructions()}
    
)

chain = prompt | model | fixing_parser

# ------------------ Sidebar Controls ------------------
st.sidebar.header("Controls")

paper_title = st.sidebar.text_input(
    "Research Paper Title",
    value="Attention Is All You Need"
)

style = st.sidebar.selectbox(
    "Explanation Style",
    ["Expert", "Beginner", "Mixed"]
)

length = st.sidebar.selectbox(
    "Summary Length",
    ["Short", "Moderate", "Detailed"]
)

st.sidebar.header("Logs")
log_box = st.sidebar.empty()


# ------------------ KEY Fucntions ------------------

def sidebar_log(message):
    log_box.markdown(f"- {message}")

def generate_pdf(summary, paper_title):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"<b>{paper_title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    def add_section(title, content):
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(content.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 12))

    add_section("Overview", summary.Overview)
    add_section("Key Insights", summary.Key_insights)
    add_section("Mathematical and Technical Details", summary.Mathematical_and_Technical)
    add_section("Analogies and Intuitive Explanations", summary.Analogies_and_Intuitive_Explanations)
    add_section("Critical Perspective", summary.Critical_Perspective)
    add_section("Clarity and Accessibility", summary.Clarity_and_Accessibility)
    add_section("Importance of the Study", summary.Important)

    doc.build(story)
    buffer.seek(0)
    return buffer

def format_math(text):
    if "$$" in text:
        st.markdown(text)
    else:
        st.write(text)

generate = st.sidebar.button("Generate Summary")



# ------------------ Generate Summary ------------------
if generate:
    log_box.empty()  # clear previous logs

    if not paper_title.strip():
        sidebar_log("Waiting for paper title input")
        st.warning("Please enter a research paper title.")
    else:
        sidebar_log("Paper title received")
        sidebar_log("Initializing model pipeline")

        with st.spinner("Generating summary..."):
            try:
                sidebar_log("Invoking LLM")

                result = chain.invoke({
                    "paper_input": paper_title,
                    "style_input": style,
                    "length_input": length
                })

                sidebar_log("Summary generation completed")
                sidebar_log("Rendering output")

                st.success("Summary Generated Successfully!")

                pdf_buffer = generate_pdf(result, paper_title)

                # ------------------ Display Output ------------------
                st.markdown("### 1. Overview")
                st.write(result.Overview)

                st.markdown("### 2. Key Insights")
                st.write(result.Key_insights)

                st.markdown("### 3. Mathematical and Technical Details")

                math_text = result.Mathematical_and_Technical

                # Render text + equations properly
                st.markdown(math_text, unsafe_allow_html=False)


                st.markdown("### 4. Analogies and Intuitive Explanations")
                st.write(result.Analogies_and_Intuitive_Explanations)

                st.markdown("### 5. Critical Perspective")
                st.write(result.Critical_Perspective)

                st.markdown("### 6. Clarity and Accessibility")
                st.write(result.Clarity_and_Accessibility)

                st.markdown("### 7. Importance of the Study")
                st.write(result.Important)

                st.download_button(
                label="Download Summary as PDF",
                data=pdf_buffer,
                file_name=f"{paper_title.replace(' ', '_')}_summary.pdf",
                mime="application/pdf"
)

            except Exception as e:
                sidebar_log("Error occurred during generation")
                st.error(f"Error: {e}")
