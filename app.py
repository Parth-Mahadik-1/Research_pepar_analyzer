import streamlit as st 
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from pydantic import BaseModel , Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from io import BytesIO
from reportlab.lib.styles import ParagraphStyle

from dotenv import load_dotenv


# ------------------ Setup ------------------

load_dotenv()


st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Research Paper Summarizer (LLM)")
st.write("Summarize academic papers with technical depth and clarity.")

# ------------------ LLM ------------------


llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task = "text-generation",
    temperature = 0.7,
)

model = ChatHuggingFace(llm=llm)

class PaperSummary(BaseModel):
    Overview: str = Field(
        description="High-level objective, problem statement, and approach of the paper"
    )
    Key_insights: str = Field(
        description="Main findings, results, or contributions of the paper"
        "Write mathematical formulas using LaTeX blocks ($$ ... $$). "
        "Avoid inline equations."
    )
    
    Basic_Info: str = Field(
        description="Paper domain, methodology type, and why it matters"
    )


parser = PydanticOutputParser(pydantic_object=PaperSummary)

prompt = PromptTemplate(

    template="""
You are an expert academic writer and technical communicator.

Create a concise, technically accurate summary of the research paper titled:
"{paper_input}"

Explanation Style: {style_input}
Target Length: {length_input}

Your response MUST strictly follow the JSON format defined below.
Do NOT add extra keys.
Do NOT omit any required key.

{format_instructions}

Guidelines:

1. Overview
- Clearly state the paper’s main objective and the core problem it addresses.
- Briefly describe the overall methodology or architectural approach.
- Keep this section concise and limited to approximately 4 lines.
- Avoid detailed equations or excessive technical depth here.

2. Key_insights
- Present the most important technical contributions and findings.
- Explicitly include the central mathematical formulations or functions introduced in the paper.
- "Write mathematical formulas using LaTeX blocks ($$ ... $$). "
- "Avoid inline equations."

3. Basic_Info
- Specify the research domain (e.g., NLP, Machine Learning, Sequence Modeling).
- Indicate whether the work is theoretical, experimental, or applied.
- Briefly explain the broader impact or long-term relevance of the work.

Important:
If the paper does not provide required information, explicitly write:
"Insufficient information available"

Do NOT guess, invent, or add unsupported claims.
""",
    
input_variables=["paper_input", "style_input", "length_input"],
partial_variables={"format_instructions": parser.get_format_instructions()}

)

chain = prompt | model | parser

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
    add_section("Basic Information", summary.Basic_Info)

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
    log_box.empty()

    if not paper_title.strip():
        sidebar_log("Waiting for paper title input")
        st.warning("Please enter a research paper title.")
    else:
        sidebar_log("Paper title received")
        sidebar_log("Invoking LLM")

        with st.spinner("Generating summary..."):
            try:
                result = chain.invoke({
                    "paper_input": paper_title,
                    "style_input": style,
                    "length_input": length
                })

                sidebar_log("Summary generated successfully")
                st.success("Summary Generated!")

                pdf_buffer = generate_pdf(result, paper_title)

                # -------- Display Output --------
                st.markdown("### 1. Overview")
                st.write(result.Overview)

                st.markdown("### 2.Key_insights")
                format_math(result.Key_insights)

                st.markdown("### 3. Basic Information")
                st.write(result.Basic_Info)

                st.download_button(
                    label="Download Summary as PDF",
                    data=pdf_buffer,
                    file_name=f"{paper_title.replace(' ', '_')}_summary.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                sidebar_log("Error during generation")
                st.error(f"Error: {e}")

