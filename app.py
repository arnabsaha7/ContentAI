import asyncio
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Content Researcher & Writer", page_icon="🤖", layout="wide")

# Title
st.title("Smart Content Generator: AI-Powered Research & Writing")
st.markdown("Effortlessly Generate Engaging, Well-Researched Blog Posts Using Advanced AI Agents")

# Sidebar
with st.sidebar:
    st.header("Content Settings")
    topic = st.text_area("Enter your topic", height=100, placeholder="Enter the topic")
    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    st.markdown("---")
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)
    with st.expander("ℹ️ How to use"):
        st.markdown("""
            1. Enter your desired content topic
            2. Play with the temperature to see variations in output
            3. Click 'Generate Content' to start
            4. Wait for the AI to generate your article
            5. Download the result as a markdown file
    """)

def create_agents(llm):
    search_tool = SerperDevTool(n=10)

    senior_researcher = Agent(
        role = "Senior Researcher",
        goal = f"Research, analyze & summarize comprehensive information on {topic} from reliable web sources",
        backstory = """
            You're an expert research analyst with advanced web research skills. \
            You excel at finding, analyzing, and synthesizing information from \
            across the internet using search tools. You're skilled at distinguishing \
            reliable sources from unreliable ones, fact-checking, \
            cross-referencing information, and identifying key patterns and insights. \
            You provide well-organized research briefs with proper citations and source verification. \
            Your analysis includes both raw data and interpreted insights, making complex \
            information accessible and actionable.
        """,
        allow_delegation= False,
        verbose = False,
        tools = [search_tool],
        llm = llm
    )
    content_writer = Agent(
        role = "Content Writer",
        goal = "Transform research findings into engaging blog posts while maintaining accuracy.",
        backstory = """
            You're a skilled content writer with a talent for turning complex research \
            into accessible and engaging blog posts. You excel in understanding the \
            core message of research findings and crafting compelling narratives that \
            captivate readers. Your expertise lies in maintaining the balance between \
            accuracy and readability, ensuring that the content is both informative \
            and easy to understand. You have a deep understanding of SEO best practices, \
            enabling you to optimize content for search engines without compromising \
            on quality. You work closely with researchers and editors to ensure that \
            every piece is factually accurate and well-structured, catering to the \
            intended audience's needs and interests.
        """,
        allow_delegation = False,
        verbose = False,
        llm = llm
    )
    return senior_researcher, content_writer

async def generate_content(topic):
    llm = LLM(model="gpt-4")
    senior_researcher, content_writer = create_agents(llm)

    research_task = Task(
        description = ("""
                1. Conduct comprehensive research on {topic} including:
                    - Recent developments and news
                    - Key industry trends and innovations
                    - Expert opinions and analysis
                    - Statistical data and market insights
                2. Evaluate source credibility and fact-check all information
                3. Organize findings into a structured research brief
                4. Include all relevant citations and sources
            """),
        expected_output = """
                A detailed research report containing:
                    - Executive summary of key findings
                    - Comprehensive analysis of current trends and developments
                    - List of verified facts and statistics
                    - All citations and links to original sources
                    - Clear categorization of main themes and patterns
                Format with clear sections and bullet points for easy references.
            """,
        agent = senior_researcher
    )
    writing_task = Task(
        description = (""" 
                Using the research brief provided, create an engaging blog post that:
                    1. Transforms technical information into accessible content
                    2. Maintains all factual accuracy and citations from the research
                    3. Includes: 
                        - Attention-grabbing introduction
                        - Well-structured body section with clear headings
                        - Compelling conclusion 
                    4. Preserves all sourcce citations in [Source: URL] format
                    5. Includes a Reference section at the end.
            """
        ),
        expected_output = """
                A polished blog post in markdown format that:
                    - Engages readers while maintaininf accuracy
                    - Contains properly structured sections
                    - Includes Incline citations hyperlinked to the original source url
                    - Presents information in an accessible yet informative way
                    - Follows proper mardown formatting, use H1 for the title and H3 for sub-sections
            """,
        agent = content_writer
    )
    
    crew = Crew(agents=[senior_researcher, content_writer], tasks=[research_task, writing_task], verbose=False)
    result = await crew.kickoff_async(inputs={"topic": topic})
    return result

if generate_button:
    with st.spinner('Generating content... This may take a while!'):
        try:
            result = asyncio.run(generate_content(topic))
            st.markdown("### Generated Content")
            st.markdown(result)
            st.download_button(
                label="Download Content",
                data=result.raw,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error has occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("POWERED BY CREWAI")
