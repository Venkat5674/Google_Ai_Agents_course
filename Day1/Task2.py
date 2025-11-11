"""
ADK Multi-Agent Demo

This script reproduces the examples from your notebook:
- API key setup (Kaggle-friendly)
- imports (Agent, SequentialAgent, ParallelAgent, LoopAgent, tools)
- retry options
- Research + Summarizer coordinator agent
- Sequential blog pipeline (outline -> writer -> editor)
- Parallel research team + aggregator
- Loop refinement example (writer -> critic -> refiner -> exit function)

Usage:
- Ensure GOOGLE_API_KEY is available in the environment (or Kaggle Secrets)
- Run with Python 3.11+ in an environment with google-adk & google-genai installed

Example (in a terminal / notebook cell):
    python adk_multi_agent_demo.py

Note: runner.run_debug calls are async — this script runs them sequentially using asyncio.
"""

import os
import asyncio
import subprocess
from typing import Any, Dict

# ADK imports
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

# Optional: Kaggle secret helper (safe if running in Kaggle)
try:
    from kaggle_secrets import UserSecretsClient
except Exception:
    UserSecretsClient = None


# ---------------------------
# 1. Configure API key
# ---------------------------

def configure_api_key_from_kaggle():
    """Try to load GOOGLE_API_KEY from Kaggle secrets (if available) and set ENV var."""
    if UserSecretsClient is None:
        return
    try:
        key = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            print("✅ Loaded GOOGLE_API_KEY from Kaggle secrets.")
    except Exception as e:
        print("(Kaggle) couldn't read secret:", e)


configure_api_key_from_kaggle()

# ---------------------------
# 2. Retry config
# ---------------------------
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Common model factory helper
def gemini_model(name: str = "gemini-2.5-flash-lite") -> Gemini:
    return Gemini(model=name, retry_options=retry_config)


# ---------------------------
# 3. Example: Research + Summarizer Coordinator
# ---------------------------

def build_research_summarizer_coordinator() -> Agent:
    research_agent = Agent(
        name="ResearchAgent",
        model=gemini_model(),
        instruction=(
            "You are a specialized research agent. Your only job is to use the google_search tool "
            "to find 2-3 pieces of relevant information on the given topic and present the findings with citations."
        ),
        tools=[google_search],
        output_key="research_findings",
    )

    summarizer_agent = Agent(
        name="SummarizerAgent",
        model=gemini_model(),
        instruction=(
            "Read the provided research findings: {research_findings}\n"
            "Create a concise summary as a bulleted list with 3-5 key points."
        ),
        output_key="final_summary",
    )

    root_agent = Agent(
        name="ResearchCoordinator",
        model=gemini_model(),
        instruction=(
            "You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.\n"
            "1. First, call the ResearchAgent tool to find relevant information.\n"
            "2. Next, call the SummarizerAgent tool to create a concise summary.\n"
            "3. Present the final summary clearly to the user as your response."
        ),
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )

    return root_agent


# ---------------------------
# 4. Sequential Blog Pipeline
# ---------------------------

def build_sequential_blog_pipeline() -> SequentialAgent:
    outline_agent = Agent(
        name="OutlineAgent",
        model=gemini_model(),
        instruction=(
            "Create a blog outline for the given topic with:\n"
            "1. A catchy headline\n2. An introduction hook\n3. 3-5 main sections with 2-3 bullet points for each\n4. A concluding thought"
        ),
        output_key="blog_outline",
    )

    writer_agent = Agent(
        name="WriterAgent",
        model=gemini_model(),
        instruction=(
            "Following this outline strictly: {blog_outline}\nWrite a brief, 200 to 300-word blog post with an engaging and informative tone."
        ),
        output_key="blog_draft",
    )

    editor_agent = Agent(
        name="EditorAgent",
        model=gemini_model(),
        instruction=(
            "Edit this draft: {blog_draft}\nYour task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity."
        ),
        output_key="final_blog",
    )

    return SequentialAgent(name="BlogPipeline", sub_agents=[outline_agent, writer_agent, editor_agent])


# ---------------------------
# 5. Parallel Research + Aggregator
# ---------------------------

def build_parallel_research_system() -> SequentialAgent:
    tech_researcher = Agent(
        name="TechResearcher",
        model=gemini_model(),
        instruction=(
            "Research the latest AI/ML trends. Include 3 key developments, the main companies involved, and the potential impact. Keep the report very concise (100 words)."
        ),
        tools=[google_search],
        output_key="tech_research",
    )

    health_researcher = Agent(
        name="HealthResearcher",
        model=gemini_model(),
        instruction=(
            "Research recent medical breakthroughs. Include 3 significant advances, their practical applications, and estimated timelines. Keep the report concise (100 words)."
        ),
        tools=[google_search],
        output_key="health_research",
    )

    finance_researcher = Agent(
        name="FinanceResearcher",
        model=gemini_model(),
        instruction=(
            "Research current fintech trends. Include 3 key trends, their market implications, and the future outlook. Keep the report concise (100 words)."
        ),
        tools=[google_search],
        output_key="finance_research",
    )

    aggregator_agent = Agent(
        name="AggregatorAgent",
        model=gemini_model(),
        instruction=(
            "Combine these three research findings into a single executive summary:\n\n"
            "**Technology Trends:**\n{tech_research}\n\n**Health Breakthroughs:**\n{health_research}\n\n**Finance Innovations:**\n{finance_research}\n\n"
            "Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words."
        ),
        output_key="executive_summary",
    )

    parallel_team = ParallelAgent(name="ParallelResearchTeam", sub_agents=[tech_researcher, health_researcher, finance_researcher])

    return SequentialAgent(name="ResearchSystem", sub_agents=[parallel_team, aggregator_agent])


# ---------------------------
# 6. Loop (refinement) example
# ---------------------------

def exit_loop() -> Dict[str, Any]:
    """Signal function to exit the loop. The agent must call this when critique == 'APPROVED'."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


def build_loop_refinement_pipeline(max_iterations: int = 3) -> SequentialAgent:
    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=gemini_model(),
        instruction=(
            "Based on the user's prompt, write the first draft of a short story (around 100-150 words).\n"
            "Output only the story text, with no introduction or explanation."
        ),
        output_key="current_story",
    )

    critic_agent = Agent(
        name="CriticAgent",
        model=gemini_model(),
        instruction=(
            "You are a constructive story critic. Review the story provided below.\n"
            "Story: {current_story}\n\nEvaluate the story's plot, characters, and pacing.\n"
            "- If the story is well-written and complete, YOU MUST respond with the exact phrase: 'APPROVED'\n"
            "- Otherwise, provide 2-3 specific, actionable suggestions for improvement."
        ),
        output_key="critique",
    )

    refiner_agent = Agent(
        name="RefinerAgent",
        model=gemini_model(),
        instruction=(
            "You are a story refiner. You have a story draft and critique.\n\n"
            "Story Draft: {current_story}\nCritique: {critique}\n\n"
            "Your task is to analyze the critique.\n"
            "- IF the critique is EXACTLY 'APPROVED', you MUST call the exit_loop function and nothing else.\n"
            "- OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique."
        ),
        tools=[FunctionTool(exit_loop)],
        output_key="current_story",  # Overwrite story on each refinement
    )

    loop_agent = LoopAgent(name="StoryRefinementLoop", sub_agents=[critic_agent, refiner_agent], max_iterations=max_iterations)

    return SequentialAgent(name="StoryPipeline", sub_agents=[initial_writer_agent, loop_agent])


# ---------------------------
# 7. Runner helpers
# ---------------------------

async def run_agent_and_print(agent: Agent, prompt: str):
    runner = InMemoryRunner(agent=agent)
    print(f"\n--- Running agent: {agent.name} ---")
    response = await runner.run_debug(prompt)
    # The run_debug call prints traces to stdout; response object may contain full data structure.
    print("--- run_debug returned: ---")
    print(response)
    print(f"--- Finished agent: {agent.name} ---\n")


async def main():
    # Example 1: Research Coordinator
    coordinator = build_research_summarizer_coordinator()
    await run_agent_and_print(coordinator, "What are the latest advancements in quantum computing and what do they mean for AI?")

    # Example 2: Sequential blog pipeline
    blog_pipeline = build_sequential_blog_pipeline()
    await run_agent_and_print(blog_pipeline, "Write a blog post about the benefits of multi-agent systems for software developers")

    # Example 3: Parallel research + aggregator
    research_system = build_parallel_research_system()
    await run_agent_and_print(research_system, "Run the daily executive briefing on Tech, Health, and Finance")

    # Example 4: Loop refinement
    loop_pipeline = build_loop_refinement_pipeline(max_iterations=2)
    await run_agent_and_print(loop_pipeline, "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as exc:
        print("Error running demo:", exc)
