# STAMP MCP Server

A FastMCP-based Model Context Protocol server wrapping [STAMP](https://github.com/KatherLab/STAMP)'s tools, enabling seamless integration of STAMP preprocessing, training, encoding, evaluation, and inference into LLM-based pipelines.

## Overview

This server lets LLM agents invoke STAMP tools via structured calls. It exposes the following tools:

- `preprocess_stamp()`: tile & extract WSI features  
- `train_stamp()`: train weakly supervised models  
- `crossval_stamp()`: k-fold cross‑validation  
- `deploy_stamp()`: inference on held‑out data
- `encode_slides_stamp()`: slide-level feature encoding  
- `encode_patients_stamp()`: patient-level feature encoding
- `heatmaps_stamp()`: model-based heatmap visualization  
- `statistics_stamp()`: compute classification metrics  
- `read_file()` & `list_files()`: safe disk access  
- `check_available_devices()`: query Torch/Platform device availability
- `analyze_csv()` & `list_column_values`: useful for clinical and slide tables

Each tool serializes config into YAML and directly calls STAMP's internal `_run_cli()` function, streaming logs back in real-time and returning execution results.

## Installation
To run the MCP server is as simple as intalling STAMP as it is explained in the main README.md file, but adding `--extra mcp` to the command. For a GPU repository installation it would be like this:
```bash
uv sync --extra build --extra gpu --extra mcp
```

## Example using OpenAI Agents SDK

This example demonstrates how to connect an STAMP MCP server into an agent workflow using the official OpenAI Agents SDK v0.1.0

Start STAMP MCP server with:

```
python server.py
```

The server should start successfully. After that, create a separate workspace with a new virtual environment and install the following:

```bash
uv pip install openai-agents>=0.1.0
```

The following file defines an agent that extract features from given Whole Slide Images:

```python
# stamp_agent.py
import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, enable_verbose_stdout_logging
from agents.mcp import MCPServerStreamableHttp
from datetime import timedelta

enable_verbose_stdout_logging()

model = OpenAIChatCompletionsModel(model="gpt-o3",)

async def preprocess_workflow(mcp_server: MCPServerStreamableHttp):
    agent = Agent(
        name="Oncology researcher",
        instructions=(
            "You are a bioinformatic researcher assigned to extract features from Whole Slide Images"
        ),
        model=model,
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )
    message = "Step 1: Check available devices via `check_available_devices`. " \
            "Step 2: Run `preprocess_stamp` on slides in `agent_workspace/slides`, " \
            "extracting features with extractor='ctranspath', " \
            "no cache, output to `agent_workspace/feats`, " \
            "using the best device. " \
            "Step 3: If preprocessing fails, adjust and retry once. " \
            "Finally, summarize which device was used and whether it succeeded."
    result = await Runner.run(starting_agent=agent, input=message)
    print("Agent finished:", result.final_output)

async def main():
    async with MCPServerStreamableHttp(
        name="STAMP MCP",
        params={
            "url": "http://127.0.0.1:8000/mcp",
            "timeout": timedelta(minutes=10),
            "sse_read_timeout": timedelta(hours=4),
        },
        client_session_timeout_seconds=6000,
    ) as server:
        await preprocess_workflow(server)

if __name__ == "__main__":
    asyncio.run(main())
```

> :warning: Note: Some STAMP tools (e.g., preprocessing, cross-validation) may run for a long time—please adjust both timeout and sse_read_timeout to avoid connection drops.



On a different console from the one you are running the MCP server, run your agent:

```
python stamp_agent.py
```
