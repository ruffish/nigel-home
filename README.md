# Agentic Conversation Agent (Home Assistant)

This repository contains:

- A Home Assistant **add-on** that runs an agent server (LLMs + tools + RAG + memory).
- A Home Assistant **custom integration** that registers a Conversation Entity and forwards Assist queries to the add-on.

## Install

### 1) Add-on

Add this repo as an add-on repository in Home Assistant, then install/start the add-on.

### 2) Custom integration

Copy `custom_components/agentic_conversation_agent` into your Home Assistant `config/custom_components/` folder (or install via HACS as a custom repository).

### 3) Select agent

In Home Assistant: Settings → Voice assistants → Conversation agent, select **Agentic Conversation Agent**.

## Tools

Drop tool definition files into the add-on's `/data/tools` directory (shown as the add-on's data folder). Supported types:

- `ha_service`: call Home Assistant services
- `http`: call external HTTP APIs
- `python`: run a Python script in the add-on container

See `docs/tool_schema.md`.

## Local embeddings + local DB

- **Vector database**: stored locally inside the add-on at `/data/agent.db` (SQLite).
- **Embeddings**:
	- `embeddings.provider: simple` is fully local (deterministic) and requires no external services.
	- For **local AI embeddings**, set `embeddings.provider: ollama` and point `embeddings.base_url` to your local Ollama instance (for the HA Ollama add-on, `http://ollama:11434` is a common default) and choose a model like `nomic-embed-text`.
