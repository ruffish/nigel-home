DOMAIN = "agentic_conversation_agent"

# Config keys
CONF_LLM_PROVIDER = "llm_provider"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_BASE_URL = "llm_base_url"

# Agentic settings
CONF_EMBEDDINGS_PROVIDER = "embeddings_provider"
CONF_EMBEDDINGS_MODEL = "embeddings_model"
CONF_EMBEDDINGS_API_KEY = "embeddings_api_key"
CONF_EMBEDDINGS_BASE_URL = "embeddings_base_url"

CONF_TOOLS_DIR = "tools_dir"
CONF_ALLOW_PYTHON = "allow_python"
CONF_AGENT_MAX_STEPS = "agent_max_steps"
CONF_TOOL_TOP_K = "tool_top_k"
CONF_MEMORY_TOP_K = "memory_top_k"
CONF_MEMORY_EXPIRES_DAYS = "memory_expires_days"
CONF_BUFFER_MAX_TURNS = "buffer_max_turns"

# Defaults
DEFAULT_LLM_PROVIDER = "google"
DEFAULT_LLM_MODEL = "gemini-1.5-flash"
DEFAULT_TIMEOUT = 60

DEFAULT_EMBEDDINGS_PROVIDER = "simple"
DEFAULT_EMBEDDINGS_MODEL = ""
DEFAULT_TOOLS_DIR = ""
DEFAULT_ALLOW_PYTHON = False
DEFAULT_AGENT_MAX_STEPS = 6
DEFAULT_TOOL_TOP_K = 8
DEFAULT_MEMORY_TOP_K = 5
DEFAULT_MEMORY_EXPIRES_DAYS = 30
DEFAULT_BUFFER_MAX_TURNS = 12

# Provider choices
LLM_PROVIDERS = ["google", "openai", "openai_compatible"]

EMBEDDINGS_PROVIDERS = ["simple", "ollama", "openai", "openai_compatible", "google"]
