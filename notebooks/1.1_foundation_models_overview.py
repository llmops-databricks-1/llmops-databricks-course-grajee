# Databricks notebook source
"""
Lecture 1.1: Foundation Models Overview
Topics: Foundation models on Databricks, hosted models, external models, pricing comparison.
"""

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI

# Initialize workspace client
w = WorkspaceClient()

# COMMAND ----------

# List available foundation model endpoints
endpoints = w.serving_endpoints.list()

logger.info("Available Foundation Model Endpoints:")
logger.info("-" * 80)
for endpoint in endpoints:
    if endpoint.name and "databricks" in endpoint.name:
        logger.info(f"Name: {endpoint.name}")
        logger.info(f"State: {endpoint.state}")
        logger.info("-" * 80)

# COMMAND ----------

# Call Meta Llama via Databricks Foundation Model API
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)
model_name = "databricks-llama-4-maverick"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain LLMOps in 3 sentences."}
    ],
    max_tokens=200,
    temperature=0.7
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")
logger.info(f"Input tokens: {response.usage.prompt_tokens}")
logger.info(f"Output tokens: {response.usage.completion_tokens}")

# COMMAND ----------

# Cost calculation helpers


def calculate_api_cost(input_tokens: int, output_tokens: int,
                       input_dbu_per_1m: float, output_dbu_per_1m: float) -> float:
    """Calculate DBU cost for pay-per-token API."""
    input_cost = (input_tokens / 1_000_000) * input_dbu_per_1m
    output_cost = (output_tokens / 1_000_000) * output_dbu_per_1m
    return input_cost + output_cost


def calculate_provisioned_cost(hours: int, dbu_per_hour: float) -> float:
    """Calculate DBU cost for provisioned throughput."""
    return hours * dbu_per_hour


# Example: 1M input tokens, 500K output tokens with Llama 3.3 70B
api_cost = calculate_api_cost(1_000_000, 500_000, 7.143, 21.429)
logger.info(f"Pay-per-token cost: {api_cost:.2f} DBUs")

# Example: 24 hours with Llama 3.2 1B provisioned (entry capacity)
provisioned_cost = calculate_provisioned_cost(24, 42.857)
logger.info(f"Provisioned throughput cost (24h): {provisioned_cost:.2f} DBUs")

# Break-even analysis
input_tokens = 6_000_000
output_tokens = 4_000_000
api_cost_equivalent = calculate_api_cost(input_tokens, output_tokens, 7.143, 21.429)
logger.info(f"For {input_tokens + output_tokens:,} tokens in 24h:")
logger.info(f"  API cost (Llama 3.3 70B): {api_cost_equivalent:.2f} DBUs")
logger.info(f"  Provisioned cost (Llama 3.2 1B): {provisioned_cost:.2f} DBUs")
logger.info(f"  Difference: {api_cost_equivalent - provisioned_cost:.2f} DBUs")
