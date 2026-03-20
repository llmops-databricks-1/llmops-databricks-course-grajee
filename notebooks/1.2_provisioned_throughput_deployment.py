# Databricks notebook source
"""
Lecture 1.2: Provisioned Throughput Deployment
Deploy a dedicated LLaMA endpoint with provisioned throughput, monitor it, and clean up.
"""

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from loguru import logger
from openai import OpenAI

w = WorkspaceClient()

# COMMAND ----------

# Configuration
ENDPOINT_NAME = "llama-3-2-1b-provisioned-rgopinat"
MODEL_NAME = "system.ai.llama_v3_2_1b_instruct"
WORKLOAD_SIZE = "Small"
SCALE_TO_ZERO = True
MIN_PROVISIONED_THROUGHPUT = 0
MAX_PROVISIONED_THROUGHPUT = 20

catalog = "mlops_dev"
schema = "rgopinat"
BUDGET_POLICY_ID = None

# COMMAND ----------

# Check if endpoint already exists


def endpoint_exists(endpoint_name: str) -> bool:
    """Check if serving endpoint exists."""
    try:
        w.serving_endpoints.get(endpoint_name)
        return True
    except Exception:
        return False


if endpoint_exists(ENDPOINT_NAME):
    logger.info(f"Endpoint '{ENDPOINT_NAME}' already exists")
else:
    logger.info(f"Endpoint '{ENDPOINT_NAME}' does not exist. Ready to create.")

# COMMAND ----------

# Create endpoint with provisioned throughput
ai_gateway_cfg = AiGatewayConfig(
    inference_table_config=AiGatewayInferenceTableConfig(
        enabled=True,
        catalog_name=catalog,
        schema_name=schema,
        table_name_prefix="provisioned_throughput_monitoring",
    ),
    usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
)

endpoint_config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version="1",
            workload_size=WORKLOAD_SIZE,
            scale_to_zero_enabled=SCALE_TO_ZERO,
            min_provisioned_throughput=MIN_PROVISIONED_THROUGHPUT,
            max_provisioned_throughput=MAX_PROVISIONED_THROUGHPUT,
        )
    ],
)

if not endpoint_exists(ENDPOINT_NAME):
    logger.info(f"Creating endpoint '{ENDPOINT_NAME}'...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=endpoint_config,
        ai_gateway=ai_gateway_cfg,
        budget_policy_id=BUDGET_POLICY_ID,
    )

# COMMAND ----------

# Monitor endpoint deployment


def wait_for_endpoint(endpoint_name: str, timeout_minutes: int = 30) -> bool:
    """Wait for endpoint to be ready."""
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while True:
        try:
            endpoint = w.serving_endpoints.get(endpoint_name)
            config_state = endpoint.state.config_update
            ready_state = endpoint.state.ready

            logger.info(f"Status: config_update={config_state}, ready={ready_state}")

            if config_state.value == "NOT_UPDATING" and ready_state.value == "READY":
                logger.info(f"Endpoint '{endpoint_name}' is ready!")
                return True

            if config_state.value == "UPDATE_FAILED":
                logger.error("Endpoint creation failed!")
                return False

            if time.time() - start_time > timeout_seconds:
                logger.warning("Timeout waiting for endpoint")
                return False

            time.sleep(30)

        except Exception as e:
            logger.error(f"Error checking endpoint: {e}")
            return False


wait_for_endpoint(ENDPOINT_NAME)

# COMMAND ----------

# Call the provisioned endpoint
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints"
)

response = client.chat.completions.create(
    model=ENDPOINT_NAME,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the benefits of provisioned throughput for LLMs."}
    ],
    max_tokens=500,
    temperature=0.7
)

logger.info("Response:")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")

# COMMAND ----------

# Get endpoint metrics


def get_endpoint_metrics(endpoint_name: str) -> None:
    """Get endpoint metrics and status."""
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"State: {endpoint.state.config_update}")
        for entity in endpoint.config.served_entities:
            logger.info(f"  Model: {entity.entity_name}")
            logger.info(f"  Workload Size: {entity.workload_size}")
            logger.info(f"  Min Throughput: {entity.min_provisioned_throughput} model units")
            logger.info(f"  Max Throughput: {entity.max_provisioned_throughput} model units")
            logger.info(f"  Scale to Zero: {entity.scale_to_zero_enabled}")
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")


get_endpoint_metrics(ENDPOINT_NAME)

# COMMAND ----------

# Cleanup — delete endpoint when done to avoid charges


def delete_endpoint(endpoint_name: str) -> None:
    """Delete a serving endpoint."""
    try:
        w.serving_endpoints.delete(endpoint_name)
        logger.info(f"Endpoint '{endpoint_name}' deleted successfully")
    except Exception as e:
        logger.error(f"Error deleting endpoint: {e}")


delete_endpoint(ENDPOINT_NAME)
