# Databricks notebook source
"""
Lecture 1.4: External Models with Custom Provider
Create an OpenAI DALL-E endpoint via Databricks External Models and generate images.

Prerequisites:
  - OpenAI API key stored in Databricks Secrets:
      scope: llmops_course
      key:   openai_key
"""

import base64
from io import BytesIO

import mlflow.deployments
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from PIL import Image

# COMMAND ----------

# Create the external model endpoint

client = mlflow.deployments.get_deploy_client("databricks")

ENDPOINT_NAME = "openai-dalle-custom-rgopinat"

try:
    existing = client.get_endpoint(ENDPOINT_NAME)
    logger.info(f"Endpoint '{ENDPOINT_NAME}' already exists")
except Exception:
    logger.info(f"Creating External Model endpoint: {ENDPOINT_NAME}")
    endpoint = client.create_endpoint(
        name=ENDPOINT_NAME,
        config={
            "served_entities": [{
                "name": "dalle-image-generation",
                "external_model": {
                    "name": "dall-e-3",
                    "provider": "openai",
                    "task": "llm/v1/images",
                    "openai_config": {
                        "openai_api_key": "{{secrets/rgopinat_secrets/openai_key}}",
                        "openai_api_base": "https://api.openai.com/v1",
                        "openai_api_type": "openai"
                    }
                }
            }]
        }
    )
    logger.info(f"Endpoint created: {ENDPOINT_NAME}")

# COMMAND ----------

# Call the endpoint to generate an image

w = WorkspaceClient()
host = w.config.host
token = w.tokens.create(lifetime_seconds=1200).token_value

openai_client = OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)

response = openai_client.images.generate(
    model=ENDPOINT_NAME,
    prompt="Two cats wearing superhero capes in a sunny garden",
    n=1,
    style="vivid",
    quality="standard",
    response_format="b64_json"
)

logger.info("Image generated successfully!")

# COMMAND ----------

# Display the generated image

image_data = response.data[0].b64_json
image_bytes = base64.b64decode(image_data)
image = Image.open(BytesIO(image_bytes))

display(image)

logger.info(f"Image size: {image.size}")
