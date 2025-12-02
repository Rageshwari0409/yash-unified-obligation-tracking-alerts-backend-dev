#!/usr/bin/env bash
set -e

CONFIG_FILE=$1
ENV=$2
OUTPUT_DIR="infra-repo/terraform/lambda"

echo "📄 Using environment: $ENV"
echo "📄 Reading and modifying config file: $CONFIG_FILE"

# -------------------------------
# Replace __ENV__ placeholders IN-PLACE
# -------------------------------
sed -i "s/__env__/${ENV}/g" "$CONFIG_FILE"

# -------------------------------
# Read values directly from modified file
# -------------------------------
REGION=$(yq e '.aws.region' "$CONFIG_FILE")
ACCOUNT=$(yq e '.aws.account_id' "$CONFIG_FILE")
FUNCTION=$(yq e '.lambda.function_name' "$CONFIG_FILE")
TIMEOUT=$(yq e '.lambda.lambda_timeout' "$CONFIG_FILE")
MEMORY=$(yq e '.lambda.memory_size' "$CONFIG_FILE")
IMAGE_TAG=$(yq e '.lambda.image_tag' "$CONFIG_FILE")
ZONE_NAME=$(yq e '.lambda.zone_name' "$CONFIG_FILE")
PRIVATE_ALB_NAME=$(yq e '.lambda.private_alb_name' "$CONFIG_FILE")
ECR=$(yq e '.lambda.ecr_name' "$CONFIG_FILE")
EPHEMERAL_STORAGE=$(yq e '.lambda.ephemeral_storage' "$CONFIG_FILE")

# -------------------------------
# Backend config
# -------------------------------
cat <<EOF > "$OUTPUT_DIR/backend.tfbackend"
bucket         = "yash-unified-${ENV}-lambda-tfstate-bucket"
key            = "${FUNCTION}.tfstate"
region         = "${REGION}"
dynamodb_table = "yash-unified-${ENV}-lambda-table"
encrypt        = true
EOF

echo "✅ Generated backend.tfbackend"

# -------------------------------
# Extract env vars, secrets, tags
# -------------------------------
ENV_VARS_BLOCK=$(yq e '.lambda.environment_variables' "$CONFIG_FILE" | yq e -o=json)
SECRETS_LIST=$(yq e '.lambda.secrets' "$CONFIG_FILE" | yq e -o=json)
TAGS_BLOCK=$(yq e '.lambda.tags' "$CONFIG_FILE" | yq e -o=json)

[ "$TAGS_BLOCK" == "null" ] && TAGS_BLOCK="{}"

# -------------------------------
# Generate terraform.auto.tfvars
# -------------------------------
cat <<EOF > "$OUTPUT_DIR/terraform.auto.tfvars"
aws_region           = "${REGION}"
aws_account_id       = "${ACCOUNT}"
env_name             = "${ENV}"
function_name        = "${FUNCTION}"
lambda_timeout       = ${TIMEOUT}
memory_size          = ${MEMORY}
image_tag            = "${IMAGE_TAG}"
zone_name            = "${ZONE_NAME}"
private_alb_name     = "${PRIVATE_ALB_NAME}"
ecr_repository_name  = "${ECR}"
ephemeral_storage    = ${EPHEMERAL_STORAGE}

# Dynamic env vars and secrets
environment_variables = ${ENV_VARS_BLOCK}
secrets               = ${SECRETS_LIST}
tags                  = ${TAGS_BLOCK}
EOF

echo "✅ Generated terraform.auto.tfvars in $OUTPUT_DIR"
 