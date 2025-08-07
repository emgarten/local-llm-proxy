# Prompt Passage

A local proxy for LLMs, providing a unified interface for multiple models and support for identity based authentication.

## Getting started

First create your ~/.prompt-passage.yaml file and configure your providers:

### Example config

```yaml
service:
  port: 8095
  auth:
    type: apikey
    key: localkey
providers:
  azure-o4-mini-env:
    endpoint: "https://{service}.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
    model: o4-mini
    auth:
      type: apikey
      envKey: AZURE_OPENAI_API_KEY
    transform: ".messages as $m | .inputs=$m | del(.messages)"
  azure-o4-mini-key:
    endpoint: "https://{service}.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
    model: o4-mini
    auth:
      type: apikey
      key: djjskskskkkk
  azure-o4-mini-azure:
    endpoint: "https://{service}.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview"
    model: o4-mini
    auth:
      type: azure
```

The optional `transform` field contains a [jq](https://stedolan.github.io/jq/) expression that
modifies the JSON body of outgoing requests. The transform runs only when the incoming body
is valid JSON. In the example above, the `messages` field is renamed to `inputs` while the
rest of the body is left unchanged.

### Running prompt-passage

Run prompt-passage to start the local proxy

```bash
# Run az login if using azure credentials

# Run prompt passage
pipx run prompt-passage
```

### Connecting

Use `OpenAI compatible`, `Azure OpenAI`, or similar option from the tool you are trying to connect with.

Base url: `http://localhost:8095/providers/{your provider name}/`

API token can be any value unless auth is enabled.


## Dev environment setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install packages
make install

# Lint and type check
make check
```
## Docker

Build the container image:

```bash
docker build -t prompt-passage .
```

When using the `azure` authentication method, mount your Azure CLI credentials directory:

```bash
docker run -p 8095:8095 -v ~/.prompt-passage.yaml:/etc/prompt-passage.yaml -v ~/.azure:/root/.azure -e AZURE_OPENAI_API_KEY prompt-passage
```

Docker compose

```yaml
services:
  prompt-passage:
    image: prompt-passage
    ports:
      - "8095:8095"
    volumes:
      - ~/.prompt-passage.yaml:/etc/prompt-passage.yaml  # mount config file
      - ~/.azure:/root/.azure # mount azure cli credentials if needed
    environment:
      - AZURE_OPENAI_API_KEY  # include any env vars used in the config
```

## HTTPS

To serve the API over HTTPS, set `PROMPT_PASSAGE_CERTFILE` and optionally
define `PROMPT_PASSAGE_KEYFILE` and `PROMPT_PASSAGE_CA_CERTS` for the
private key and CA bundle:

Example:

```bash
PROMPT_PASSAGE_CERTFILE=/path/server.crt \
PROMPT_PASSAGE_KEYFILE=/path/server.key \
PROMPT_PASSAGE_CA_CERTS=/path/ca.pem \
python -m prompt_passage.cli
```

