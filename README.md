# PropertyLM UK

Local UK property-address matching using embeddings, deterministic address
identity checks, cross-encoder reranking and Qdrant.

## Runtime requirements

- Docker Desktop or Docker Engine
- Git
- A compatible `uk_properties` Qdrant snapshot supplied separately

Python is not required on the host for runtime use.

## Start locally

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/propertylm-uk.git
cd propertylm-uk
cp .env.example .env
./scripts/start_runtime.sh
