# Enterprise RAG Blueprint
Presenting an AI engineering blueprint for on-premises enterprise RAG solutions. The blueprint includes an end-to-end reference architecture based on the 4+1 architectural view model, a reference application, and best practices for tooling, development, and deployment pipelines.

## Associated paper:
AI-Engineering Blueprint for Scalable On-Premises Retrieval-Augmented Generation Systems

## Quick Start
To try the RAG Reference Application, fork and clone the repository 

Rename the ```example.env``` to ```.env``` and add the Base URL, API Key and Model name for your LLM. Note: The call to the LLM is executed using an OpenAI-compatible API (https://platform.openai.com/docs/api-reference/chat/create) which calls ```[baseurl]/chat/completions```.

then run 
```
docker compose up --build
```

All containers will be built or downloaded and started on your local machine. 

The first start will take some time due to the building of the containers and pulling the embedding model for the ollama container.

The Frontend is accessable via 
```
http://0.0.0.0:8501
```

At the first login, an admin user needs to be created. You can then create more users or just login with the admin user and password.
Once logged in you can try the App by uploading documents in the upload documents tab and chat with these documents in the chat tab.

Note: Since this is a development example, the reasoning of the model is shown in the output. This can be changed ...

## Run services or tests locally
To run the services or the pytests locally, the dependencies from ```loader/requirements.txt``` and ```chain/requirements``` (for running the frontend locally also ```frontend/requirements.txt```) need to be installed as well as the ```./requirements-test.txt```  

## Deployment
Deployment is configured using GitHub Actions and a CI/CD pipeline. You can find this pipeline and the documentation in ./github/workflows and in the repo at GitHub Actions.

## Open Topics
- Automated naming of chats 
