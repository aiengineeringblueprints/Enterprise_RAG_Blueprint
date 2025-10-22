# Deploy this app on your server
The GitHub actions pipeline ```ci-cd.yml``` is an implemented pipeline, containing the following steps:
1. Unit Test: Runs pytests for chain and loader services
2. Build and Test: Builds the docker containers and starts and tests them one by one
3. Deploy: Deploys the containers on different remote servers for the stages ```dev```, ```staging``` and ```production``` via SCP and loads and starts the container via SSH

Note: 
- A ```docker-compose.yml``` and an ```.env``` for the deployment on the servers in the deployment path is needed
- The following secrets need to be defined in the GitHub repo 
    - ```SSH_PRIVATE_KEY```: Private SSH Key
    - ```DEV_USER```, ```PROD_USER```, ```PROD_USER```: Users to connect to the host 
    - ```DEV_HOST```, ```STAGING_HOST```, ```PROD_HOST```: IP address or hostname of your servers
    - ```DEV_DEPLOY_PATH```,```STAGING_DEPLOY_PATH```, ```PROD_DEPLOY_PATH```: Paths where the containers are deployed and the docker compose is ran
    
## Stages
| Stage | Branch | Trigger | Deploy to Server |
|-------|--------|---------|-------------------|
| **Development** | `dev` | Push to `dev` | Automatically |
| **Staging** | `main` | Push to `main` | Automatically |
| **Production** | `main` | Push main `main` | After manual approval |

