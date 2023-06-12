#======================#
# Install, clean, test #
#======================#

install_requirements:
	@pip install -r requirements_dev.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info

test_structure:
	@bash tests/test_structure.sh

#======================#
#          API         #
#======================#

run_api:
	uvicorn leafscan.fast:app --reload --port 8000


#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project ${PROJECT_ID}



#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest .

# Alternative if previous doesn´t work. Needs additional setup.
# Probably don´t need this. Used to build arm on linux amd64
docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest \
		bash

# Push and deploy to cloud

docker_push:
	docker push $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest

docker_deploy:
	gcloud run deploy \
		--project $(GCP_PROJECT_ID) \
		--image $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME):latest \
		--platform managed \
		--memory=2Gi \
		--region europe-west1 \
		--env-vars-file .env.yaml

run_operationnal:
	@cd leafscan
	python -c 'from leafscan.main import operationnal; operationnal()'
