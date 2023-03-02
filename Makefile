# Names of the code directory and the Docker image, change them to match your project
DOCKER_IMAGE_NAME := music-ner-eacl2023
DOCKER_CONTAINER_NAME := music-ner-eacl2023
CODE_DIRECTORY := music-ner-eacl2023

DOCKER_PARAMS=  -it --rm --name=$(DOCKER_CONTAINER_NAME) 
# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
DOCKER_PARAMS+= --gpus '"device=0,1"'
# Run Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) -v $(PWD):/workspace $(DOCKER_IMAGE_NAME)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run 		Run the Docker image in a container, after building it"
	@echo "	run-bash	Same as 'run', and launches an interactive bash session in the container while mounting the current directory"
	@echo "	stop		Stop the container if it is running"
	@echo "	poetry		Use poetry to modify 'pyproject.toml' and 'poetry.lock' files (e.g. 'make poetry add requests' to add the 'requests' package)"
	@echo "	check		Check coding conventions using multiple tools"
	@echo "	clean		Format your code using black and isort to fit coding conventions"


build:
	docker build -t $(DOCKER_IMAGE_NAME) .

run: build stop
	docker run $(DOCKER_PARAMS) $(DOCKER_IMAGE_NAME)

run-bash:
	$(DOCKER_RUN_MOUNT) /bin/bash

stop:
	docker stop $(DOCKER_IMAGE_NAME) || true && docker rm $(DOCKER_IMAGE_NAME) || true

poetry:
	$(DOCKER_RUN_MOUNT) poetry $(filter-out $@,$(MAKECMDGOALS))
%:	# Avoid printing anything after executing the 'poetry' target
	@:

check:
	$(DOCKER_RUN_MOUNT) poetry run pyflakes $(CODE_DIRECTORY) || exit 1
	$(DOCKER_RUN_MOUNT) poetry run black --check $(CODE_DIRECTORY) || exit 1
	$(DOCKER_RUN_MOUNT) poetry run isort --check $(CODE_DIRECTORY) || exit 1
	@echo "\nAll is good !\n"

clean:
	$(DOCKER_RUN_MOUNT) poetry run black $(CODE_DIRECTORY)
	$(DOCKER_RUN_MOUNT) poetry run isort $(CODE_DIRECTORY)