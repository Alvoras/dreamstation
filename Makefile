default: help

## help : This help
help: Makefile
		@printf "\n Dreamstation\n\n"
		@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'
		@printf ""

## install_full : [INSTALL] Download dependencies, models, vendor packages, then create virtualenv and install python packages
install_full: deps model vendor venv install_pip

## install : [INSTALL] Create virtualenv and install python packages
install: venv install_pip

## deps : [INSTALL] Download models, vendor packages, create virtualenv and install python packages
deps:
	sudo apt install exempi python3-dev python3-virtualenv build-essential

## vendor : [INSTALL] Download required resources
vendor:
	git clone https://github.com/openai/CLIP lib/vendor/CLIP
	git clone https://github.com/CompVis/taming-transformers.git lib/vendor/taming-transformers

## model : [INSTALL] Download sflckr model
model:
	curl -L -o sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1'
	curl -L -o sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'

## install_pip : [INSTALL] Install python packages within the virtualvenv
install_pip:
	./venv/bin/pip install -r ./requirements.txt

## venv : [INSTALL] Create the virtualenv
venv:
	virtualenv venv --python=`which python3.7`

## req : [DEV] Freeze pyhton requirements into requirements.txt
req:
	./venv/bin/pip freeze | grep -v "pkg-resources" > requirements.txt
