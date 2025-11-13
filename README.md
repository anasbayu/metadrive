# Public Repository of Anzz thesis

## How to setup
### Local Machine
- make sure UV is installed in your local machine
- clone this repo to your local machine
- navigate to your clonned project folder
- type 'uv sync'
- install metadrive, follow instruction on https://metadrive-simulator.readthedocs.io/en/latest/install.html
- before pulling metadrive asset, make sure to install its dependency


### Docker
- make sure docker is installed in your local machine
- pull a UV image (see https://docs.astral.sh/uv/guides/integration/docker/#available-images)
- example: 'docker pull ghcr.io/astral-sh/uv:python3.12-bookworm-slim'
- clone this project in a dedicated location
- run UV docker image (docker run -t -d --name <project-name> --runtime=nvidia --net=host -v /home/path/to/metadrive:/root/metadrive <image-id>)
- execute the image bash (docker exec -it <project-name> /bin/bash)
- navigate to your volume (e.g. cd /root/metadrive)
- sync the project (uv sync)
- install metadrive, follow instruction on https://metadrive-simulator.readthedocs.io/en/latest/install.html
- before pulling metadrive asset, make sure to install its dependency
apt-get update
apt-get install -y libgl1-mesa-glx
- pull metadrive asset with "python -m metadrive.pull_asset"
