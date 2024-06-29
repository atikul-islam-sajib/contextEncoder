import yaml


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
