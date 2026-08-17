import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir="configs"):
    return {
        "models": load_yaml(f"{config_dir}/models.yaml"),
        "concepts": load_yaml(f"{config_dir}/concepts.yaml"),
        "languages": load_yaml(f"{config_dir}/languages.yaml"),
        "experiment": load_yaml(f"{config_dir}/experiment.yaml"),
    }
