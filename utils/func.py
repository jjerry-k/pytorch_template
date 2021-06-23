import yaml

def config_parser(path):
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    for main_key, main_val in config.items():
        print(main_key)
        for sub_key, sub_val in main_val.items():
            print(f"{sub_key}: {sub_val}")
        print("\n")
    return config

def write_yaml(path, config):
    with open(path, "w") as ymlfile:
        yaml.dump(config, ymlfile, sort_keys=False)