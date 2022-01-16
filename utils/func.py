import yaml
import wandb

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

def data_version_control(config):
    for mode in ["train", "test"]:
        with wandb.init(entity="jjerry", job_type="dvc", project="TestArtifact") as run:
            print(f"{mode.capitalize()} Dataset Version Control")
            artifact = wandb.Artifact(name=f'{mode}_dataset', type="dataset", description=config["DATA"]["NAME"])
            artifact.add_dir(os.path.join(config["DATA"]["ROOT"], config["DATA"]["NAME"], mode))
            run.log_artifact(artifact)
            run.finish()

def model_version_control(config, mode):
    with wandb.init(entity="jjerry", job_type="mvc", project="TestArtifact") as run:
        artifact = wandb.Artifact(name=f'{config["MODEL"]["BASEMODEL"]}', type="model")
        run.use_artifact(f'{mode}_dataset:latest')
        artifact.add_file(os.path.join(config["COMMON"]["SAVEPATH"], "best.pth.tar"))

        run.log_artifact(artifact)
        run.finish()