import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../config", config_name="main")
def process_data(config: DictConfig):

    print(OmegaConf.to_yaml(config))
    # print(config.process.keep_columns)


process_data()
