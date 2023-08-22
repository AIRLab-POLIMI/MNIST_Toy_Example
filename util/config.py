import argparse
import yaml

# Define a class to hold the configuration parameters
class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    # Load the YAML configuration file
    def load_config(config_filepath = "config.yaml"):
        
        # Load the configuration from the YAML file
        with open(config_filepath, "r") as file:
            config_dict = yaml.safe_load(file)

        return Config(config_dict)