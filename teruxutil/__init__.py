import os

if os.environ.get('TXU_CONFIG_FILE'):
    from .config import Config, YamlConfigStrategy
    config_file = os.environ.get('TXU_CONFIG_FILE')
    Config(YamlConfigStrategy(config_file)).load()
