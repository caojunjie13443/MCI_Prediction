import six
import yaml


class Config(object):
    def __init__(self):
        pass


def read_config_from_yaml_file(config_yaml):
    config = Config()
    yaml_config = yaml.load(open(config_yaml), Loader=yaml.FullLoader)
    for (key, value) in six.iteritems(yaml_config):
        config.__dict__[key] = value

    return config


def load_model_configs(config_file, ftype="json"):
    assert (ftype in ("json", "yaml", "yml"))
    if ftype == "json":
        return "Unfinished json config file!"
    else:
        return read_config_from_yaml_file(config_file)