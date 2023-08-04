import yaml


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict save to yaml"""
    with open(save_path, 'a+') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True, sort_keys=False))


def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
