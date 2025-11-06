from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
import yaml

class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls, yaml_file: str):
        super().__init__(settings_cls)
        self.yaml_file = yaml_file

    def get_field_value(self, field_name):
        with open(self.yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return data.get(field_name)

class AppSettings(BaseSettings):
    endpoint_name: str
    catalog: str

    @classmethod
    def settings_customise_sources(cls, init_settings, env_settings, file_secret_settings):
        return (
            YamlConfigSettingsSource(cls, "config.yaml"),
            env_settings,
            init_settings,
            file_secret_settings,
        )