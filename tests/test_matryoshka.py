from vit.matryoshka import MatryoshkaConfig


class TestMatryoshka:

    def test_config_from_yaml_str(self):
        config = MatryoshkaConfig(ffn_size=10)
        config_str = config.to_yaml()
        config_from_str = MatryoshkaConfig.from_yaml(config_str)
        assert config == config_from_str

    def test_config_from_yaml_path(self, tmp_path):
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            f.write(MatryoshkaConfig(ffn_size=10).to_yaml())
        config_from_path = MatryoshkaConfig.from_yaml(path)
        assert MatryoshkaConfig(ffn_size=10) == config_from_path
