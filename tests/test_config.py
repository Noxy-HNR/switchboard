"""Invalid configuration must fail before a service starts or sends requests."""
import pytest
from pydantic import ValidationError
from switchboard.config import Config, load_config


@pytest.mark.parametrize('value', [
    {'server': {'port': 70000}}, {'server': {'prt': 8003}},
    {'decisions': {'backends': ['typo']}}, {'decisions': {'budget_ms': -1}},
    {'decisions': {'cache_max_entries': 0}},
    {'decisions': {'local': {'timeout_s': 0}}},
    {'policy': {'chars_per_token': 0}}, {'policy': {'difficulty_ewma_alpha': 2}},
    {'policy': {'value_of_quality_s': float('nan')}},
    {'privacy': {'preview_chars': 999}}, {'observability': {'retention_days': 0}},
])
def test_bad_configuration_is_rejected(value):
    with pytest.raises(ValidationError):
        Config.model_validate(value)


def test_environment_overrides_and_secrets_do_not_enter_repr(tmp_path):
    path = tmp_path / 'config' / 'switchboard.toml'
    path.parent.mkdir()
    path.write_text('[server]\nport=8002\n[privacy]\nallow_cloud=true\n')
    c = load_config(path, environ={'SWITCHBOARD_SERVER__PORT':'8123',
        'SWITCHBOARD_PRIVACY__ALLOW_CLOUD':'false', 'TYPESAFE_API_KEY':'synthetic-secret'})
    assert c.server.port == 8123
    assert not c.jev_enabled()
    assert 'synthetic-secret' not in repr(c)
    with pytest.raises(ValueError):
        load_config(path, environ={'SWITCHBOARD_PRIVACY__ALLOW_CLOUD':'flase'})
