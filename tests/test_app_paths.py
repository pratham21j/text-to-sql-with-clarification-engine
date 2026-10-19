from pathlib import Path
from types import SimpleNamespace

from text_to_sql import app_paths
from text_to_sql.database import DEFAULT_DATABASE_PATH
from text_to_sql.llm_provider import GeminiProvider


class _FakeModels:
    def __init__(self):
        self.arguments = None

    def generate_content(self, **arguments):
        self.arguments = arguments
        return SimpleNamespace(text='{"ok":true}')


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_source_resource_paths_use_project_root():
    project_root = Path(__file__).resolve().parents[1]

    assert app_paths.application_directory() == project_root
    assert app_paths.resource_path("data", "academic.db") == (
        project_root / "data" / "academic.db"
    )
    assert DEFAULT_DATABASE_PATH == project_root / "data" / "academic.db"
    assert app_paths.environment_file_path() == project_root / ".env"


def test_frozen_paths_separate_user_config_from_bundled_resources(
    monkeypatch,
    tmp_path,
):
    application_directory = tmp_path / "TextToSQL"
    bundle_directory = application_directory / "_internal"
    executable = application_directory / "TextToSQL.exe"
    monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(app_paths.sys, "executable", str(executable))
    monkeypatch.setattr(app_paths.sys, "_MEIPASS", str(bundle_directory), raising=False)

    assert app_paths.application_directory() == application_directory
    assert app_paths.environment_file_path() == application_directory / ".env"
    assert app_paths.resource_path("data", "academic.db") == (
        bundle_directory / "data" / "academic.db"
    )


def test_provider_reads_environment_from_executable_directory(
    monkeypatch,
    tmp_path,
):
    application_directory = tmp_path / "TextToSQL"
    application_directory.mkdir()
    config_file = application_directory / ".env"
    config_file.write_text(
        "GEMINI_API_KEY=packaged-test-key\nGEMINI_MODEL=packaged-test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        app_paths.sys,
        "executable",
        str(application_directory / "TextToSQL.exe"),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = _FakeClient()

    provider = GeminiProvider(client=client)
    response = provider.generate_json("prompt", {"type": "OBJECT"})

    assert response == '{"ok":true}'
    assert client.models.arguments["model"] == "packaged-test-model"
    assert config_file.parent == app_paths.environment_file_path().parent
