"""Интерфейс: ввод, кнопка, сообщение или список. Сеть не используется."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import llm
from llm import MSG_ACCESS, MSG_EMPTY, MSG_TOO_LONG

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=30).run()


def _by_label(elements, label: str):
    for element in elements:
        if element.label == label:
            return element
    raise AssertionError(f"нет поля «{label}»")


def _submit(app: AppTest) -> AppTest:
    return _by_label(app.button, "Сгенерировать список покупок").click().run()


def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_ACCESS_TOKEN",
        "GIGACHAT_USER",
        "GIGACHAT_PASSWORD",
        "GIGACHAT_CERT_FILE",
        "GIGACHAT_KEY_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_ui_has_input_controls() -> None:
    app = _app()

    assert _by_label(app.text_area, "Что приготовить").label == "Что приготовить"
    assert _by_label(app.selectbox, "На сколько человек").options == ["1", "2", "4", "6"]
    assert _by_label(app.selectbox, "Формат").options == [
        "Список продуктов",
        "Список + шаги",
    ]
    assert _by_label(app.button, "Сгенерировать список покупок").label
    assert _by_label(app.text_area, "Список покупок").label == "Список покупок"


def test_ui_empty_text_shows_message_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("API не должен вызываться")

    monkeypatch.setattr(llm, "_call_model", fail)
    app = _app()
    _by_label(app.text_area, "Что приготовить").set_value("  ")
    app = _submit(app)

    assert [item.value for item in app.error] == [MSG_EMPTY]
    assert len(app.exception) == 0
    assert _by_label(app.text_area, "Список покупок").value in ("", None)


def test_ui_long_text_shows_message_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("API не должен вызываться")

    monkeypatch.setattr(llm, "_call_model", fail)
    app = _app()
    _by_label(app.text_area, "Что приготовить").set_value("а" * 501)
    app = _submit(app)

    assert [item.value for item in app.error] == [MSG_TOO_LONG]
    assert len(app.exception) == 0


def test_ui_normal_text_shows_shopping_list(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_model(system_text: str, user_text: str) -> str:
        captured["user"] = user_text
        return "- спагетти — 200 г\n- бекон — 100 г"

    monkeypatch.setattr(llm, "_call_model", fake_model)
    app = _app()
    _by_label(app.text_area, "Что приготовить").set_value("паста карбонара")
    _by_label(app.selectbox, "На сколько человек").set_value("6")
    _by_label(app.selectbox, "Формат").set_value("Список + шаги")
    app = _submit(app)

    assert len(app.error) == 0
    assert len(app.exception) == 0
    assert _by_label(app.text_area, "Список покупок").value == (
        "- спагетти — 200 г\n- бекон — 100 г"
    )
    assert "Количество человек: 6" in captured["user"]
    assert "Список + шаги" in captured["user"]
    assert "паста карбонара" in captured["user"]


def test_ui_missing_key_is_one_sentence(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_credentials(monkeypatch)
    app = _app()
    _by_label(app.text_area, "Что приготовить").set_value("омлет")
    app = _submit(app)

    assert [item.value for item in app.error] == [MSG_ACCESS]
    assert len(app.exception) == 0
    assert "Traceback" not in app.error[0].value
