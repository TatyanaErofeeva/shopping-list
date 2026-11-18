"""Три обязательных сценария и карта ошибок API без утечки секретов."""

import json
from types import SimpleNamespace

import pytest

import llm
from llm import (
    FORMAT_LIST,
    FORMAT_STEPS,
    MAX_INPUT_LENGTH,
    MSG_ACCESS,
    MSG_EMPTY,
    MSG_EMPTY_MODEL,
    MSG_NETWORK,
    MSG_SERVER,
    MSG_TOO_LONG,
    ShoppingListError,
    generate_shopping_list,
)

SECRET = "super-secret-key"


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)


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


def _list_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        messages=[SimpleNamespace(content=[SimpleNamespace(text=text)])]
    )


def test_normal_text_returns_shopping_list(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_model(system_text: str, user_text: str) -> str:
        captured["system"] = system_text
        captured["user"] = user_text
        return "- спагетти — 200 г\n- бекон — 100 г\n- яйца — 2 шт."

    monkeypatch.setattr(llm, "_call_model", fake_model)

    result = generate_shopping_list("паста карбонара", "2", FORMAT_LIST)

    assert result == "- спагетти — 200 г\n- бекон — 100 г\n- яйца — 2 шт."
    assert "Количество человек: 2" in captured["user"]
    assert "Список продуктов" in captured["user"]
    assert "Не добавляй шаги" in captured["user"]
    assert "паста карбонара" in captured["user"]


def test_steps_format_is_requested_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_model(system_text: str, user_text: str) -> str:
        captured["user"] = user_text
        return "Продукты:\n- мука — 300 г\nШаги:\n1. Смешайте.\n2. Испеките."

    monkeypatch.setattr(llm, "_call_model", fake_model)

    result = generate_shopping_list("блины", "4", FORMAT_STEPS)

    assert "Продукты:" in result
    assert "Количество человек: 4" in captured["user"]
    assert "Список + шаги" in captured["user"]
    assert "от 5 до 8" in captured["user"]


def test_empty_text_does_not_call_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("API не должен вызываться")

    monkeypatch.setattr(llm, "_call_model", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("   \n\t", "1", FORMAT_LIST)

    assert error.value.message == MSG_EMPTY


def test_long_text_does_not_call_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("API не должен вызываться")

    monkeypatch.setattr(llm, "_call_model", fail)
    dish = "борщ " + ("овощ " * 200)

    assert len(dish.strip()) > MAX_INPUT_LENGTH
    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list(dish, "6", FORMAT_STEPS)

    assert error.value.message == MSG_TOO_LONG


def test_length_limit_allows_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_call_model", lambda _system, _user: "- соль — щепотка")

    result = generate_shopping_list("а" * MAX_INPUT_LENGTH, "1", FORMAT_LIST)

    assert "соль" in result

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("а" * (MAX_INPUT_LENGTH + 1), "1", FORMAT_LIST)

    assert error.value.message == MSG_TOO_LONG


def test_missing_key_does_not_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_credentials(monkeypatch)

    def fail() -> None:
        raise AssertionError("клиент API не должен создаваться")

    monkeypatch.setattr(llm, "_make_client", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("омлет", "1", FORMAT_LIST)

    assert error.value.message == MSG_ACCESS
    assert SECRET not in error.value.message


def test_client_error_is_not_retried_and_hides_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"count": 0}

    class AuthenticationError(Exception):
        status_code = 401

        def __str__(self) -> str:
            return f"invalid token {SECRET}"

    def fail(_system: str, _user: str) -> None:
        calls["count"] += 1
        raise AuthenticationError()

    monkeypatch.setattr(llm, "_request_once", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("омлет", "2", FORMAT_LIST)

    assert calls["count"] == 1
    assert error.value.message == MSG_ACCESS
    assert SECRET not in error.value.message


def test_server_error_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"count": 0}

    class ServerError(Exception):
        def __init__(self) -> None:
            super().__init__(f"bad gateway {SECRET}")
            self.status_code = 503

    def fail(_system: str, _user: str) -> None:
        calls["count"] += 1
        raise ServerError()

    monkeypatch.setattr(llm, "_request_once", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("суп", "4", FORMAT_LIST)

    assert calls["count"] == 2
    assert error.value.message == MSG_SERVER
    assert SECRET not in error.value.message


def test_server_error_succeeds_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"count": 0}

    class ServerError(Exception):
        status_code = 500

    def flaky(_system: str, _user: str) -> SimpleNamespace:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ServerError("temporary")
        return _list_response("- рис — 200 г")

    monkeypatch.setattr(llm, "_request_once", flaky)

    result = generate_shopping_list("плов", "4", FORMAT_LIST)

    assert calls["count"] == 2
    assert result == "- рис — 200 г"


def test_timeout_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"count": 0}

    def fail(_system: str, _user: str) -> None:
        calls["count"] += 1
        raise TimeoutError(f"timed out, key={SECRET}")

    monkeypatch.setattr(llm, "_request_once", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("салат", "2", FORMAT_LIST)

    assert calls["count"] == 2
    assert error.value.message == MSG_NETWORK
    assert SECRET not in error.value.message


def test_empty_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm,
        "_request_once",
        lambda _system, _user: _list_response("```\n   \n```"),
    )

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("каша", "1", FORMAT_LIST)

    assert error.value.message == MSG_EMPTY_MODEL


def test_broken_json_is_one_sentence_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    calls = {"count": 0}

    def fail(_system: str, _user: str) -> None:
        calls["count"] += 1
        raise json.JSONDecodeError("broken body", "not-json", 0)

    monkeypatch.setattr(llm, "_request_once", fail)

    with pytest.raises(ShoppingListError) as error:
        generate_shopping_list("каша", "1", FORMAT_LIST)

    assert calls["count"] == 1
    assert error.value.message == MSG_EMPTY_MODEL


def test_legacy_choices_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="- лук — 1 шт."))]
    )
    monkeypatch.setattr(llm, "_request_once", lambda _system, _user: legacy)

    assert generate_shopping_list("лук", "1", FORMAT_LIST) == "- лук — 1 шт."
