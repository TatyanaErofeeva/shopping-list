"""Вызов GigaChat для сценария «список покупок по блюду».

Проверка текста выполняется до создания клиента и до любого сетевого запроса.
В интерфейс попадают только короткие сообщения из этого модуля: текст исключения
провайдера и учётные данные наружу не отдаются.
"""

from __future__ import annotations

import os
import time
from typing import Any

MAX_INPUT_LENGTH = 500
RETRY_PAUSE_SECONDS = 0.5

PEOPLE_OPTIONS = ("1", "2", "4", "6")
FORMAT_LIST = "Список продуктов"
FORMAT_STEPS = "Список + шаги"
FORMAT_OPTIONS = (FORMAT_LIST, FORMAT_STEPS)

MSG_EMPTY = "Напишите, что хотите приготовить."
MSG_TOO_LONG = "Слишком длинный текст — сократите или разбейте на части."
MSG_BAD_OPTIONS = "Выберите количество человек и формат из списка."
MSG_ACCESS = "Не настроен доступ к сервису."
MSG_SERVER = "Сервис временно недоступен. Попробуйте позже."
MSG_NETWORK = "Не удалось получить ответ. Попробуйте ещё раз."
MSG_EMPTY_MODEL = "Пустой ответ модели. Попробуйте переформулировать вопрос."

_CLIENT_ERROR_NAMES = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "ForbiddenError",
        "NotFoundError",
        "RequestEntityTooLargeError",
        "UnprocessableEntityError",
        "RateLimitError",
        "ModelNotSpecifiedError",
    }
)
_BROKEN_RESPONSE_NAMES = frozenset(
    {
        "ValidationError",
        "JSONDecodeError",
        "DecodeError",
        "LengthFinishReasonError",
    }
)
_CREDENTIAL_ENV_VARS = (
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_ACCESS_TOKEN",
    "GIGACHAT_USER",
    "GIGACHAT_PASSWORD",
    "GIGACHAT_CERT_FILE",
    "GIGACHAT_KEY_FILE",
)


class ShoppingListError(Exception):
    """Ошибка, которую можно показать пользователю одной фразой."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def generate_shopping_list(dish: str, people: str, format: str) -> str:
    """Вернуть список покупок. При ошибке выбрасывает ShoppingListError.

    dish — что приготовить, people — «1», «2», «4» или «6»,
    format — «Список продуктов» или «Список + шаги».
    """
    cleaned = _validate_dish(dish)
    response_format = _validate_options(people, format)
    system_text, user_text = _build_prompt(cleaned, str(people).strip(), response_format)
    raw = _call_model(system_text, user_text)
    text = _normalize_output(raw)
    if not text:
        raise ShoppingListError(MSG_EMPTY_MODEL)
    return text


def _validate_dish(dish: str) -> str:
    cleaned = (dish or "").strip()
    if not cleaned:
        raise ShoppingListError(MSG_EMPTY)
    if len(cleaned) > MAX_INPUT_LENGTH:
        raise ShoppingListError(MSG_TOO_LONG)
    return cleaned


def _validate_options(people: str, response_format: str) -> str:
    if str(people).strip() not in PEOPLE_OPTIONS or response_format not in FORMAT_OPTIONS:
        raise ShoppingListError(MSG_BAD_OPTIONS)
    return response_format


def _build_prompt(dish: str, people: str, response_format: str) -> tuple[str, str]:
    system_text = (
        "Ты составляешь список покупок для домашнего блюда. "
        "Отвечай только по-русски. Не добавляй вступление, пояснения и вопросы. "
        "Число людей бери только из сообщения пользователя и пересчитывай количество "
        "продуктов строго под это число. Не предлагай другие блюда."
    )
    if response_format == FORMAT_LIST:
        format_rules = (
            "Формат ответа: Список продуктов.\n"
            "Верни только список продуктов, каждый продукт с новой строки в виде "
            "«- продукт — количество».\n"
            "Не добавляй шаги приготовления, заголовки и комментарии."
        )
    else:
        format_rules = (
            "Формат ответа: Список + шаги.\n"
            "Сначала блок «Продукты:». Каждый продукт с новой строки в виде "
            "«- продукт — количество».\n"
            "Затем блок «Шаги:». Дай от 5 до 8 коротких шагов приготовления, "
            "каждый шаг с новой строки и с номером. Один шаг — одно короткое предложение."
        )
    user_text = (
        f"Блюдо: {dish}\n"
        f"Количество человек: {people}\n"
        f"{format_rules}\n"
        f"Объём продуктов рассчитай на {people} чел."
    )
    return system_text, user_text


def _call_model(system_text: str, user_text: str) -> str:
    kind = "network"
    for attempt in range(2):
        try:
            response = _request_once(system_text, user_text)
            return _extract_text(response)
        except ShoppingListError:
            raise
        except Exception as exc:
            kind = _classify(exc)
            if kind in {"server", "network"} and attempt == 0:
                time.sleep(RETRY_PAUSE_SECONDS)
                continue
            break
    raise ShoppingListError(_message_for(kind))


def _message_for(kind: str) -> str:
    if kind == "client":
        return MSG_ACCESS
    if kind == "server":
        return MSG_SERVER
    if kind == "broken":
        return MSG_EMPTY_MODEL
    return MSG_NETWORK


def _request_once(system_text: str, user_text: str) -> Any:
    if not _has_credentials():
        raise ShoppingListError(MSG_ACCESS)
    client = _make_client()
    try:
        payload = _payload(system_text, user_text)
        chat = getattr(client, "chat", None)
        create = getattr(chat, "create", None)
        if callable(create):
            return create(payload)
        if callable(chat):
            return chat(payload)
        raise ShoppingListError(MSG_ACCESS)
    finally:
        _close_client(client)


def _make_client() -> Any:
    try:
        from gigachat import GigaChat
    except ImportError:
        raise ShoppingListError(MSG_ACCESS) from None

    kwargs: dict[str, Any] = {
        "model": os.getenv("GIGACHAT_MODEL", "GigaChat-2").strip() or "GigaChat-2",
        "scope": os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip() or "GIGACHAT_API_PERS",
        "verify_ssl_certs": _env_bool("GIGACHAT_VERIFY_SSL_CERTS", True),
        "timeout": _timeout(),
        "max_retries": 0,
    }
    _copy_env(kwargs, "credentials", "GIGACHAT_CREDENTIALS")
    _copy_env(kwargs, "access_token", "GIGACHAT_ACCESS_TOKEN")
    _copy_env(kwargs, "user", "GIGACHAT_USER")
    _copy_env(kwargs, "password", "GIGACHAT_PASSWORD")
    _copy_env(kwargs, "ca_bundle_file", "GIGACHAT_CA_BUNDLE_FILE")
    _copy_env(kwargs, "base_url", "GIGACHAT_BASE_URL")
    _copy_env(kwargs, "auth_url", "GIGACHAT_AUTH_URL")
    return GigaChat(**kwargs)


def _payload(system_text: str, user_text: str) -> Any:
    try:
        from gigachat.models import ChatCompletionRequest, ChatMessage
    except ImportError:
        return {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
        }
    return ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content=system_text),
            ChatMessage(role="user", content=user_text),
        ],
        temperature=0.2,
    )


def _close_client(client: Any) -> None:
    closer = getattr(client, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        return


def _has_credentials() -> bool:
    if os.getenv("GIGACHAT_CREDENTIALS", "").strip():
        return True
    if os.getenv("GIGACHAT_ACCESS_TOKEN", "").strip():
        return True
    if os.getenv("GIGACHAT_USER", "").strip() and os.getenv("GIGACHAT_PASSWORD", "").strip():
        return True
    if os.getenv("GIGACHAT_CERT_FILE", "").strip() and os.getenv("GIGACHAT_KEY_FILE", "").strip():
        return True
    return False


def _copy_env(kwargs: dict[str, Any], argument: str, env_name: str) -> None:
    value = os.getenv(env_name, "").strip()
    if value:
        kwargs[argument] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _timeout() -> float:
    raw = os.getenv("GIGACHAT_TIMEOUT", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    if value <= 0:
        return 30.0
    return value


def _classify(exc: BaseException) -> str:
    status = _status_code(exc)
    if status is not None and 500 <= status <= 599:
        return "server"
    if status is not None and 400 <= status <= 499:
        return "client"
    name = type(exc).__name__
    if name in _CLIENT_ERROR_NAMES:
        return "client"
    if name == "ServerError":
        return "server"
    if name in _BROKEN_RESPONSE_NAMES or isinstance(exc, (ValueError, UnicodeError)):
        return "broken"
    if _is_network(exc):
        return "network"
    return "network"


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    return None


def _is_network(exc: BaseException) -> bool:
    name = type(exc).__name__
    network_names = {
        "TimeoutException",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "NetworkError",
        "RemoteProtocolError",
        "ReadError",
        "WriteError",
        "CloseError",
        "ProtocolError",
    }
    if name in network_names or isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    try:
        import httpx
    except ImportError:
        return False
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _extract_text(response: object) -> str:
    try:
        messages = getattr(response, "messages", None)
        if messages:
            text = _content_to_text(getattr(messages[0], "content", None))
            if text:
                return text
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None) if message is not None else None
            return _content_to_text(content)
    except Exception:
        return ""
    return ""


def _content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    try:
        iterator = list(content)
    except TypeError:
        return ""
    for part in iterator:
        if isinstance(part, str):
            if part.strip():
                parts.append(part.strip())
            continue
        if isinstance(part, dict):
            text = part.get("text") or part.get("content")
        else:
            text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _normalize_output(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
