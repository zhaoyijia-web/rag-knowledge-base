from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """读取项目 .env；系统环境变量优先，且不回显任何密钥。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class DeepSeekClient:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，请填写项目根目录的 .env")
        self.client = httpx.Client(timeout=httpx.Timeout(180.0, connect=30.0))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_output: bool = False,
        thinking: bool = False,
    ) -> str:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    finish_reason = data["choices"][0].get("finish_reason", "unknown")
                    raise ValueError(f"API 返回空 content，finish_reason={finish_reason}")
                return content
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"DeepSeek API 请求失败：{last_error}") from last_error
