import re

import jieba


def tokenize(text: str) -> list[str]:
    """适用于中文制度和设备参数的 BM25 分词。"""
    normalized = text.lower().replace("\u3000", " ")
    words = [word.strip() for word in jieba.lcut(normalized) if word.strip()]
    # 保留型号、标准号、数值和单位等精确字符串。
    exact = re.findall(r"[a-z]+(?:[-_.][a-z0-9]+)*|\d+(?:\.\d+)?(?:mm|mpa|kw|v|hz|℃|%)?", normalized)
    return words + exact

