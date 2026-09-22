from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
CONVERTED_DIR = PROJECT_ROOT / "data" / "converted"


@dataclass(frozen=True)
class SourceSpec:
    document_id: str
    relative_path: str
    title: str
    knowledge_domain: str
    document_type: str
    parser: str
    equipment_name: str | None = None
    cutoff_caption: str | None = None


# 白名单即首期知识库的全部数据范围；目录中的其他文件不会被扫描。
SOURCES = (
    SourceSpec("employee_handbook", "发周2026.8.4/奥智员工手册_法律法规、规章制度等.docx", "奥智员工手册", "制度文化", "员工手册", "structured"),
    SourceSpec("question_bank", "发周2026.8.12/题库 汇总.docx", "题库汇总", "培训题库", "题库", "question_bank"),
    SourceSpec("straightening_cutting_machine", "发周2026.8.4/2025年奥智  矫直切割机.docx", "矫直切割机企业标准", "设备标准", "企业标准", "structured", "矫直切割机", "图1  矫直切割机"),
    SourceSpec("pipe_rewinding_machine", "发周2026.8.4/2025年奥智  管材复绕机.doc", "管材复绕机企业标准", "设备标准", "企业标准", "structured", "管材复绕机", "图1  复绕机"),
)


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    style: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    knowledge_domain: str
    document_type: str
    source_path: str
    section_path: str
    page: None
    question_id: str | None
    equipment_name: str | None
    text: str


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" *\n *", "\n", text).strip()


def iter_inner_content(document: DocumentObject) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        values: list[str] = []
        for cell in row.cells:
            value = normalize_text(" ".join(p.text for p in cell.paragraphs))
            if not values or value != values[-1]:
                values.append(value)
        line = " | ".join(value for value in values if value)
        if line and (not rows or line != rows[-1]):
            rows.append(line)
    return "\n".join(rows)


def convert_doc_to_docx(source: Path) -> Path:
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    target = CONVERTED_DIR / f"{source.stem}.docx"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    command = ["/usr/bin/textutil", "-convert", "docx", "-output", str(target), str(source)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not target.exists():
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"DOC 转 DOCX 失败：{source.name}；{detail}")
    return target


def read_blocks(source: Path, cutoff_caption: str | None = None) -> list[Block]:
    readable = convert_doc_to_docx(source) if source.suffix.lower() == ".doc" else source
    document = Document(readable)
    blocks: list[Block] = []
    cutoff_key = normalize_text(cutoff_caption or "").replace(" ", "")
    for item in iter_inner_content(document):
        if isinstance(item, Paragraph):
            text = normalize_text(item.text)
            if not text:
                continue
            if cutoff_key and text.replace(" ", "") == cutoff_key:
                break
            blocks.append(Block("paragraph", text, item.style.name if item.style else ""))
        else:
            text = table_to_text(item)
            if text:
                blocks.append(Block("table", text, "Table"))
    return blocks


def split_long_text(text: str, max_chars: int = 700) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = [s for s in re.split(r"(?<=[。；！？])|\n", text) if s]
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
        elif current and len(current) + len(sentence) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        parts.append(current)
    return parts


HANDBOOK_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){1,3})\s+\S")
EQUIPMENT_HEADINGS = {
    "前 言", "前言", "范围", "规范性引用文件", "术语和定义", "结构、基本参数", "结构", "基本参数",
    "要求", "性能要求（空载与带载）", "性能要求", "工作噪声", "外观", "机器安全", "电气安全",
    "传动要求", "试验方法", "检验规则", "检测规则", "交收检验", "型式检验", "包装、标志、运输、贮存",
    "标志", "包装", "运输", "贮存", "质量服务", "关键零、部件及装配要求", "管材复绕机", "矫直切割机",
}


def heading_level(block: Block, spec: SourceSpec) -> int | None:
    style = block.style.lower().replace(" ", "")
    if style.startswith("heading"):
        match = re.search(r"(\d+)$", style)
        return int(match.group(1)) if match else 1
    if "章标题" in block.style:
        return 1
    if "一级条标题" in block.style:
        return 2
    if spec.document_id == "employee_handbook":
        match = HANDBOOK_NUMBERED_HEADING.match(block.text)
        if match and len(block.text) <= 80:
            return min(match.group(1).count(".") + 1, 4)
    if spec.equipment_name and block.text in EQUIPMENT_HEADINGS:
        return 1 if block.text in {"前 言", "前言", spec.equipment_name} else 2
    return None


def make_chunk(spec: SourceSpec, source_path: Path, index: int, section_path: str, body: str, question_id: str | None = None) -> Chunk:
    section_path = section_path or spec.title
    body = normalize_text(body)
    text = body if body.startswith(section_path) else f"{section_path}\n{body}"
    return Chunk(
        f"{spec.document_id}_{index:04d}", spec.document_id, spec.title, spec.knowledge_domain,
        spec.document_type, str(source_path.relative_to(PROJECT_ROOT)), section_path, None,
        question_id, spec.equipment_name, text,
    )


def chunk_structured(spec: SourceSpec, source_path: Path) -> list[Chunk]:
    headings: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_texts: list[str] = []

    def flush_section() -> None:
        nonlocal current_texts
        if current_texts:
            sections.append((" > ".join(headings) or spec.title, current_texts))
            current_texts = []

    for block in read_blocks(source_path, spec.cutoff_caption):
        level = heading_level(block, spec)
        if level is not None:
            flush_section()
            headings[:] = headings[:level - 1]
            headings.append(block.text)
        else:
            current_texts.append(block.text)
    flush_section()

    chunks: list[Chunk] = []
    for section_path, texts in sections:
        buffer = ""
        for text in texts:
            for part in split_long_text(text):
                candidate = f"{buffer}\n{part}".strip() if buffer else part
                if buffer and len(candidate) > 700:
                    chunks.append(make_chunk(spec, source_path, len(chunks) + 1, section_path, buffer))
                    buffer = part
                else:
                    buffer = candidate
        if buffer:
            chunks.append(make_chunk(spec, source_path, len(chunks) + 1, section_path, buffer))
    return chunks


QUESTION_SECTIONS = {
    "第一部分：单选题（共50题）": ("single", "单选题"),
    "第二部分：多选题（共50题）": ("multiple", "多选题"),
    "第三部分：判断题（共50题）": ("judgment", "判断题"),
    "第五部分：解答题（共50题）": ("short", "解答题"),
}


def chunk_question_bank(spec: SourceSpec, source_path: Path) -> list[Chunk]:
    lines = [block.text for block in read_blocks(source_path) if block.kind == "paragraph"]
    starts = [i for i, line in enumerate(lines) if line == "第一部分：单选题（共50题）"]
    if len(starts) < 2:
        raise ValueError("题库未找到目录后的正式第一部分")
    lines = lines[starts[1]:]
    chunks: list[Chunk] = []
    section_heading = section_key = category = ""
    counters = {key: 0 for key, _ in QUESTION_SECTIONS.values()}
    current: list[str] = []

    def emit() -> None:
        nonlocal current
        if not current:
            return
        counters[section_key] += 1
        question_id = f"QB-{section_key.upper()}-{counters[section_key]:03d}"
        path = " > ".join([part for part in (section_heading, category) if part])
        chunks.append(make_chunk(spec, source_path, len(chunks) + 1, path, "\n".join(current), question_id))
        current = []

    for line in lines:
        if line in QUESTION_SECTIONS:
            emit()
            section_heading = line
            section_key, _ = QUESTION_SECTIONS[line]
            category = ""
            continue
        if not section_key:
            continue
        if section_key == "short":
            if line.startswith("问："):
                emit()
                current = [line]
            elif current:
                current.append(line)
                if line.startswith("依据："):
                    emit()
            else:
                category = line
        else:
            # 该题库把题干、选项和答案保存在同一个 Word 段落中。
            # 也兼容未来出现的跨段题目。
            if "答案：" in line:
                emit()
                current = [line]
                emit()
            elif current:
                current.append(line)
            else:
                current = [line]
    emit()
    expected = {"single": 50, "multiple": 50, "judgment": 50, "short": 50}
    if counters != expected:
        raise ValueError(f"题库题数异常：{counters}，期望：{expected}")
    return chunks


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for spec in SOURCES:
        source_path = PROJECT_ROOT / spec.relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"找不到白名单文件：{source_path}")
        parser = chunk_question_bank if spec.parser == "question_bank" else chunk_structured
        chunks.extend(parser(spec, source_path))
    return chunks


def write_jsonl(chunks: Iterable[Chunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def write_summary(chunks: list[Chunk], output: Path) -> Path:
    summary_path = output.with_name("chunks_summary.json")
    by_document: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        item = by_document.setdefault(chunk.document_id, {"chunks": 0, "characters": 0})
        item["chunks"] += 1
        item["characters"] += len(chunk.text)
    summary_path.write_text(json.dumps({"total_chunks": len(chunks), "documents": by_document}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="解析并切分奥智 RAG 首期四份 Word 文档")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    chunks = build_chunks()
    write_jsonl(chunks, output)
    summary = write_summary(chunks, output)
    print(f"已生成 {len(chunks)} 个文本块：{output}")
    print(f"统计信息：{summary}")


if __name__ == "__main__":
    main()
