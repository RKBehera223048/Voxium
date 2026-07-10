import re
from typing import List, Dict, Any

class Formatter:
    """Utilities for formatting Markdown text elements."""

    @staticmethod
    def bold(text: str) -> str:
        return f"**{text}**"

    @staticmethod
    def italic(text: str) -> str:
        return f"*{text}*"

    @staticmethod
    def underline(text: str) -> str:
        # Markdown doesn't have native underline, often mapped to HTML or just emphasized
        return f"<u>{text}</u>"

    @staticmethod
    def heading(text: str, level: int = 1) -> str:
        level = max(1, min(6, level))
        return f"{'#' * level} {text}\n"

    @staticmethod
    def code_block(code: str, language: str = "") -> str:
        return f"```{language}\n{code}\n```\n"

    @staticmethod
    def inline_code(text: str) -> str:
        return f"`{text}`"

    @staticmethod
    def unordered_list(items: List[str]) -> str:
        return "\n".join(f"- {item}" for item in items) + "\n"

    @staticmethod
    def ordered_list(items: List[str]) -> str:
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items)) + "\n"

    @staticmethod
    def table(headers: List[str], rows: List[List[str]]) -> str:
        if not headers or not rows:
            return ""
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Header row
        header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |\n"
        
        # Separator row
        sep_str = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|\n"
        
        # Data rows
        rows_str = ""
        for row in rows:
            # Pad row if it has fewer columns than headers
            padded_row = list(row) + [""] * (len(headers) - len(row))
            rows_str += "| " + " | ".join(str(cell).ljust(w) for cell, w in zip(padded_row, col_widths)) + " |\n"

        return header_str + sep_str + rows_str
