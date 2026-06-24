"""
Voxium — Document Editor with Deterministic AST Parsing
=========================================================
MCP action handler for document formatting, highlighting,
and text manipulation in the active editor.

Uses deterministic AST parsing (inspired by graphify's extract.py)
to map document structure using pure CPU math before modifying text.
This saves LLM reasoning cycles — the LLM only sees the target
region + context instead of the entire document.

Supported file types:
    - Python: ast.parse() for classes, functions, imports
    - Markdown: regex + markdown lib for headers, code blocks, lists
    - Plain text: Paragraph detection via double-newline splitting
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# AST Data Structures
# =============================================================================

@dataclass
class ASTNode:
    """A node in the document's structural tree."""
    type: str        # module | class | function | import | decorator |
                     # header | code_block | paragraph | list_item
    name: str
    start_line: int  # 1-indexed, inclusive
    end_line: int    # 1-indexed, inclusive
    children: List[ASTNode] = field(default_factory=list)
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class DocumentAST:
    """The full structural parse of a document."""
    root: ASTNode
    filetype: str
    line_count: int

    def to_dict(self) -> dict:
        return {
            "filetype": self.filetype,
            "line_count": self.line_count,
            "structure": self.root.to_dict(),
        }


# =============================================================================
# Content Hashing
# =============================================================================

def _content_hash(text: str) -> str:
    """MD5 hash of content for change detection."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


# =============================================================================
# Python AST Parser
# =============================================================================

def _parse_python(content: str) -> ASTNode:
    """
    Parse Python source using ast.parse() to extract structural elements.

    Mirrors graphify's deterministic AST extraction approach but adapted
    for document editing rather than graph extraction. Returns exact
    line ranges for each structural element.
    """
    lines = content.split("\n")
    total_lines = len(lines)

    root = ASTNode(
        type="module",
        name="<module>",
        start_line=1,
        end_line=total_lines,
        content_hash=_content_hash(content),
    )

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        # If parsing fails, return a flat paragraph structure
        logger.warning("Python AST parse failed: %s", e)
        root.children.append(ASTNode(
            type="error",
            name=f"SyntaxError at line {e.lineno}",
            start_line=1,
            end_line=total_lines,
            metadata={"error": str(e)},
        ))
        return root

    # Extract module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        # Docstring is typically the first expression statement
        first_stmt = tree.body[0] if tree.body else None
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
            end_ln = first_stmt.end_lineno or first_stmt.lineno
            root.children.append(ASTNode(
                type="docstring",
                name="module docstring",
                start_line=first_stmt.lineno,
                end_line=end_ln,
                content_hash=_content_hash(docstring),
            ))

    # Walk the top-level AST nodes
    for node in ast.iter_child_nodes(tree):
        parsed = _parse_python_node(node, content, lines)
        if parsed:
            root.children.append(parsed)

    return root


def _parse_python_node(
    node: ast.AST,
    content: str,
    lines: list[str],
) -> Optional[ASTNode]:
    """Parse a single Python AST node into an ASTNode."""

    if isinstance(node, ast.ClassDef):
        end_line = node.end_lineno or node.lineno
        class_content = "\n".join(lines[node.lineno - 1:end_line])

        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        class_node = ASTNode(
            type="class",
            name=node.name,
            start_line=node.lineno,
            end_line=end_line,
            content_hash=_content_hash(class_content),
            metadata={
                "bases": bases,
                "decorators": [
                    ast.unparse(d) for d in node.decorator_list
                ],
            },
        )

        # Parse class body (methods, nested classes)
        for child in ast.iter_child_nodes(node):
            parsed = _parse_python_node(child, content, lines)
            if parsed:
                class_node.children.append(parsed)

        return class_node

    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        end_line = node.end_lineno or node.lineno
        func_content = "\n".join(lines[node.lineno - 1:end_line])

        # Extract arguments
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            if arg.annotation:
                try:
                    arg_name += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            args.append(arg_name)

        # Extract return type
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                pass

        is_async = isinstance(node, ast.AsyncFunctionDef)

        return ASTNode(
            type="function",
            name=node.name,
            start_line=node.lineno,
            end_line=end_line,
            content_hash=_content_hash(func_content),
            metadata={
                "async": is_async,
                "args": args,
                "return_type": return_type,
                "decorators": [
                    ast.unparse(d) for d in node.decorator_list
                ],
                "docstring": ast.get_docstring(node),
            },
        )

    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            import_name = f"import {', '.join(names)}"
        else:
            module = node.module or ""
            names = [alias.name for alias in node.names]
            import_name = f"from {module} import {', '.join(names)}"

        return ASTNode(
            type="import",
            name=import_name,
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
        )

    elif isinstance(node, ast.Assign):
        # Global/class-level assignments
        targets = []
        for target in node.targets:
            try:
                targets.append(ast.unparse(target))
            except Exception:
                targets.append("<unknown>")

        if targets:
            return ASTNode(
                type="assignment",
                name=" = ".join(targets),
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            )

    return None


# =============================================================================
# Markdown Parser
# =============================================================================

# Regex patterns for Markdown structure
_MD_HEADER_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
_MD_FENCED_CODE_RE = re.compile(r'^```(\w*)\s*$', re.MULTILINE)
_MD_LIST_ITEM_RE = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r'^>\s+(.+)$', re.MULTILINE)
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')


def _parse_markdown(content: str) -> ASTNode:
    """
    Parse Markdown content to extract structural elements.

    Extracts headers (with hierarchy), code blocks, lists, and paragraphs
    with exact line positions.
    """
    lines = content.split("\n")
    total_lines = len(lines)

    root = ASTNode(
        type="document",
        name="<document>",
        start_line=1,
        end_line=total_lines,
        content_hash=_content_hash(content),
    )

    # Track current state
    in_code_block = False
    code_block_start = 0
    code_block_lang = ""
    current_paragraph_start = 0
    current_paragraph_lines: list[str] = []

    def _flush_paragraph():
        nonlocal current_paragraph_start, current_paragraph_lines
        if current_paragraph_lines:
            text = " ".join(current_paragraph_lines).strip()
            if text:
                end = current_paragraph_start + len(current_paragraph_lines) - 1
                root.children.append(ASTNode(
                    type="paragraph",
                    name=text[:60] + ("..." if len(text) > 60 else ""),
                    start_line=current_paragraph_start,
                    end_line=end,
                    content_hash=_content_hash(text),
                ))
        current_paragraph_lines = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Fenced code blocks
        if stripped.startswith("```"):
            if in_code_block:
                # End of code block
                root.children.append(ASTNode(
                    type="code_block",
                    name=f"code ({code_block_lang or 'text'})",
                    start_line=code_block_start,
                    end_line=i,
                    metadata={"language": code_block_lang},
                ))
                in_code_block = False
            else:
                # Start of code block
                _flush_paragraph()
                in_code_block = True
                code_block_start = i
                code_block_lang = stripped[3:].strip()
            continue

        if in_code_block:
            continue

        # Headers
        header_match = _MD_HEADER_RE.match(line)
        if header_match:
            _flush_paragraph()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            root.children.append(ASTNode(
                type="header",
                name=title,
                start_line=i,
                end_line=i,
                metadata={"level": level},
            ))
            continue

        # Block quotes
        if stripped.startswith(">"):
            _flush_paragraph()
            root.children.append(ASTNode(
                type="blockquote",
                name=stripped[1:].strip()[:60],
                start_line=i,
                end_line=i,
            ))
            continue

        # List items
        list_match = _MD_LIST_ITEM_RE.match(line)
        if list_match:
            _flush_paragraph()
            indent_level = len(list_match.group(1)) // 2
            text = list_match.group(3).strip()
            root.children.append(ASTNode(
                type="list_item",
                name=text[:60] + ("..." if len(text) > 60 else ""),
                start_line=i,
                end_line=i,
                metadata={"indent": indent_level},
            ))
            continue

        # Horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', stripped):
            _flush_paragraph()
            continue

        # Empty lines flush paragraphs
        if not stripped:
            _flush_paragraph()
            continue

        # Regular text — accumulate into paragraph
        if not current_paragraph_lines:
            current_paragraph_start = i
        current_paragraph_lines.append(stripped)

    # Flush any remaining content
    _flush_paragraph()

    # Close any unclosed code block
    if in_code_block:
        root.children.append(ASTNode(
            type="code_block",
            name=f"code ({code_block_lang or 'text'}) [unclosed]",
            start_line=code_block_start,
            end_line=total_lines,
            metadata={"language": code_block_lang, "unclosed": True},
        ))

    return root


# =============================================================================
# Plain Text Parser
# =============================================================================

def _parse_plaintext(content: str) -> ASTNode:
    """Parse plain text into paragraph blocks."""
    lines = content.split("\n")
    total_lines = len(lines)

    root = ASTNode(
        type="document",
        name="<document>",
        start_line=1,
        end_line=total_lines,
        content_hash=_content_hash(content),
    )

    # Split on double newlines for paragraphs
    current_start = 1
    current_lines: list[str] = []

    for i, line in enumerate(lines, 1):
        if not line.strip():
            if current_lines:
                text = " ".join(current_lines).strip()
                root.children.append(ASTNode(
                    type="paragraph",
                    name=text[:60] + ("..." if len(text) > 60 else ""),
                    start_line=current_start,
                    end_line=i - 1,
                    content_hash=_content_hash(text),
                ))
                current_lines = []
        else:
            if not current_lines:
                current_start = i
            current_lines.append(line.strip())

    if current_lines:
        text = " ".join(current_lines).strip()
        root.children.append(ASTNode(
            type="paragraph",
            name=text[:60] + ("..." if len(text) > 60 else ""),
            start_line=current_start,
            end_line=total_lines,
            content_hash=_content_hash(text),
        ))

    return root


# =============================================================================
# Filetype Detection
# =============================================================================

def _detect_filetype(content: str) -> str:
    """Auto-detect file type from content heuristics."""
    first_lines = content[:500]

    # Python detection
    if (first_lines.startswith("#!")
            and "python" in first_lines.split("\n")[0].lower()):
        return "python"
    if re.search(r'^\s*(def |class |import |from \S+ import )', first_lines, re.MULTILINE):
        return "python"

    # Markdown detection
    if re.search(r'^#{1,6}\s+', first_lines, re.MULTILINE):
        return "markdown"
    if "```" in first_lines:
        return "markdown"

    return "text"


# =============================================================================
# Public API — Structure Parsing
# =============================================================================

def parse_document_structure(
    content: str,
    filetype: str = "auto",
) -> DocumentAST:
    """
    Parse a document into its structural AST.

    This is the deterministic, zero-LLM approach inspired by graphify's
    AST extraction. The structure is computed with pure CPU math,
    allowing precise edits without sending the entire document to an LLM.

    Args:
        content: The document content as a string.
        filetype: "python", "markdown", "text", or "auto" for detection.

    Returns:
        DocumentAST with the structural tree.
    """
    if filetype == "auto":
        filetype = _detect_filetype(content)

    lines = content.split("\n")

    if filetype == "python":
        root = _parse_python(content)
    elif filetype == "markdown":
        root = _parse_markdown(content)
    else:
        root = _parse_plaintext(content)

    return DocumentAST(
        root=root,
        filetype=filetype,
        line_count=len(lines),
    )


# =============================================================================
# Public API — Target Location
# =============================================================================

def locate_edit_target(
    doc_ast: DocumentAST,
    target: str,
) -> Optional[ASTNode]:
    """
    Find the structural node matching a natural-language target description.

    Matching strategies (in priority order):
        1. Exact name match (case-insensitive)
        2. Partial name match (substring)
        3. Type + name ("function process_audio", "header Introduction")
        4. Ordinal ("third paragraph", "second function")
        5. Line number ("line 42", "lines 10-20")

    Args:
        doc_ast: The parsed document structure.
        target: Natural-language description of the edit target.

    Returns:
        The matching ASTNode, or None if no match found.
    """
    if not target:
        return doc_ast.root

    target_lower = target.strip().lower()
    all_nodes = _flatten_nodes(doc_ast.root)

    # Strategy 1: Exact name match
    for node in all_nodes:
        if node.name.lower() == target_lower:
            return node

    # Strategy 2: Type + name ("function process_audio")
    type_name_match = re.match(
        r'^(class|function|method|header|paragraph|code_block|import)\s+(.+)$',
        target_lower,
    )
    if type_name_match:
        target_type = type_name_match.group(1)
        target_name = type_name_match.group(2)
        for node in all_nodes:
            if node.type == target_type and target_name in node.name.lower():
                return node

    # Strategy 3: Ordinal ("third paragraph", "second function")
    ordinal_match = re.match(
        r'^(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+(.+)$',
        target_lower,
    )
    if ordinal_match:
        ordinal_str = ordinal_match.group(1)
        node_type = ordinal_match.group(2).rstrip("s")  # "paragraphs" → "paragraph"

        ordinal_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10,
        }
        idx = ordinal_map.get(ordinal_str)
        if idx is None:
            # Try parsing "3rd", "5th" etc.
            try:
                idx = int(re.sub(r'[a-z]+$', '', ordinal_str))
            except ValueError:
                idx = None

        if idx:
            typed_nodes = [n for n in all_nodes if n.type == node_type]
            if 0 < idx <= len(typed_nodes):
                return typed_nodes[idx - 1]

    # Strategy 4: Line number ("line 42", "lines 10-20")
    line_match = re.match(r'^lines?\s+(\d+)(?:\s*[-–]\s*(\d+))?$', target_lower)
    if line_match:
        target_line = int(line_match.group(1))
        for node in all_nodes:
            if node.start_line <= target_line <= node.end_line:
                return node

    # Strategy 5: Partial name match (substring)
    for node in all_nodes:
        if target_lower in node.name.lower():
            return node

    return None


def _flatten_nodes(node: ASTNode) -> list[ASTNode]:
    """Recursively flatten the AST tree into a list."""
    result = [node]
    for child in node.children:
        result.extend(_flatten_nodes(child))
    return result


# =============================================================================
# Public API — Edit Operations
# =============================================================================

def apply_edit(
    content: str,
    node: ASTNode,
    operation: str,
    payload: str = "",
) -> str:
    """
    Apply an edit operation to a specific structural region.

    Operations:
        - replace: Replace the node's content with payload
        - insert_before: Insert payload before the node
        - insert_after: Insert payload after the node
        - delete: Remove the node's content
        - format: Auto-format the node's content

    Args:
        content: The full document content.
        node: The target ASTNode (with line ranges).
        operation: The edit operation.
        payload: The content for replace/insert operations.

    Returns:
        The modified document content.
    """
    lines = content.split("\n")

    # Convert to 0-indexed
    start = node.start_line - 1
    end = node.end_line

    if operation == "replace":
        payload_lines = payload.split("\n") if payload else []
        lines[start:end] = payload_lines

    elif operation == "insert_before":
        payload_lines = payload.split("\n") if payload else []
        lines[start:start] = payload_lines

    elif operation == "insert_after":
        payload_lines = payload.split("\n") if payload else []
        lines[end:end] = payload_lines

    elif operation == "delete":
        lines[start:end] = []

    elif operation == "format":
        # Basic formatting: strip trailing whitespace, normalize blank lines
        region = lines[start:end]
        formatted = []
        prev_blank = False
        for line in region:
            stripped = line.rstrip()
            is_blank = not stripped
            if is_blank and prev_blank:
                continue  # Collapse multiple blank lines
            formatted.append(stripped)
            prev_blank = is_blank
        lines[start:end] = formatted

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return "\n".join(lines)


# =============================================================================
# MCP Action Handler
# =============================================================================

async def handle_document_action(
    action: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Handle document editing agent commands.

    Orchestrates the three-step deterministic editing flow:
        1. Parse structure (pure CPU — no LLM needed)
        2. Locate target (by name, type, ordinal, or line)
        3. Apply edit (on exact line ranges)

    Actions:
        - analyze: Return the document's structural AST
        - replace, insert_before, insert_after, delete, format:
            Edit operations on a target region
        - locate: Find a structural element by description

    Parameters:
        content: The document text
        filetype: "python" | "markdown" | "text" | "auto"
        target: Natural-language description of the edit target
        payload: Content for replace/insert operations
    """
    logger.info("Document action: %s params=%s", action, list(parameters.keys()))

    content = parameters.get("content", "")
    filetype = parameters.get("filetype", "auto")

    if not content:
        return {
            "success": False,
            "error": "No document content provided",
            "action": action,
        }

    # Step 1: Parse document structure (deterministic, pure CPU)
    try:
        doc_ast = parse_document_structure(content, filetype)
    except Exception as e:
        logger.error("AST parse failed: %s", e)
        return {
            "success": False,
            "error": f"Document parsing failed: {e}",
            "action": action,
        }

    # Action: analyze — return the full AST
    if action == "analyze":
        return {
            "success": True,
            "structure": doc_ast.to_dict(),
            "action": action,
        }

    # Step 2: Locate target
    target = parameters.get("target", "")
    node = locate_edit_target(doc_ast, target)

    # Action: locate — just find the target
    if action == "locate":
        if node:
            return {
                "success": True,
                "node": node.to_dict(),
                "action": action,
            }
        return {
            "success": False,
            "error": f"Could not locate target: {target}",
            "action": action,
        }

    # Step 3: Apply edit
    if action in ("replace", "insert_before", "insert_after", "delete", "format"):
        if not node:
            return {
                "success": False,
                "error": f"Could not locate edit target: {target}",
                "action": action,
            }

        payload = parameters.get("payload", "")
        try:
            result = apply_edit(content, node, action, payload)
            return {
                "success": True,
                "content": result,
                "node": node.to_dict(),
                "action": action,
                "lines_affected": node.line_count,
            }
        except Exception as e:
            logger.error("Edit operation failed: %s", e)
            return {
                "success": False,
                "error": f"Edit failed: {e}",
                "action": action,
            }

    return {
        "success": False,
        "error": f"Unknown action: {action}",
        "action": action,
    }
