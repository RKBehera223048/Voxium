import os
from abc import ABC, abstractmethod
from typing import Optional

class BaseExporter(ABC):
    """Abstract base class for document exporters."""
    
    @abstractmethod
    def export(self, content: str, output_path: str, title: Optional[str] = None) -> str:
        """Export the markdown content to the specific format at output_path.
        Returns the absolute path of the generated file."""
        pass

class DocumentExporter:
    """Unified document export interface."""
    
    def __init__(self):
        from .markdown import MarkdownExporter
        from .docx import DocxExporter
        from .ppt import PptExporter
        from .latex import LatexExporter
        from .txt import TxtExporter
        
        self.exporters = {
            "md": MarkdownExporter(),
            "markdown": MarkdownExporter(),
            "docx": DocxExporter(),
            "ppt": PptExporter(),
            "pptx": PptExporter(),
            "tex": LatexExporter(),
            "latex": LatexExporter(),
            "txt": TxtExporter(),
            "text": TxtExporter()
        }
        
    def export(self, content: str, format: str, output_path: str, title: Optional[str] = None) -> str:
        """Export document to specified format."""
        fmt = format.lower().strip('.')
        if fmt not in self.exporters:
            raise ValueError(f"Unsupported export format: {format}")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
        exporter = self.exporters[fmt]
        return exporter.export(content, output_path, title)
