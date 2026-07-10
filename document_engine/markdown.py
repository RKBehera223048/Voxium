import os
from typing import Optional
from .exporter import BaseExporter

class MarkdownExporter(BaseExporter):
    """Exports content as standard Markdown."""
    
    def export(self, content: str, output_path: str, title: Optional[str] = None) -> str:
        final_content = ""
        
        if title:
            final_content += f"# {title}\n\n"
            
        final_content += content
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        return os.path.abspath(output_path)
