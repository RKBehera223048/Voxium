import os
import re
from typing import Optional
from .exporter import BaseExporter

class TxtExporter(BaseExporter):
    """Exports content as plain text, stripping markdown formatting."""
    
    def export(self, content: str, output_path: str, title: Optional[str] = None) -> str:
        # Strip bold/italic markers
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', content)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
        
        # Strip code blocks and inline code
        text = re.sub(r'```[a-zA-Z]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Strip headings
        text = re.sub(r'^#+\s+(.*?)$', r'\1', text, flags=re.MULTILINE)
        
        # Strip links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Add title
        final_text = ""
        if title:
            final_text += f"{title}\n"
            final_text += "=" * len(title) + "\n\n"
            
        final_text += text
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
            
        return os.path.abspath(output_path)
