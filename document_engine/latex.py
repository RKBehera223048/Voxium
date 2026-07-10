import os
import re
from typing import Optional
from .exporter import BaseExporter

class LatexExporter(BaseExporter):
    """Exports markdown content to LaTeX."""
    
    def export(self, content: str, output_path: str, title: Optional[str] = None) -> str:
        latex = [
            "\\documentclass{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage{hyperref}",
            "\\usepackage{graphicx}",
            ""
        ]
        
        if title:
            latex.extend([
                f"\\title{{{title}}}",
                "\\begin{document}",
                "\\maketitle",
                ""
            ])
        else:
            latex.extend([
                "\\begin{document}",
                ""
            ])
            
        lines = content.split('\n')
        in_code_block = False
        
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    latex.append("\\end{verbatim}")
                    in_code_block = False
                else:
                    latex.append("\\begin{verbatim}")
                    in_code_block = True
                continue
                
            if in_code_block:
                latex.append(line)
                continue
                
            h1 = re.match(r'^#\s+(.*)$', line)
            if h1:
                latex.append(f"\\section{{{h1.group(1)}}}")
                continue
                
            h2 = re.match(r'^##\s+(.*)$', line)
            if h2:
                latex.append(f"\\subsection{{{h2.group(1)}}}")
                continue
                
            h3 = re.match(r'^###\s+(.*)$', line)
            if h3:
                latex.append(f"\\subsubsection{{{h3.group(1)}}}")
                continue
                
            line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
            line = re.sub(r'\*(.*?)\*', r'\\textit{\1}', line)
            
            if line.startswith('- ') or line.startswith('* '):
                latex.append(f"\\textbullet\\ {line[2:]} \\\\")
                continue
                
            if line.strip() == '':
                latex.append("")
                continue
                
            latex.append(line)
            
        latex.append("\\end{document}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(latex))
            
        return os.path.abspath(output_path)
