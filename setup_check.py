"""Check dependency imports and requirements fingerprint before launching."""
import hashlib, sys, importlib, subprocess
from pathlib import Path
from importlib.metadata import version
root=Path(__file__).resolve().parent
mark=Path(sys.prefix)/'bargcc_requirements.sha256'
digest=hashlib.sha256((root/'requirements.txt').read_bytes()).hexdigest()
try:
    from packaging.specifiers import SpecifierSet
    if '--mark' not in sys.argv and (not mark.exists() or mark.read_text().strip()!=digest):sys.exit(1)
    for m in ('streamlit','plotly','numpy','pandas','scipy','sklearn','threadpoolctl','PIL','pypdf','pytesseract'):importlib.import_module(m)
    installed = version('streamlit')
    if installed not in SpecifierSet('>=1.51,<2'):
        raise RuntimeError(f'Streamlit {installed} is unsupported; install Streamlit >=1.51,<2.')
    if '--mark' in sys.argv:
        subprocess.run([sys.executable,'-m','pip','check'],check=True);mark.write_text(digest)
except Exception as e:print(f'Dependency check: {e}');sys.exit(1)
print(f'Dependencies ready (Streamlit {installed}); no installation needed.')
