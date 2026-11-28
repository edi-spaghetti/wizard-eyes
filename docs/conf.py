# Sphinx configuration for the Wizard Eyes documentation.
# Build (from the repo root):
#   python -m pip install -r docs/requirements.txt
#   python -m sphinx -W -b html docs docs/_build/html

project = 'Wizard Eyes'
author = 'edi-spaghetti'
copyright = '2026, edi-spaghetti'

extensions = []
templates_path = []
exclude_patterns = ['_build']

html_theme = 'furo'
html_title = 'Wizard Eyes'
html_static_path = []

# PowerShell examples throughout
highlight_language = 'powershell'
