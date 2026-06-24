import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "QALMA"
copyright = "2025, QILPCM-IFLP-CONICET"
author = "QILPCM-IFLP-CONICET"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.imgconverter",  # Automatically handles image format conversions
]

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if os.path.isdir("_static") else []
html_logo = "logo.svg"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}


# -- Autodoc -----------------------------------------------------------------

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,
}

# -- Napoleon (NumPy docstrings) ---------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Todo --------------------------------------------------------------------

todo_include_todos = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "qutip": ("https://qutip.readthedocs.io/en/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "never"  # set to "auto" once RTD has ALPS installed
nbsphinx_allow_errors = False

# -- MathJax -----------------------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"\left|#1\right\rangle", 1],
            "bra": [r"\left\langle#1\right|", 1],
            "tr": r"\mathrm{Tr}",
            "expect": [r"\left\langle#1\right\rangle", 1],
        }
    }
}


# SyntaxWarning from LaTeX strings in notebook cell outputs (Python 3.12+)
import warnings

# Register ipython3 as an alias for ipython so nbsphinx notebooks render correctly
from pygments.lexers import get_lexer_by_name  # noqa: E402
from sphinx.highlighting import lexers  # noqa: E402

try:
    lexers["ipython3"] = get_lexer_by_name("ipython")
except Exception:
    pass

# SyntaxWarning from LaTeX strings in notebook cell outputs (Python 3.12+)
warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
)
