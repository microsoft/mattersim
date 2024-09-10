# MatterSim Documentations

## Prerequisite
To compile the documentations on your local machine you will need to install
the following dependency in your environment:
```bash
# sphinx
pip install sphinx sphinx-autodoc-typehints sphinx_book_theme sphinx-copybutton

# enable Markdown documentation in sphinx
pip install recommonmark

# enable python jupyter notebook in sphinx
pip install nbsphinx nbconvert

# install pandoc
conda install -c conda-forge pandoc
```

## Compile the docs
Under the root of this repo, execute the following commandline
```bash
sphinx-build -b html docs docs/_build
```
To browse the documentation on your local machine, you may start a minimal
HTTP server with
```bash
python3 -m http.server --directory docs/_build 8000
```
In a web browser, e.g. Chrome or Edge, you can read the docs at [localhost at port 8000](http://localhost:8000).
