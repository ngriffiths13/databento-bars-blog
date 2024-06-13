# Databento Bars Blogposts

This repository contains the notebook to do the analysis for the blogposts I wrote with Databento.

To setup the environment, you will want to create a venv and install the requirements:

To install the requirements, run:

`pip install -e .`

If you are using rye, you can run the following command to install the requirements:

`rye sync`

Then with the virtual environment activated you can run:

`marimo edit src/databento_bars_blog/db_blog.py`

The notebook will not work out of the box because the data files it attempts to read from are not included in this repository. You will have to bring your own data for the analysis.
