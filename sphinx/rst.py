import sys
import os
from glob import glob


def save(name, text):
    # save some text in a file
    text_file = open(name, "w")
    text_file.write(text)
    text_file.close()


# get the folder to read files from and the destination folder for the rst files
folder = sys.argv[1]
dest = sys.argv[2]

# this is the text that will appear at the top of the index.rst file
index = f"""
.. {folder} documentation master file

=========================================
Welcome to {folder}'s documentation!
=========================================

this is the readme of my project

.. toctree::
   :maxdepth: 2
   :caption: Contents

"""

# generate rst files for all .py files in a folder
excluded_files = ['__init__.py']
for f in os.walk(folder):
    # read folder, subfolders and files
    subfolder = f[0]
    subfolders = f[1]
    files = f[2]

    # continue if not pycache
    if subfolder[-11:] != '__pycache__':
        # keep last folder name
        name = subfolder.split('/')[-1]

        # keep only .py files
        files = [f for f in files if f[-3:] ==
                 '.py' and f not in excluded_files]

        # generate the text to be saved in the rst file
        text = f"{name}\n===================\n"
        for f in files:
            filename = f[:-3]
            text += f"""
{filename}
----------------
.. automodule:: {'.'.join(subfolder.split('/'))+'.'+filename}
   :members:
            """
        save(f"{dest}/{name}.rst", text)

        # add this file to the index.rst
        index += f"""
   {name}"""

# this text willl be placed at the bottom of the index.rst file
index += f"""

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
save(f"{dest}/index.rst", index)
