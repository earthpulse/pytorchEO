#from sphinx.setup_command import BuildDoc
#from packaging.version import parse
import setuptools
#assert parse(setuptools.__version__) >= parse('36.2')
#cmdclass = {'build_sphinx': BuildDoc}

name = 'pytorch_eo'
version = '22.02'
release = '22.02.25'
author = 'earthpulse'
description = 'DL4EO'
email = 'it@earthpulse.es'
url = 'https://github.com/earthpulse/pytorch_eo'
keywords = ['deep learning', 'earth observation',
            'neural networks', 'pytorch', 'pytorch lightning']

setuptools.setup(
    name=name,
    packages=setuptools.find_packages(),
    version=release,
    license='MIT',
    description=description,
    author=author,
    author_email=email,
    url=url,
    keywords=keywords,
    install_requires=['numpy', 'torch >= 1.4', 'torchvision', 'pytorch_lightning',
                      'rasterio', 'scikit-image', 'scikit-learn', 'albumentations', 'einops', 'pandas'],
    python_requires='>=3.6',
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Development Status :: 3 - Alpha',
        # Define that your audience are developers
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # cmdclass=cmdclass,
    # command_options={
    #     'build_sphinx': {
    #         'project': ('setup.py', name),
    #         'version': ('setup.py', version),
    #         'release': ('setup.py', release),
    #         'source_dir': ('setup.py', 'sphinx/source')}}
)
