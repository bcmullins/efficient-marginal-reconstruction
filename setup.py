from setuptools import setup, find_packages

setup(
    name='rem',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',
        'linear_operator',
        'tqdm',
        'pandas',
        'scipy'
    ],
    author='Brett Mullins',
    author_email='bmullins@umass.edu',
    description='This repo implements the methods from "Efficient and Private Marginal Reconstruction with Local Non-Negativity" (2024)',
    url='https://github.com/bcmullins/efficient-marginal-reconstruction',
    python_requires='>=3.10',
)