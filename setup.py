from setuptools import setup, find_packages

# Read dependencies from requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="ALGen-2",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    author='Julia Vilalta-Mor, Isaac Filella-Merce, Alexis Molina',
    description='Generative VAE with coupled inner AL cycle',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.7'
)

