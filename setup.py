from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="FINPAK",
    version="0.1.0",
    packages=find_packages(where="finpak"),
    package_dir={"": "finpak"},
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            # Add your command line scripts here
        ],
    },
)
