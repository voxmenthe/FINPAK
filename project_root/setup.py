from setuptools import setup, find_packages

setup(
    name="financial_analysis",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            # Add your command line scripts here
        ],
    },
)
