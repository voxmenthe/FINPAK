#!/bin/bash


# Upgrade pip and install poetry
pip install --upgrade pip
pip install poetry

# Update the lock file if necessary
poetry lock

# Install dependencies and the project
poetry install

# Create and install the IPython kernel for the project
# poetry run python -m python -m ipykernel install --user --name=finpak --display-name "Finpak"
# poetry run python -m ipykernel install --sys-prefix --name=finpak --display-name "Finpak"
python -m ipykernel install --user --name=finpak --display-name "Finpak" # install globally outside of poetry

echo "Jupyter kernel 'Finpak' has been installed."

echo "Project setup complete!"
