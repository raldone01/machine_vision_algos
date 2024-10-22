#!/bin/fish

# Make sure we are in the parent directory of this script
cd (dirname (status -f))
cd ../

# echo "Working directory: " (pwd)

if test -d .conda
  echo "Updating existing environment"
  conda env update --file python/conda_env.yml --prefix .conda --prune
else
  echo "Creating new environment"
  conda env create --file python/conda_env.yml --prefix .conda
end
