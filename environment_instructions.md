# Virtual Enviroment Instructions

## Getting Started (first to-do's):

Install Conda or Miniconda (https://www.anaconda.com/docs/getting-started/miniconda/install#command-prompt) and add it to the system path/environment variables.

1. Make sure the nfl_env.yml file is in your current working directory.
2. Create it using the following: 
  - "conda env create -f nfl_env.yml"
  - Alternatively, you can just include a relative path to the .yml (e.g. conda env create -f Documents/NFL/nfl_env.yml)
3. Activate the environment in your terminal using the following:
  - “conda activate NFL_env"
4. Finally, you have to select your Python interpreter to be from this new environment. This is done whatever code editor you are using. This isn't difficult but is probably best done case-by-case.


## For the future:

For adding/updating the environment in the future: 
1. Activate the package as above
  - "conda activate NFL_env"
2. Install packages you want to add (conda install <insert package name(s) here>)
3. Update the .yml file: 
  - "conda env export -n NFL_env --from-history > nfl_env.yml"
4. Push the new .yml to the github


## For retrieving updates anyone else has made to the environment:

1. Pull newest version of the .yml from the github
2. Update the existing environment on your machine:
  - "conda env update -f nfl_env.yml --prune"





