This file contains instructions for repo setup.

1. Ensure that no conda environment is active with `conda deactivate`
2. `conda create -n compressor python=3.7`
3. `conda activate compressor`
4. `pip install -r original_codebase/requirements.txt`
5. `pip check` to see if there are any broken requirements

Since Python 3.7 is not configured for arm64, follow the next instructions to ensure compatibility:

1. Install Miniforge for MacOSX-x86_64 through: https://github.com/conda-forge/miniforge/releases/tag/26.5.3-0
2. Install Rosetta
3. Start a shell with x86_64 with `arch -x86_64 zsh`
4. Verify with `uname -m` > `x86_64`
5. Ensure that no conda environment is active with `conda deactivate`
6. Create a conda environment with the Miniforge install with `[filepath]/miniforge3/bin/conda create -n compressor python=3.7`.
7. Activate the environment with `source [filepath]/miniforge3/bin/activate compressor`