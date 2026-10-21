#!/bin/zsh
echo "arch: $(uname -m)"

source /Users/josh/miniforge3/etc/profile.d/conda.sh 
conda activate compressor
echo "python: $(which python)"

BASE=(python original_codebase/main_text.py --k 5 --runs 3 --ncd --report_dir clean_reports/AG_News/ncd)

$BASE --no_normalize --no_remove_stopwords --no_stem 
$BASE --no_remove_stopwords --no_stem 
$BASE --no_stem 

# chmod +x run.sh

# log everything
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > run_log.txt 2>&1 &

# log nothing
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2>&1 &

# log only errors
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2> errors.txt &