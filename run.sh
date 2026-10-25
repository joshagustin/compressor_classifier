#!/bin/zsh
echo "arch: $(uname -m)"

source /Users/josh/miniforge3/etc/profile.d/conda.sh 
conda activate compressor
echo "python: $(which python)"

BASE=(python original_codebase/main_text.py --k 5 --runs 3 --cos --ncd --report_dir clean_reports/R52/cos_ncd --dataset R52 --data_dir data/datasets/R52)

$BASE --no_normalize --no_remove_stopwords --no_stem
$BASE --no_remove_stopwords --no_stem
$BASE --no_stem
$BASE --no_remove_stopwords
$BASE

NEWS20=(python original_codebase/main_text.py --k 5 --runs 3 --cos --ncd --report_dir clean_reports/20News/cos_ncd --dataset 20News)

$NEWS20 --no_remove_stopwords
$NEWS20

OHSUMED=(python original_codebase/main_text.py --k 5 --runs 3 --cos --ncd --report_dir clean_reports/Ohsumed/cos_ncd --dataset Ohsumed --data_dir data/datasets/Ohsumed)

$OHSUMED --no_stem
$OHSUMED

# chmod +x run.sh

# log everything
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > run_log.txt 2>&1 &

# log nothing
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2>&1 &

# log only errors
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2> errors.txt &