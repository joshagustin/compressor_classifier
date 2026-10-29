#!/bin/zsh
echo "arch: $(uname -m)"

source /Users/josh/miniforge3/etc/profile.d/conda.sh 
conda activate compressor
echo "python: $(which python)"

BASE=(python original_codebase/main_text.py --k 5 --runs 5 --dataset Ohsumed --data_dir data/datasets/Ohsumed)

$BASE --report_dir reports/Ohsumed/ncd --ncd
sleep 300
$BASE --report_dir reports/Ohsumed/ncd --ncd --no_normalize --no_stem --no_remove_stopwords
sleep 300
$BASE --report_dir reports/Ohsumed/ncd --ncd --no_stem --no_remove_stopwords
sleep 300
$BASE --report_dir reports/Ohsumed/ncd --ncd --no_stem
sleep 300
$BASE --report_dir reports/Ohsumed/ncd --ncd --no_remove_stopwords
sleep 300

$BASE --report_dir reports/Ohsumed/cos --cos_sim 
sleep 300

$BASE --report_dir reports/Ohsumed/cos_ncd --ncd --cos_sim
sleep 300
$BASE --report_dir reports/Ohsumed/cos_ncd --ncd --cos_sim --no_normalize --no_stem --no_remove_stopwords
sleep 300
$BASE --report_dir reports/Ohsumed/cos_ncd --ncd --cos_sim --no_stem --no_remove_stopwords
sleep 300
$BASE --report_dir reports/Ohsumed/cos_ncd --ncd --cos_sim --no_stem
sleep 300
$BASE --report_dir reports/Ohsumed/cos_ncd --ncd --cos_sim --no_remove_stopwords

# chmod +x run.sh

# log everything
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > run_log.txt 2>&1 &

# log nothing
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2>&1 &

# log only errors
# nohup env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL arch -x86_64 /bin/zsh run.sh > /dev/null 2> errors.txt &