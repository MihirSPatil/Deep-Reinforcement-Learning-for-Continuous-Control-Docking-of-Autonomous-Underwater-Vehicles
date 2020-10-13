#!/bin/zsh
export expt_name='euclidean_dist'
export file_path="$HOME/reward_logs"

if [ ! -d "${file_path}" ]; then
  mkdir ${file_path}
fi

if [ ! -f "${file_path}/${expt_name}.json" ]; then
  touch "${file_path}/${expt_name}.json"
fi
