for log_f in $(ls); do echo "${log_f}: "; cat $log_f | grep -- '- Acc' | awk 'BEGIN {max=0;}; {if ($8 > max){max=$8;}} END {print max}'; done
