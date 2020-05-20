find integration_test -name '*test*.py' | while read line; do
    python $line
done
