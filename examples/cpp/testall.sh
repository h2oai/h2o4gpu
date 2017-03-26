rm -rf error.txt out.txt
time ./run 2>> error.txt 1>> out.txt
grep "END SOLVE" out.txt
