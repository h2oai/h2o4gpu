LOGEXT=$1

mkdir -p ./tmp/
rm -rf ./tmp/h2oai-test.$LOGEXT.log
tests=`ls testsbig/test*.py`
#echo "list of tests: $tests"
echo "LOG FILE: ./tmp/h2o4gpu-testbig.$LOGEXT.log"
for test in $tests
do
    echo "Doing $test"
	H2OGLM_DISABLEPYTEST=1 H2OGLM_PERFORMANCE=1 python $test  &>> ./tmp/h2o4gpu-testbig.$LOGEXT.log
done

