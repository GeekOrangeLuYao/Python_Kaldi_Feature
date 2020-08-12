# shellcheck disable=SC2155
export Python_Kaldi_Feature=`pwd`
export PYTHONPATH=$Python_Kaldi_Feature:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/base:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/feature:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/featurebin:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/matrix:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/util:$PYTHONPATH
export PYTHONPATH=$Python_Kaldi_Feature/audio:$PYTHONPATH

echo "add path finished"
