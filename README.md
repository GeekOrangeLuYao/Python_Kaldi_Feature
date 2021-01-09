# Python_Kaldi_Feature
The Kaldi Feature Written by Python

## Extractor Feature
Please see the `featurebin/`.

You should change the `conf/` and write a `.ini` file as a config file.
Note that you can write in several settings into one `.ini`
and you can use `config_section` to 

You should first prepare the `wav.scp` for the project
which is similar with `Kaldi`
> **Note that the project do not support the pipe data for extracting features!**
>
> **I will add this function in the following version**

The project will write to some `*.ark` files and a `feats.scp`.
The features the project get will not be compressed,
however, you can use the feature_reader code to read the kaldi feature.
> The `Kaldi` features compress is irreversible which means you cannot get the features before compressing,
> if you use `wsj/steps/make_mfcc.sh`, etc., the `compress` parameter is always true unless you change it,
> there're **no** plan to add compress functions).

## Project Structure and Details

### `Pytorch` API
Please see the `torch_feature/` and use the `feature_kernels`.
The torch-api for computing the feature more quickly and
you can insert the feature-extractor into the model,
and you can even train them as I provide the `requires_grad`! 

### Difference between `Kaldi`
1. **DO NOT** realize the `dither` for each,
you can see the function but we will not use that,
which may cause the great difficulty for debugging.

2. **DO NOT** do the `vltn` part in our projects,
so when we will get the single instance for `mel_banks`

3. Use more value-call for the functions instead of
reference-call, which is simpler for python programming.
So you can see there're less `assert` to check the dim problems.


## To-do List
1. Add pitches
2. Merge more functions into one function, such as computer power spectrogram
3. Add PLP features
4. Use Subprocess to let the `wav` can be a `pipe data`
5. Add multiprocess part to improve the effectiveness
6. More useful verbose and logging code
7. `Tensorflow` support
8. Compare with `torchaudio`

Please give me more suggestions and help improvement!
You can email [me](mailto:OrangeLuyao@outlook.com) and ask any questions in issue!

## Reference
* [Kaldi](http://www.kaldi-asr.org/)
* The kaldi-io code is from [kaldi-python-io](https://github.com/funcwj/kaldi-python-io)
