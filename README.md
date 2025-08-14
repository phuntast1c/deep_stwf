# Binaural Speech Enhancement

This repository contains supplementary code for the paper "[Imposing Correlation Structures for Deep Binaural Spatio-Temporal Wiener Filtering](https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/SP2023_6.pdf?v=1743258194)" by M. Tammen and S. Doclo, IEEE Trans. Audio, Speech and Language Processing, vol. 33, pp. 1278-1292, 2025.

## Installation

1.  Clone this repository to your local machine.
2.  Make sure Anaconda or Miniconda are available.
3.  Create and activate the conda environment using the provided `environment.yml` file. We recommend using Mamba for a faster installation.
    ```bash
    # If you don't have mamba, install it first:
    # conda install -n base -c conda-forge mamba
    mamba env create -f environment.yml
    conda activate j2
    ```


## Usage

You can use the `inference.py` script to enhance a noisy audio file using one of the pretrained models mentioned in the paper.

### Example

To run the inference script, use a command like the following:

```bash
python inference.py --model stwf_noCommonSTCM_noRTF --input data/noisy.wav --output data/noisy_enhanced.wav
```

This command will:
-   Load the `stwf_noCommonSTCM_noRTF` model.
-   Process the `data/noisy.wav` file.
-   Save the enhanced audio to `data/noisy_enhanced.wav`.

If you don't specify an output file, the enhanced audio will be saved in the same directory as the input file with `_enhanced` appended to the name.

### Available Models

The following models are available for use with the `--model` argument (see Table II in the paper):
- `stwf_noCommonSTCM_noRTF`
- `stwf_CommonSTCM_noRTF`
- `stwf_CommonSTCM_globalRTF`
- `stwf_CommonSTCM_ipsiRTF`
- `stwf_bilat_CommonSTCM_noRTF`
- `stwf_bilat_CommonSTCM_global`
- `df_noRTF`
