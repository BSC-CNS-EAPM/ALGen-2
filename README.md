<a id="readme-top"></a>

[![DOI](https://img.shields.io/badge/DOI-10.1038/s42004--025--01635--7-blue.svg)](https://www.nature.com/articles/s42004-025-01635-7)
[![arXiv](https://img.shields.io/badge/arXiv-2305.06334-b31b1b.svg)](https://arxiv.org/abs/2305.06334)
[![arXiv](https://img.shields.io/badge/arXiv-2506.15309-b31b1b.svg)](https://arxiv.org/abs/2506.15309)
[![License](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.7-blue?logo=python)](https://www.python.org/)
[![Last Commit](https://img.shields.io/github/last-commit/IFilella/ALGen-2)](https://github.com/IFilella/ALGen-2)


# ALGen-2

✨**ALGen**✨ is the main component of a molecular generative modeling (GM) workflow designed to generate novel chemical compounds with desired properties. ALGen encompasses the **initial stages** of the workflow illustrated in *Figure 1*, from data preprocessing through molecule generation and the implementation of the inner active learning (AL) cycles.

The pipeline includes the following core stages:

1. 🧠 **Training**  
   A variational autoencoder (VAE) is first trained on a general molecule dataset to learn how to generate chemically valid structures. The model is then fine-tuned on a specific training dataset to focus generation toward molecules with desired affinity to a target.

2. 🧪 **Molecule Generation**  
   After training, the VAE is used to sample new candidate molecules from its latent space.

   The VAE architecture comprised an encoder, a latent space, and a decoder. The input is represented as a 3D tensor resulting from the one-hot encoding of the SMILES sequence. The encoder fed the input into a Long Short-Term Memory (LSTM) layer followed by a fully connected layer with 256 units. This layer further processes the encoded sequence information and reduces its dimensionality into a 128-dimensional latent vector. The decoder reconstructs the sequences, expanding the latent vector through an LSTM layer followed by a fully connected layer with 256 units. The activation function used was ReLu. Finally, a softmax activation layer was applied to produce the output sequence.

3. 🔁 **Inner Active Learning (AL) Cycles**  
   Generated molecules are evaluated for chemical validity and ranked based on drug-likeness (QED), synthetic accessibility (SA), and similarity to the initial-specific training set. Molecules meeting defined thresholds are added to a temporal-specific dataset for subsequent fine-tuning. These AL cycles are repeated iteratively, enabling progressive refinement of the model toward molecules with increasingly desirable properties.



<div style="text-align: center"><img src="images/figure1.png" alt="Alt Text" width=400/></div>


For further technical details and results please refer to the [paper](#citation).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Table of Contents
- [Introduction](#algen-1) 🚀
- [New Features](#new-features) ⚡
- [Installation](#getting-started) 🛠️
- [Usage](#usage) 🧪
- [Configuration](#configuration) ⚙️
- [Examples](#examples) 💡
- [License](#license) 📜
- [Citation](#citation) 📃
- [Contact](#contact) 📩

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## New Features

- 🔬 **Substructural molecular filtering as a first inner Active Learning (AL) step:**  
  Incorporates substructural molecular filters that remove unwanted motifs before applying molecular property filters.

  These substructural filters removes: molecules with any ring equal or greater than 8 atoms, molecules with 3 or more fused rings, molecules with 3/4-atom rings fused with another ring, molecules with 4-atom rings that have more than 2 substitutions, molecules with more than 2 7-membered rings, molecules with consecutive nitrogens in a rings, molecules with more than 4 quiral centers, other manual added SMARTS patterns.

- ⚖️ **Molecular weight added to molecular properties filter:**  
  In addition to Tanimoto similarity, QED, and SA, a **molecular weight range filter** has been added, allowing users to define both minimum and maximum MW limits.
  This is part of the second filter of the inner AL (molecular properties)

- 🔁 **Reversible Tanimoto similarity direction:**  
  A new inverted mode (`">"`) keeps molecules if **at least one tanimoto similarity value exceeds** the threshold. This is useful for lead optimization purposes, were we aim to generate molecules with small variations.

  The regular mode (`"<"`) continues to filter molecules **below** the maximum tanimoto threshold. This mode is useful for promoting diversity/variability.

- 📦 **Catalog filter integration in inner AL:**  
  Users can now enable RDKit **catalog filters** as an optional third filtering stage in the inner AL cycle.
  
  These catalogs are: PAINS, Brenk, NIH, CHEMBL

- 🎯 **Subset-based similarity filtering (optional):**  
  Tanimoto similarity can now be computed against a **user-defined subset** instead of the specific set, enabling guided molecule generation toward user specific chemotypes.

- 🎲 **Fixed seed and separation of the seed used for training and for generation:**  
  Ensures reproducibility.

- 🔥 **Configurable sampling method:**  
  Users may choose between the original **temperature sampling** method or the newly added **top-p (nucleus) sampling** for more controlled generation.

- 🏋️ **Improved training configuration:**  
  Training now supports:
  - **CosineDecayRestarts** learning-rate schedule  
  - **Gradient clipping** in Adam  
  - **ReduceLROnPlateau** for adaptive learning-rate tuning  
  These updates improve stability and convergence on general datasets.

- 🌿 **Trifurcated molecule filter (optional):**  
  Optionally detect and select **trifurcated structures**, with configurable minimum branch counts, for specialized design workflows.

- **High molecular weight filter implementation:**
  TO DO

- 🧱 **(FUTURE IMPLEMENTATION) Scaffold-level similarity filtering:**  
  Future versions may compute Tanimoto similarity based on **scaffolds rather than full molecules**, enabling greater scaffold diversity during generation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation
To install ALGen, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/IFilella/ALGen-2
    cd ALGen-2
    ```

2. **Create virtual environment**:
    ```bash
    conda create -n ALGen python=3.10
    conda activate ALGen
    ```

3. **Install ALGen**:
    ```bash
    make install
    ```


    >**Hardware Requirements**: ALGen is designed to run efficiently on a single GPU. A CUDA-compatible GPU with sufficient RAM is recommended for optimal performance

    Install compatible TensorFlow and CUDA Toolkit.

    First check your CUDA version:
    
    ````
    nvcc --version
    ````
    
    Then, check your CUDA compatibility [here](https://www.tensorflow.org/install/source#gpu) and install correspondingly:
   
    ```
    conda install -c conda-forge tensorflow=x.x            # replace x for your version
    ```

    You can check your installation running this commands on python:
    ```python
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    ```
Others:

4. **Test ALGen**:
    ```bash
    make test
    ```

5. **Uninstall ALGen and clean**:
    ```bash
    make uninstall
    make clean
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage
Once the environment is set up and the package is installed, you can run the generative pipeline using the following command:

```bash
python generator.py \
  -g data/general_set.smi \
  -e 100 \
  -v 0.1 \
  -t 0.1 \
  -b 100 \
  -s 1.2 \
  -q 1000 \
  -n algen_run \
  -o results \
  -pa data/specific_set.smi \
  -ial 10 \
  -qed 0.6 \
  -sa 5 \
  -ta 0.6
```
> **Warning**: The training sets (-g data/general_set.smi, -pa data/specific_set.smi) are used for the initial training of the VAE, not for generating.

For a full list of parameters, see the [Configuration](#configuration) section.

More examples are provided in the [Examples](#examples) section.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

The behavior of ALGen is controlled via command-line arguments. 
Below is a list of configurable parameters and their purpose:

| Argument | Description | Default |
|----------|-------------|---------|
| `-g`, `--general_set_smiles` | Path to the general SMILES file used for training. | *Required* |
| `-e`, `--epochs` | Number of training epochs. | 100 |
| `-v`, `--validation_size` | Proportion of the dataset to be used as the validation set (e.g., `0.1` for 10%). | 0.1 |
| `-t`, `--test_size` | Proportion of the dataset to be used as the test set. | 0.1 |
| `-b`, `--batch_size` | Number of samples per training batch. | 100 |
| `-s`, `--sampling_temperature` | Sampling temperature during generation; controls randomness. Lower values make outputs more deterministic. | 1.2 |
| `-q`, `--quantity` | Target amount defined by --quantity-mode (valid molecules or total attempts). | *Required* |
| `-n`, `--name` | Name of the current generative run. Used to label output files. | *Required* |
| `-o`, `--outdir` | Path to the output directory where results will be saved. | *Required* |
| `-pa`, `--specific_set_smiles` | Path to specific SMILES file for transfer learning. | *Optional* |
| `-pt`, `--pretrained` | Path to a pretrained weights file to fine-tune training. | *None* |
| `-ial`, `--inner_al` | Number of inner active learning (AL) iterations to run. | *Optional* |
| `-qed`, `--druglikeness` | QED threshold (quantitative estimate of drug-likeness) score for generated molecules. | *Optional* |
| `-sa`, `--sascore` | Synthetic accessibility score threshold for generated molecules. | *Optional* |
| `-ta`, `--tanimoto` | Tanimoto similarity threshold for generated molecules. | *Optional* |
| `-tad`, `--ta_direction` | Direction of the Tanimoto threshold filter. | '<' |
| `-tp`, `--top_p` | Nucleus (top-p) sampling cutoff (0-1). Omit to disable nucleus sampling. | *Optional* |
| `-sub`, `--subset_optimizer` | If optimizer, try the subset where to compute Tanimoto similarity. | *Optional* |
| `-mw`, `--molecular_weight` | Filter of molecular weight range, e.g. --mw 20 500. | *Optional* |
| `-r`, `--restart` | Flag to restart a previously interrupted run from the last checkpoint. | *False* |
| `-trifurcated_filter`, `--trifurcated_filter` | Apply trifurcated substructure filter, indicating the min branch size if it is applied (False or int). | *Optional* |
| `-acf`, `--apply_catalog_filter` | Run the PAINS/Brenk/NIH/CHEMBL substructure filter step (skipped by default). | *Optional* |
| `--qed_weights` | Custom QED component weights as 8 floats: `[MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS]`. Overrides the default QED weights. | *Optional* |
| `--seed` | Random seed for reproducible training/generation. Omit to allow stochastic runs. | *Optional* |
| `--scale` | Latent space noise scale factor for generation. | 0.5 |
| `--gen_seed` | Seed for generation only (latent noise + nucleus sampling). | *Optional* |
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Examples

In these examples, ALGen generates 1000 molecules at each inner AL cycle from a total of 10. These are filtered according to the threshold values of 0.6, 5, and 0.6 (QED, SA, and Tanimoto, respectively).


🔹**Molecular Generation from pretrained weights**

Including the `-pt`.
```bash
python generator.py \
  -g data/general_set.smi \
  -e 100 \
  -v 0.1 \
  -t 0.1 \
  -b 100 \
  -s 1.2 \
  -q 1000 \
  -n algen_run \
  -o results \
  -pa data/specific_set.smi \
  -pt data/pretrained_weights.hdf5 \
  -ial 5 \
  -qed 0.6 \
  -sa 5 \
  -ta 0.6
```
Although having the pretrained weights from the training of the general training set, the `-g` flag should also be provided for model sizes and analysis purposes.

🔹**Molecular Generation with training from general training set**

Without the `-pt`.

```bash
python generator.py \
  -g data/general_set.smi \
  -e 100 \
  -v 0.1 \
  -t 0.1 \
  -b 16 \
  -s 1.2 \
  -q 1000 \
  -n algen_run \
  -o results \
  -pa data/specific_set.smi \
  -ial 5 \
  -qed 0.6 \
  -sa 5 \
  -ta 0.6
```

🔹**Restart the Molecular Generation**

Add the `-r` (`-pt` and `-pa` are no longer necessary).

```bash
python generator.py \
  -g data/general_set.smi \
  -e 100 \
  -v 0.1 \
  -t 0.1 \
  -b 100 \
  -s 1.2 \
  -q 1000 \
  -n algen_run \
  -o results \
  -ial 10 \
  -qed 0.6 \
  -sa 5 \
  -ta 0.6
  -r
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). For full licensing terms, please refer to the [LICENSE](LICENSE) file in this repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Citation
If you use this work in your research, please cite the following paper:

> Filella-Merce, I., Molina, A., Díaz, L. et al. Optimizing drug design by merging generative AI with a physics-based active learning framework. Commun Chem 8, 238 (2025). https://doi.org/10.1038/s42004-025-01635-7 

Or its previous version:
> Filella-Merce, I., *et al.* (2023). *Optimizing Drug Design by Merging Generative AI With Active Learning Frameworks.* arXiv:2305.06334. [https://arxiv.org/abs/2305.06334](https://arxiv.org/abs/2305.06334)

This work has also been used in the following scientific works:

> Vilalta-Mor, J., et al. (2025). Active Learning-Guided Seq2Seq Variational Autoencoder for Multi-target Inhibitor Generation. https://arxiv.org/abs/2506.15309

Stay tunned for more publications!


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact
**Júlia Vilalta-Mor**
Email: [julia.vilalta@bsc.es](mailto:julia.vilalta@bsc.es)
Affiliation: Barcelona Supercomputing Center

**Violeta Basten-Romero**
Email: [violeta.basten@bsc.es](mailto:violeta.basten@bsc.es)
Affiliation: Barcelona Supercomputing Center

**Isaac Filella-Mercè**
Email: [isaac.filella1@bsc.es](mailto:isaac.filella1@bsc.es)
Affiliation: Barcelona Supercomputing Center

**Alexis Molina**
Email: [alexis.molina@nostrumbiodiscovery.com](mailto:alexis.molina@nostrumbiodiscovery.com)
Affiliation: Nostrum Biodiscovery S.L.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

##