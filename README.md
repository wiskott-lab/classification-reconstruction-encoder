# Classification Reconstruction Encoder

## Overview
This repository hosts the official implementation of the paper ["Classification and Reconstruction Processes in 
Deep Predictive Coding Networks: Antagonists or Allies?"](https://arxiv.org/abs/2401.09237) Within this repository, we provide the implementation
of the primary tool discussed in the paper, the Classification Reconstruction Encoder (CRE), 
along with the necessary training scripts.

## Requirements
- Python 3.9.6.
- timm 0.9.5
- torch 2.0.1 
- torchvision 0.15.2
- numpy 1.25.1

## Installation
Clone the repository to your local machine and install the requirements.

## Usage
To train a FC-based or CNN-based CRE, run the training script train_cre.py. To train a ViT-based CRE, run the training
script train_cre_vit.py. Model and training parameters can be changed in both training scripts.

## Citation
'''
@Article{rathjens2024classification,
      title={Classification and Reconstruction Processes in Deep Predictive Coding Networks: Antagonists or Allies?}, 
      author={Jan Rathjens and Laurenz Wiskott},
      year={2024},
      eprint={2401.09237},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
'''
## Contact
For questions and feedback, contact us at [jan.rathjens@rub.de](mailto:jan.rathjens@rub.de).

## License
This project is under the BSD-3-Clause license. See [LICENSE](LICENSE) for details.


