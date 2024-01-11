import numpy as np
import random
import modules.encoders as encoders
import modules.decoders as decoders
import modules.classifiers as classifiers
import modules.cres as cres
from torchvision import datasets, transforms
import torch
import config
from torch.functional import F


def get_datasets(dataset_id):
    """
    Retrieves and prepares training and testing datasets based on the specified dataset ID.

    Parameters:
    - dataset_id (str): Identifier for the dataset (e.g., 'MNIST', 'FashionMNIST', 'CIFAR10').

    Returns:
    - tuple: Contains the normalization transform, data augmentation transform,
    training dataset, and testing dataset.

    Depending on the dataset_id, this function initializes the appropriate datasets with
    necessary transforms (normalization and data augmentation). It supports 'MNIST',
    'FashionMNIST', and 'CIFAR10'. For unknown dataset IDs, it raises a NameError.
    """
    if dataset_id == 'MNIST':
        transform_normalization = transforms.Normalize(0.5, 0.5)
        transform_data_augmentation = transforms.Compose([transform_normalization])
        dataset_train = datasets.MNIST(root=config.DATASETS_DIR, train=True, download=True,
                                       transform=transforms.ToTensor())
        dataset_test = datasets.MNIST(root=config.DATASETS_DIR, train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), transform_normalization]))
    elif dataset_id == 'FashionMNIST':
        transform_normalization = transforms.Normalize(0.5, 0.5)
        transform_data_augmentation = transforms.Compose([transform_normalization])
        dataset_train = datasets.FashionMNIST(root=config.DATASETS_DIR, train=True, download=True,
                                              transform=transforms.ToTensor())
        dataset_test = datasets.FashionMNIST(root=config.DATASETS_DIR, train=False, download=True,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(), transform_normalization]))
    elif dataset_id == 'CIFAR10':
        transform_normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform_data_augmentation = transforms.Compose([transforms.RandomRotation(15), transform_normalization])
        dataset_train = datasets.CIFAR10(root=config.DATASETS_DIR, train=True, download=True,
                                         transform=transforms.ToTensor())
        dataset_test = datasets.CIFAR10(root=config.DATASETS_DIR, train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(), transform_normalization]))
    else:
        raise NameError("Unknown dataset id")
    return transform_normalization, transform_data_augmentation, dataset_train, dataset_test


def init_seeds(seed):
    """
    Initializes random seeds for reproducibility.

    Parameters:
    - seed (int or None): Specific seed value or None for random seed.

    Sets seeds for Python's random, NumPy, and PyTorch (including CUDA if available).
    Returns the seed used.
    """
    seed = random.randint(0, 2147483647) if seed is None else seed  # 32-bit integer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If you are using CUDA (GPU), set the seed for that as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def init_optim(module_type: str, optim_state=None, *args, **kwargs):
    """
    Initializes an optimizer from PyTorch's torch.optim.

    Parameters:
    - module_type (str): Optimizer type (e.g., 'Adam').
    - optim_state (dict, optional): State dictionary to load into the optimizer.

    Returns:
    - Optimizer: An instance of the specified optimizer.
    """

    optim = getattr(torch.optim, module_type)(*args, **kwargs)
    if optim_state:
        optim.load_state_dict(optim_state)
    return optim


def init_module(module, module_type: str, *args, **kwargs):
    """
    Initializes a module (e.g., encoder, decoder, classifier) using its type.

    Parameters:
    - module (module): The module from which to create an instance.
    - module_type (str): Type of the module to initialize.

    Returns:
    - Module instance.
    """
    return getattr(module, module_type)(*args, **kwargs)


def init_optims(model, enc_optim_conf, dec_optim_conf, class_optim_conf, enc_optim_state=None,
                dec_optim_state=None, class_optim_state=None):
    """
    Initializes optimizers for encoder, decoder, and classifier of a model.

    Parameters:
    - model: Model containing encoder, decoder, and classifier.
    - *_optim_conf: Configuration dicts for each optimizer.
    - *_optim_state: Optional state dicts to load into each optimizer.

    Returns:
    - Tuple of initialized optimizers (encoder, decoder, classifier).
    """
    optim_enc = init_optim(**enc_optim_conf, params=model.encoder.parameters(), optim_state=enc_optim_state)
    optim_dec = init_optim(**dec_optim_conf, params=model.decoder.parameters(), optim_state=dec_optim_state)
    optim_class = init_optim(**class_optim_conf, params=model.classifier.parameters(), optim_state=class_optim_state)
    return optim_enc, optim_dec, optim_class


def init_model(cre_conf, enc_conf, dec_conf, class_conf, model_state=None):
    """
    Initializes a CRE model with encoder, decoder, and classifier.

    Parameters:
    - *_conf: Configuration dicts for CRE, encoder, decoder, and classifier.
    - model_state (dict, optional): State dictionary to load into the model.

    Returns:
    - Initialized CRE model.
    """

    encoder = init_module(encoders, **enc_conf)
    decoder = init_module(decoders, **dec_conf)
    classifier = init_module(classifiers, **class_conf)
    model = init_module(cres, encoder=encoder, decoder=decoder, classifier=classifier, **cre_conf)
    if model_state:
        model.load_state_dict(model_state)
    return model


def optims_step(optim_enc, optim_dec, optim_class, loss, recons_loss, class_loss):
    """
    Performs an optimization step for encoder, decoder, and classifier optimizers.

    Parameters:
    - optim_*: Optimizers for encoder, decoder, and classifier.
    - loss, recons_loss, class_loss: Computed losses for backward pass.

    Executes backward pass and steps for each optimizer based on the losses.
    """
    optim_dec.zero_grad()
    recons_loss.backward(retain_graph=True)
    optim_class.zero_grad()
    class_loss.backward(retain_graph=True)
    optim_enc.zero_grad()
    loss.backward()
    optim_dec.step()
    optim_class.step()
    optim_enc.step()


def eval_cre(dataloader, model, trade_off, n_eval_batches=float('inf')):
    """
    Evaluate the model using the given dataloader.

    args:
     - dataloader (DataLoader): A DataLoader object containing the dataset to evaluate the model.
     - model (torch.nn.Module): The ViT-based CRE model to be evaluated.
     - trade_off (float): The trade-off parameter to balance between reconstruction and classification loss.
     - n_eval_batches (float, optional): Maximum number of batches to evaluate. Defaults to infinity.


    Returns:
     - float: The average total loss over all evaluated samples.
    """
    model.eval()
    total_mse, total_ce, total_loss, total_acc = 0, 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= n_eval_batches:
                break
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            recons, logits = model(inputs)
            mse_loss = F.mse_loss(recons, inputs)
            ce_loss = F.cross_entropy(logits, labels)
            acc = accuracy(logits, labels)

            loss = trade_off * mse_loss + (1 - trade_off) * ce_loss
            batch_size = len(inputs)
            total_acc += acc * batch_size
            total_mse += mse_loss * batch_size
            total_ce += ce_loss * batch_size
            total_loss += loss * batch_size
            total_samples += batch_size

        print('test/loss: ', (total_loss / total_samples).item())
        print('test/recons_loss: ', (total_mse / total_samples).item())
        print('test/class_loss: ', (total_ce / total_samples).item())
        print('test/accuracy: ', (total_acc / total_samples).item())

    return total_loss / total_samples


def eval_cre_vit(dataloader, model, trade_off, n_eval_batches=float('inf'), mask_ratio=0):
    """
    Evaluate the model using the given dataloader.

     args:
      - dataloader (DataLoader): A DataLoader object containing the dataset to evaluate the model.
      - model (torch.nn.Module): The ViT-based CRE model to be evaluated.
      - trade_off (float): The trade-off parameter to balance between reconstruction and classification loss.
      - n_eval_batches (float, optional): Maximum number of batches to evaluate. Defaults to infinity.
      - mask_ratio (float, optional): The ratio of patches to mask in the input images for masked evaluation.
        A value greater than 0 activates masked evaluation.

      Returns:
      - float: The average total loss over all evaluated samples.
    """
    model.eval()
    model.eval()
    total_mse, total_ce, total_acc, total_loss = 0, 0, 0, 0
    total_mse_masked, total_ce_masked, total_acc_masked, total_loss_masked = 0, 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= n_eval_batches:
                break
            x, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            recons, logits, _ = model(x, mask_ratio=0)
            mse_loss = F.mse_loss(input=model.unpatchify(recons), target=x)
            ce_loss = F.cross_entropy(input=logits, target=labels)
            acc = accuracy(logits=logits, target=labels)
            loss = trade_off * mse_loss + (1 - trade_off) * ce_loss
            batch_size = len(inputs)
            total_acc += acc * batch_size
            total_mse += mse_loss * batch_size
            total_ce += ce_loss * batch_size
            total_loss += loss * batch_size
            total_samples += batch_size

            if mask_ratio > 0:
                recons, logits, mask = model(x, mask_ratio=mask_ratio)
                recons_loss_masked = compute_cre_vit_mask_loss(mask=mask, samples=x, model=model, recons=recons)
                ce_loss_masked = F.cross_entropy(input=logits, target=labels)
                acc_masked = accuracy(logits=logits, target=labels)
                loss_masked = trade_off * recons_loss_masked + (1 - trade_off) * ce_loss_masked
                total_mse_masked += recons_loss_masked * batch_size
                total_ce_masked += ce_loss_masked * batch_size
                total_loss_masked += loss_masked * batch_size
                total_acc_masked += acc_masked * batch_size

    print('test/loss: ', (total_loss / total_samples).item())
    print('test/recons_loss: ', (total_mse / total_samples).item())
    print('test/class_loss: ', (total_ce / total_samples).item())
    print('test/accuracy: ', (total_acc / total_samples).item())
    if mask_ratio > 0:
        print('test/loss_masked: ', (total_loss_masked / total_samples).item())
        print('test/recons_loss_masked: ', (total_mse_masked / total_samples).item())
        print('test/class_loss_masked: ', (total_ce_masked / total_samples).item())
        print('test/accuracy_masked: ', (total_acc_masked / total_samples).item())

    return total_loss / total_samples


def compute_cre_vit_mask_loss(mask, samples, model, recons):
    """
    Calculates masked reconstruction loss for ViT-based CRE model. Computes MSE loss on patches masked by 'mask'.

    Parameters:
    - mask (Tensor): Mask indicating which patches are masked.
    - samples (Tensor): Original image batch.
    - model (torch.nn.Module): ViT-based CRE model for patchifying images.
    - recons (Tensor): Reconstructed images from the model.

    Returns:
    - Tensor: Mean reconstruction loss over masked patches.
    """
    target = model.patchify(samples)
    recons_loss = (recons - target) ** 2
    recons_loss = recons_loss.mean(dim=-1)  # [N, L], mean loss per patch
    recons_loss = (recons_loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return recons_loss


def accuracy(logits, target):
    """
    Calculates accuracy of model predictions.

    Parameters:
    - logits (Tensor): Model's logits.
    - target (Tensor): True labels.

    Returns:
    - Tensor: Proportion of correct predictions.
    """
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


def get_num_patches(img_size, patch_size):
    """
    Determines number of patches in an image based on image and patch sizes.

    Parameters:
    - img_size (tuple): Dimensions of the image (height, width).
    - patch_size (tuple): Dimensions of each patch (height, width).

    Returns:
    - int: Total number of patches.
    """
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    num_patches = grid_size[0] * grid_size[1]
    return num_patches
