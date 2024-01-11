import numpy as np
import random
import config
import modules.encoders as encoders
import modules.decoders as decoders
import modules.classifiers as classifiers
import modules.cres as cres
import neptune
import yaml
import uuid
from copy import deepcopy
from sklearn.decomposition import PCA
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import os


def get_datasets(dataset_id):
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


def transform_for_vit(batch, n_patches, embed_dim, add_class_token=True):
    if add_class_token:
        batch = batch.view(batch.shape[0], n_patches, -1)
        rnd_class_token = torch.randn(size=(batch.shape[0], 1, embed_dim), device=config.DEVICE)
        batch = torch.cat((rnd_class_token, batch), dim=1)
    else:
        batch = batch.view(batch.shape[0], n_patches, -1)
    return batch


def make_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


def init_seeds(seed):
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
    optim = getattr(torch.optim, module_type)(*args, **kwargs)
    if optim_state:
        optim.load_state_dict(optim_state)
    return optim


def init_module(module, module_type: str, *args, **kwargs):
    return getattr(module, module_type)(*args, **kwargs)


def init_optims(model, enc_optim_conf, dec_optim_conf, class_optim_conf, enc_optim_state=None,
                dec_optim_state=None, class_optim_state=None):
    optim_enc = init_optim(**enc_optim_conf, params=model.encoder.parameters(), optim_state=enc_optim_state)
    optim_dec = init_optim(**dec_optim_conf, params=model.decoder.parameters(), optim_state=dec_optim_state)
    optim_class = init_optim(**class_optim_conf, params=model.classifier.parameters(), optim_state=class_optim_state)
    return optim_enc, optim_dec, optim_class


def init_model(cre_conf, enc_conf, dec_conf, class_conf, model_state=None):
    encoder = init_module(encoders, **enc_conf)
    decoder = init_module(decoders, **dec_conf)
    classifier = init_module(classifiers, **class_conf)
    model = init_module(cres, encoder=encoder, decoder=decoder, classifier=classifier, **cre_conf)
    if model_state:
        model.load_state_dict(model_state)
    return model


def get_checkpoint(run):
    params = run["params"].fetch()
    tmp_path = str(config.TMP_DIR / str(uuid.uuid4()))
    run['checkpoint'].download(str(tmp_path))
    return params, tmp_path


def rollback(run_id, proj_id):
    # TODO
    run = neptune.init_run(with_id=run_id, project=proj_id, capture_hardware_metrics=False,
                           monitoring_namespace='monitoring')
    raise NotImplementedError


# def init_checkpoint(run_id, proj_id, model_path, checkpoint_path):
#     run = neptune.init_run(with_id=run_id, project=proj_id)
#     params = run["params"].fetch()
#     model = init_model_from_params(params)
#     run['model_state_dict'].download(str(model_path))
#     run['checkpoint'].download(str(checkpoint_path))
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     enc_optim, dec_optim, class_optim = init_optims_from_params(params, model)
#     load_optims_state_dicts(enc_optim, dec_optim, class_optim, checkpoint)
#     best_loss = checkpoint['best_loss']
#     epoch = checkpoint['epoch']
#
#     return model, run, enc_optim, dec_optim, class_optim, best_loss, epoch


def get_params(run_id: str, proj_name=config.PROJECT):
    local_run_path = make_local_run_path(run_id)
    local_params_path = local_run_path / "params.yaml"
    if not local_params_path.exists():
        run = neptune.init_run(with_id=run_id, project=proj_name, mode='read-only')
        params = run["params"].fetch()
        with open(str(local_params_path), "w") as f:
            yaml.dump(params, f)
        run.stop()
    with open(str(local_params_path), 'r') as f:
        params = yaml.safe_load(f)
    return params


def get_model_state(run_id: str, proj_name=config.PROJECT):
    local_run_path = make_local_run_path(run_id)
    local_model_state_path = local_run_path / 'model_state.pt'
    print(local_model_state_path)
    if not local_model_state_path.exists():
        run = neptune.init_run(with_id=run_id, project=proj_name, mode='read-only')
        run['model_state_dict'].download(str(local_model_state_path))
        run.stop()
    model_state_dict = torch.load(str(local_model_state_path), map_location=torch.device(config.DEVICE))
    return model_state_dict


def optims_step(optim_enc, optim_dec, optim_class, loss, recons_loss, class_loss):
    optim_dec.zero_grad()
    recons_loss.backward(retain_graph=True)
    optim_class.zero_grad()
    class_loss.backward(retain_graph=True)
    optim_enc.zero_grad()
    loss.backward()
    optim_dec.step()
    optim_class.step()
    optim_enc.step()


def save_checkpoint(model, optim_enc, optim_dec, optim_class, best_loss, test_step=None, train_step=None):
    tmp_path = str(config.TMP_DIR / str(uuid.uuid4()))
    torch.save({
        'model_state_dict': deepcopy(model.state_dict()),
        'optim_state_enc': deepcopy(optim_enc.state_dict()),
        'optim_state_dec': deepcopy(optim_dec.state_dict()),
        'optim_state_class': deepcopy(optim_class.state_dict()),
        'best_loss': deepcopy(best_loss),
        'train_step': train_step,
        'test_step': test_step
    }, tmp_path)
    return tmp_path


def init_model_from_params(params):
    encoder = init_module(encoders, **params['enc_config'])
    decoder = init_module(decoders, **params['dec_config'])
    classifier = init_module(classifiers, **params['class_config'])
    model = init_module(cres, encoder=encoder, decoder=decoder, classifier=classifier, **params['cre_config'])
    model.to(config.DEVICE)
    return model


# def init_optims_from_params(params, model):
#     return init_optims(enc_optim_conf=params['enc_optim_config'], dec_optim_conf=params['dec_optim_config'],
#                        class_optim_conf=params['class_optim_config'], model=model)
#
# #
# def load_optims_state_dicts(optim_enc, optim_dec, optim_class, checkpoint):
#     optim_enc.load_state_dict(checkpoint['optim_state_enc'])
#     optim_dec.load_state_dict(checkpoint['optim_state_dec'])
#     optim_class.load_state_dict(checkpoint['optim_state_class'])


def init_model_from_neptune(run_id: str, proj_name=config.PROJECT):
    params = get_params(run_id=run_id, proj_name=proj_name)
    model = init_model_from_params(params)
    model_state = get_model_state(run_id=run_id, proj_name=proj_name)
    model.load_state_dict(model_state)
    return model


def make_local_run_path(run_id):
    local_run_path = config.RUNS_DIR / run_id
    local_run_path.mkdir(exist_ok=True)
    return local_run_path


def get_pca_loaders(n_dim, dataset_id):
    transform = transforms.Compose([transforms.ToTensor(), transforms.transforms.Normalize(0.5, 0.5)])

    if dataset_id == 'MNIST':
        data_train = datasets.MNIST(root=config.DATASETS_DIR, train=True, transform=transform, download=True)
        data_test = datasets.MNIST(root=config.DATASETS_DIR, train=False, transform=transform, download=True)
    else:
        data_train = datasets.FashionMNIST(root=config.DATASETS_DIR, train=True, transform=transform, download=True)
        data_test = datasets.FashionMNIST(root=config.DATASETS_DIR, train=False, transform=transform, download=True)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=len(data_train), shuffle=False)

    loader_test = torch.utils.data.DataLoader(data_test, batch_size=len(data_test), shuffle=False)

    train_input, train_targets = next(iter(loader_train))
    train_input = train_input.view(len(data_train), -1).numpy()

    pca = PCA(n_components=n_dim)
    train_input_pca = pca.fit_transform(train_input)
    pca_dataset_train = TensorDataset(torch.from_numpy(train_input_pca), train_targets)
    pca_train_loader = DataLoader(pca_dataset_train, batch_size=512, shuffle=True, drop_last=True)

    test_input, test_targets = next(iter(loader_test))
    test_input = test_input.view(len(data_test), -1).numpy()
    test_input_pca = pca.transform(test_input)
    pca_dataset_test = TensorDataset(torch.from_numpy(test_input_pca), test_targets)
    pca_test_loader = DataLoader(pca_dataset_test, batch_size=512, shuffle=True, drop_last=True)

    return pca_train_loader, pca_test_loader


def get_rp_loaders(n_dim, dataset_id, batch_size=512):
    transform = transforms.Compose([transforms.ToTensor(), transforms.transforms.Normalize(0.5, 0.5)])

    if dataset_id == 'MNIST':
        data_train = datasets.MNIST(root=config.DATASETS_DIR, train=True, transform=transform, download=True)
        data_test = datasets.MNIST(root=config.DATASETS_DIR, train=False, transform=transform, download=True)
    else:
        data_train = datasets.FashionMNIST(root=config.DATASETS_DIR, train=True, transform=transform, download=True)
        data_test = datasets.FashionMNIST(root=config.DATASETS_DIR, train=False, transform=transform, download=True)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=len(data_train), shuffle=False)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=len(data_test), shuffle=False)

    train_input, train_targets = next(iter(loader_train))
    train_input = train_input.view(len(data_train), -1)

    rp_matrix = torch.randn(train_input.shape[1], n_dim)
    rp_matrix /= rp_matrix.norm(dim=0)

    rp_data_train = torch.mm(train_input, rp_matrix)
    rp_dataset_train = TensorDataset(rp_data_train, train_targets)
    rp_train_loader = DataLoader(rp_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)

    test_input, test_targets = next(iter(loader_test))
    test_input = test_input.view(len(data_test), -1)
    rp_data_test = torch.mm(test_input, rp_matrix)

    rp_dataset_test = TensorDataset(rp_data_test, test_targets)
    rp_test_loader = DataLoader(rp_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    return rp_train_loader, rp_test_loader

import torch
import config
from torch.functional import F


def eval_cre(dataloader, model, trade_off, n_eval_batches=float('inf')):
    """
    Evaluate the model using the given dataloader.

    Args:
    - dataloader: Data loader for evaluation.
    - model: The neural network model to evaluate.
    - trade_off: Trade-off parameter between reconstruction and classification loss.
    - run: Optional logging object for tracking metrics.
    - n_eval_batches: Number of batches to evaluate (default is all batches).

    Returns:
    - Average loss over the evaluated batches.
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
    target = model.patchify(samples)
    recons_loss = (recons - target) ** 2
    recons_loss = recons_loss.mean(dim=-1)  # [N, L], mean loss per patch
    recons_loss = (recons_loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return recons_loss


def accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


def get_num_patches(img_size, patch_size):
    grid_size = tuple([s // p for s, p in zip(img_size, patch_size)])
    num_patches = grid_size[0] * grid_size[1]
    return num_patches



if __name__ == '__main__':
    get_rp_loaders(n_dim=3, dataset_id='MNIST')
