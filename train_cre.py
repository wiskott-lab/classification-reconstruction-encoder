from torch.utils.data import DataLoader
import torch
import config
from torch.functional import F
from utils import init_seeds, optims_step, init_optims, init_model, get_datasets, eval_cre
from copy import deepcopy
import uuid

# script for training a FC-based or CNN-based CRE

if __name__ == '__main__':
    # Setting hyperparameters for the model and training
    epochs, latent_dim, trade_off, batch_size = 10, 4, 0.9, 512
    lr_e, lr_d, lr_c, dropout_prob = 0.001, 0.001, 0.001, 0.0
    dataset_id = 'MNIST'  #
    model_path = 'test_model_' + str(uuid.uuid4()) + '.pt'  # Path for saving the trained model

    # setup for FC-based CRE components
    cre_config = {'module_type': 'CRE'}
    enc_config = {'module_type': 'EncoderFCMedium', 'latent_dim': latent_dim}
    dec_config = {'module_type': 'DecoderFCMedium', 'latent_dim': latent_dim}
    class_config = {'module_type': 'ClassifierFCMedium', 'latent_dim': latent_dim}

    # alternative setup for CNN-based CRE components
    # cre_config = {'module_type': 'CRE'}
    # enc_config = {'module_type': 'EncoderCNNSmall', 'dropout_prob': dropout_prob}
    # dec_config = {'module_type': 'DecoderCNNSmall'}
    # class_config = {'module_type': 'ClassifierCNNSmall'}

    # Configuration for the optimizers
    enc_optim_config = {'module_type': 'Adam', 'lr': lr_e}
    dec_optim_config = {'module_type': 'Adam', 'lr': lr_d}
    class_optim_config = {'module_type': 'Adam', 'lr': lr_c}

    seed = init_seeds(None)  # Initialize random seeds

    transform_normalization, transform_data_augmentation, dataset_train, dataset_test = get_datasets(dataset_id)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    model = init_model(cre_config, enc_config, dec_config, class_config)
    optim_enc, optim_dec, optim_class = init_optims(model, enc_optim_config, dec_optim_config,
                                                    class_optim_config)

    best_loss = 2147483647  # Initialize best loss for model saving criterion
    model.to(config.DEVICE)

    print('---Epoch 0---')
    eval_cre(dataloader=dataloader_test, model=model, trade_off=trade_off)  # eval model before training

    # Training loop
    for epoch in range(epochs):
        print('---Epoch ', epoch + 1, '----')
        model.train()
        for batch_id, (inputs, labels) in enumerate(dataloader_train):
            samples, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            recons, logits = model(transform_data_augmentation(samples))
            recons_loss = F.mse_loss(input=recons, target=transform_normalization(samples))
            class_loss = F.cross_entropy(input=logits, target=labels)
            loss = trade_off * recons_loss + (1 - trade_off) * class_loss
            optims_step(optim_enc, optim_dec, optim_class, loss, recons_loss,
                        class_loss)
        loss = eval_cre(dataloader=dataloader_test, model=model,
                        trade_off=trade_off)
        if loss < best_loss:
            torch.save(deepcopy(model.state_dict()), model_path)  # Save the model if it's the best so far
            best_loss = loss
