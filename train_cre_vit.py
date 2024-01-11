from torch.utils.data import DataLoader
import torch
import config
from torch.functional import F
from utils import init_seeds, optims_step, init_optims, init_model, get_datasets, get_num_patches, eval_cre_vit, \
    compute_cre_vit_mask_loss
from copy import deepcopy
import uuid

# script for training a ViT-based CRE

if __name__ == '__main__':
    # Setting hyperparameters for the model and training
    epochs, embed_dim, trade_off, batch_size = 10, 8, 0.4, 512
    mask_ratio = 0.0
    masked_loss = 'v1'
    img_size, patch_size, in_chans, num_classes = 28, 7, 1, 10
    enc_num_heads, dec_num_heads = 4, 4
    enc_embed_dim, dec_embed_dim = embed_dim, embed_dim
    enc_depth, dec_depth = 1, 1
    enc_mlp_ratio, dec_mlp_ratio = 4.0, 4.0
    lr_e, lr_d, lr_c, dropout_prob = 0.001, 0.001, 0.001, 0.0  # learning rates
    dataset_id = 'MNIST'
    model_path = 'test_model_' + str(uuid.uuid4()) + '.pt'  # Path for saving the trained model

    # setup for ViT-based CRE components
    cre_conf = {'module_type': 'CREViT'}
    enc_config = {'module_type': 'EncoderViT', 'img_size': img_size, 'in_chans': in_chans,
                  'patch_size': patch_size,
                  'embed_dim': enc_embed_dim, 'num_heads': enc_num_heads, 'mlp_ratio': enc_mlp_ratio,
                  'depth': enc_depth}
    dec_config = {'module_type': 'DecoderViT', 'embed_dim': enc_embed_dim, 'decoder_embed_dim': dec_embed_dim,
                  'num_patches': get_num_patches((img_size, img_size), (patch_size, patch_size)),
                  'decoder_num_heads': dec_num_heads, 'mlp_ratio': dec_mlp_ratio, 'decoder_depth': dec_depth,
                  'patch_size': patch_size, 'in_chans': in_chans}
    class_config = {'module_type': 'ClassifierViT', 'embed_dim': enc_embed_dim, 'num_classes': num_classes}

    # Configuration for the optimizers.
    enc_optim_config = {'module_type': 'Adam', 'lr': lr_e}
    dec_optim_config = {'module_type': 'Adam', 'lr': lr_d}
    class_optim_config = {'module_type': 'Adam', 'lr': lr_c}
    seed = init_seeds(None)  # Initialize random seeds

    transform_normalization, transform_data_augmentation, dataset_train, dataset_test = get_datasets(dataset_id)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    model = init_model(cre_conf, enc_config, dec_config, class_config)
    model.to(config.DEVICE)

    optim_enc, optim_dec, optim_class = init_optims(model, enc_optim_config, dec_optim_config,
                                                    class_optim_config)

    best_loss = 2147483647  # Initialize best loss for model saving criterion

    print('---Epoch 0---')
    eval_cre_vit(dataloader=dataloader_test, model=model, trade_off=trade_off,
                 mask_ratio=mask_ratio)  # eval model before training

    # Training loop
    for epoch in range(epochs):
        print('---Epoch ', epoch + 1, '----')
        model.train()
        for batch_id, (inputs, labels) in enumerate(dataloader_train):
            samples, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            recons, logits, mask = model(transform_data_augmentation(samples), mask_ratio=mask_ratio)
            if mask_ratio > 0 and masked_loss == 'v1':  # compute loss according to mask ratio and variant
                recons_loss = compute_cre_vit_mask_loss(mask=mask, samples=transform_normalization(samples),
                                                        model=model, recons=recons)
            else:
                recons_loss = F.mse_loss(input=model.unpatchify(recons), target=transform_normalization(samples))
            class_loss = F.cross_entropy(input=logits, target=labels)
            loss = trade_off * recons_loss + (1 - trade_off) * class_loss
            optims_step(optim_enc, optim_dec, optim_class, loss, recons_loss, class_loss)

        loss = eval_cre_vit(dataloader=dataloader_test, model=model,
                            trade_off=trade_off, mask_ratio=mask_ratio)  # eval model after every epoch

        if loss < best_loss:
            torch.save(deepcopy(model.state_dict()), model_path)  # Save the model if it's the best so far
            best_loss = loss
