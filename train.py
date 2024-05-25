"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
import torch
from tqdm import tqdm
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    
    opt = TrainOptions().parse()   # get training options

    # get train and valid dataset
    opt.phase = "train"
    train_dataset = create_dataset(opt)
    print(f'The number of training images = {len(train_dataset)}')
    opt.phase = "val"
    valid_dataset = create_dataset(opt)
    print(f'The number of validation images = {len(valid_dataset)}')

    # get generative model
    model = create_model(opt)      
    model.setup(opt)               

    # get visualizer that display/save images and plots
    visualizer = Visualizer(opt)   
    
    train_total_iters = 0           
    valid_total_iters = 0           
    max_psnr = 0.0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

        # ready for training
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        train_epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        valid_epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        # traning
        # log
        log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        for i, data in tqdm(enumerate(train_dataset)):  # inner loop within one epoch
            if (data["A"].max() == -1. and data["B"].max() == -1.):
                iter_data_time = time.time()
            else:
                iter_start_time = time.time()  # timer for computation per iteration
                
                if train_total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                train_total_iters += data["A"].size(0)
                train_epoch_iter += data["A"].size(0)
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                # visualize generated images
                if train_total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = train_total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # display loss log
                losses = model.get_current_losses()
                if train_total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / data["A"].size(0)
                    visualizer.print_current_losses(epoch, train_epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(train_epoch_iter) / len(train_dataset), losses)
                
                iter_data_time = time.time()

        # valid
        # log
        log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Validation Loss (%s) ================\n' % now)
        
        valid_psnr = 0.0           # when this loss is minimum, the model is saved

        if opt.model == "cycle_gan":
            current_valid_loss = {"D_A": 0.0, "G_A": 0.0, "cycle_A": 0.0, "idt_A": 0.0, "D_B": 0.0, "G_B": 0.0, "cycle_B": 0.0, "idt_B": 0.0}
        elif opt.model == "pix2pix":
            current_valid_loss = {"G_GAN": 0.0, "G_L1": 0.0, "D_real": 0.0, "D_fake": 0.0}

        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_dataset)):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if (data["A"].max() == -1. and data["B"].max() == -1.):
                    iter_data_time = time.time()
                else:
                    if train_total_iters % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    valid_total_iters += data["A"].size(0)
                    valid_epoch_iter += data["A"].size(0)
                    model.set_input(data)         # unpack data from dataset and apply preprocessing
                    model.test()   # calculate loss functions, get gradients, update network weights

                    # visualize generated images
                    if valid_total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                        save_result = valid_total_iters % opt.update_html_freq == 0
                        model.compute_visuals()
                        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                    # display loss log
                    losses = model.get_current_losses()
                    if valid_total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / data["A"].size(0)
                        visualizer.print_current_losses(epoch, valid_epoch_iter, losses, t_comp, t_data)
                        if opt.display_id > 0:
                            visualizer.plot_current_losses(epoch, float(valid_epoch_iter) / len(valid_dataset), losses)

                    iter_data_time = time.time()
                    if opt.model == "cycle_gan":
                        current_valid_loss["D_A"] += losses['D_A']
                        current_valid_loss["G_A"] += losses['G_A']
                        current_valid_loss["cycle_A"] += losses['cycle_A']
                        current_valid_loss["idt_A"] += losses['idt_A']
                        current_valid_loss["D_B"] += losses['D_B']
                        current_valid_loss["G_B"] += losses['G_B']
                        current_valid_loss["cycle_B"] += losses['cycle_B']
                        current_valid_loss["idt_B"] += losses['idt_B']
                        valid_psnr += model.calc_psnr()
                    
                    elif opt.model == "pix2pix":
                        current_valid_loss["G_GAN"] += losses['G_GAN']
                        current_valid_loss["G_L1"] += losses['G_L1']
                        current_valid_loss["D_real"] += losses['D_real']
                        current_valid_loss["D_fake"] += losses['D_fake']
                        valid_psnr += model.calc_psnr()

        valid_psnr /= len(valid_dataset)
        print(max_psnr)
        print(valid_psnr)
        if max_psnr < valid_psnr:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, training iters %d. valid iters %d' % (epoch, train_total_iters, valid_total_iters))
            model.save_networks('best')
            max_psnr = valid_psnr
            # log
            log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Model Saved (%s) ================\n' % now)
                log_file.write(f'max psnr: {max_psnr}\n')

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
