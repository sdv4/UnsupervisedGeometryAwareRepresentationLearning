import torch
import numpy as np
import os
from utils import datasets as utils_data
from utils import plot_dict_batch as utils_plot_batch

import sys


sys.path.insert(0,'./ignite')
from ignite.engine.engine import Engine, State, Events

import matplotlib.image as mpimg
import IPython
import pickle


# optimization function
def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred
    engine = Engine(_update)
    return engine

def create_adversarial_trainer(netG, netD, optimizerG, optimizerD, loss_fn_gen, loss_fn_dis, device=None):
    """

    Args:
        netG: generator model
        netD: discriminator model
        optimizerG: optimization function

    """
    def _update(engine, batch):
        """

        Args:
            engine (engine.Engine):
            batch (list): tuple with dictionary at each position. Index 0 contains input images, rotation info, and
                        target background. Index 1 contains target input images (i.e. rotated images), 3D joint positions,
                        bounding TODO give better/full description of the contents
        Returns:

        """
        real_label = 1 # mark data as real
        fake_label = 0 # mark data as fake

        # update discriminator network

        # train with all real batch
        netD.zero_grad()
        real_data, _ = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
        b_size = batch[0]['img_crop'].size(0)

        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_data['img_crop']).view(-1)
        #print("batch[0] keys: ", batch[0].keys())
        #print("batch[0]['img_crop'] shape: ", batch[0]['img_crop'].shape)
        #print("batch[0]['extrinsic_rot'] shape: ", batch[0]['extrinsic_rot'].shape)
        #print("batch[1] keys: ", batch[1].keys())
        error_D_real = loss_fn_dis(output, label)
        error_D_real.backward()
        D_x = output.mean().item()

        # train with all fake batch
        netG.train() # TODO why?
        input_imgs, rotated_tgt_imgs = utils_data.nestedDictToDevice(batch, device=device) # TODO will this work for non-random input?
        fake = netG(input_imgs)
        label.fill_(fake_label)
        output = netD(fake['img_crop'].detach()).view(-1)
        errD_fake = loss_fn_dis(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        optimizerD.step()

        # update generator network
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake['img_crop']).view(-1)
        errG = loss_fn_gen(output, label) # as in tutorial
        """        
        x, y = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        """
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        return errG.item(), fake
    engine = Engine(_update)
    return engine

def create_supervised_evaluator(model, metrics={}, device=None):
    def _inference(engine, batch):  
        # now compute error
        model.eval()
        with torch.no_grad():
            x, y = utils_data.nestedDictToDevice(batch, device=device) # make it work for dict input too
            y_pred = model(x)
            
        return y_pred, y        

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def save_training_error(save_path, engine, vis, vis_windows):
    # log training error
    iteration = engine.state.iteration - 1
    loss, pose = engine.state.output
    print("Epoch[{}] Iteration[{}] Loss: {:.2f}".format(engine.state.epoch, iteration, loss))
    title="Training error"
    if vis is not None:
        vis_windows[title] = vis.line(X=np.array([engine.state.iteration]), Y=np.array([loss]),
                 update='append' if title in vis_windows else None,
                 win=vis_windows.get(title, None),
                 opts=dict(xlabel="# iteration", ylabel="loss", title=title))
    # also save as .txt for plotting
    log_name = os.path.join(save_path, 'debug_log_training.txt')
    if iteration ==0:
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, loss))


def save_testing_error(save_path, trainer, evaluator, vis, vis_windows):
    metrics = evaluator.state.metrics
    iteration = trainer.state.iteration
    print("Validation Results - Epoch: {}  Avg accuracy: {}".format(trainer.state.epoch, metrics))
    accuracies = []
    for key in metrics.keys():
        title="Testing error {}".format(key)
        avg_accuracy = metrics[key]
        accuracies.append(avg_accuracy)
        if vis is not None:
            vis_windows[title] = vis.line(X=np.array([iteration]), Y=np.array([avg_accuracy]),
                         update='append' if title in vis_windows else None,
                         win=vis_windows.get(title, None),
                         opts=dict(xlabel="# iteration", ylabel="value", title=title))
    # also save as .txt for plotting
    log_name = os.path.join(save_path, 'debug_log_testing.txt')
    if iteration ==0:
        with open(log_name, 'w') as the_file: # overwrite exiting file
            the_file.write('#iteration,loss1,loss2,...\n')     
    with open(log_name, 'a') as the_file:
        the_file.write('{},{}\n'.format(iteration, ",".join(map(str, accuracies)) ))
    return sum(accuracies)


def save_training_example(save_path, engine, vis, vis_windows, config_dict):
    # print training examples
    iteration = engine.state.iteration - 1
    loss, output = engine.state.output
    inputs, labels = engine.state.batch
    mode='training'
    img_name = os.path.join(save_path, 'debug_images_{}_{:06d}.jpg'.format(mode, iteration))
    utils_plot_batch.plot_iol(inputs, labels, output, config_dict, mode, img_name)
    #img = misc.imread(img_name)
    if vis:
        img = mpimg.imread(img_name)
        title="Training example"
        vis_windows[title] = vis.image(img.transpose(2,0,1), win=vis_windows.get(title, None),
             opts=dict(title=title+" (iteration {})".format(iteration)))


def save_test_example(save_path, trainer, evaluator, vis, vis_windows, config_dict):
    iteration_global = trainer.state.iteration
    iteration = evaluator.state.iteration - 1
    inputs, labels = evaluator.state.batch
    output, gt = evaluator.state.output # Note, comes in a different order as for training
    mode='testing_{}'.format(iteration_global)
    img_name = os.path.join(save_path, 'debug_images_{}_{:06d}.jpg'.format(mode,iteration))
    utils_plot_batch.plot_iol(inputs, labels, output, config_dict, mode, img_name)               
    if vis is not None:
        img = mpimg.imread(img_name)
        title="Testing example"+" (test iteration {})".format(iteration)
        vis_windows[title] = vis.image(img.transpose(2,0,1), win=vis_windows.get(title,None),
                                    opts=dict(title=title+" (training iteration {})".format(iteration_global)))


def load_model_state(save_path, model, optimizer, state):
    model.load_state_dict(torch.load(os.path.join(save_path,"network_best_val_t1.pth")))
    optimizer.load_state_dict(torch.load(os.path.join(save_path,"optimizer_best_val_t1.pth")))
    sate_variables = pickle.load(open(os.path.join(save_path,"state_last_best_val_t1.pickle"),'rb'))
    for key, value in sate_variables.items(): setattr(state, key, value)
    print('Loaded ',sate_variables)


def save_model_state(save_path, engine, current_loss, model, optimizer, state):
    # update the best value
    best_val = engine.state.metrics.get('best_val', 99999999)
    engine.state.metrics['best_val'] = np.minimum(current_loss, best_val)
    
    print("Saving last model")
    model_path = os.path.join(save_path,"models/")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model.state_dict(), os.path.join(model_path,"network_last_val.pth"))
    torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_last_val.pth"))
    state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
    pickle.dump(state_variables, open(os.path.join(model_path,"state_last_val.pickle"),'wb'))
    
    if current_loss==engine.state.metrics['best_val']:
        print("Saving best model (previous best_loss={} > current_loss={})".format(best_val, current_loss))
        
        torch.save(model.state_dict(), os.path.join(model_path,"network_best_val_t1.pth"))
        torch.save(optimizer.state_dict(), os.path.join(model_path,"optimizer_best_val_t1.pth"))
        state_variables = {key:value for key, value in engine.state.__dict__.items() if key in ['iteration','metrics']}
        pickle.dump(state_variables, open(os.path.join(model_path,"state_best_val_t1.pickle"),'wb'))

# Fix of original Ignite Loss to not depend on single tensor output but to accept dictionaries
from ignite.metrics import Metric


class AccumulatedLoss(Metric):
    """
    Calculates the average loss according to the passed loss_fn.
    `loss_fn` must return the average loss over all observations in the batch.
    `update` must receive output of the form (y_pred, y).
    """
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(AccumulatedLoss, self).__init__(output_transform)
        self._loss_fn = loss_fn

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output
        average_loss = self._loss_fn(y_pred, y)
        assert len(average_loss.shape) == 0, '`loss_fn` did not return the average loss'
        self._sum += average_loss.item() * 1 # HELGE: Changed here from original version
        self._num_examples += 1 # count in number of batches

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples
    
    
def transfer_partial_weights(state_dict_other, obj, submodule=0, prefix=None, add_prefix=''):
    """
    Transfer pre-trained weights from a given state dictionary to a given model of the same class.

    :param state_dict_other: the state_dict from the saved model
    :param obj: the model whose weights will be loaded from state_dict_other
    :param submodule:
    :param prefix:
    :param add_prefix:
    :return: None, as all operations are done to obj in place
    """
    print('Transferring weights...')

    own_state = obj.state_dict()
    own_encoder_state = 0
    own_decoder_state = 0
    # TODO: refactor once no longer using legacy code
    if hasattr(obj, 'encoder'):
        own_encoder_state = obj.encoder.state_dict()
    if hasattr(obj, 'decoder'):
        own_decoder_state = obj.decoder.state_dict()
    copy_count = 0
    skip_count = 0
    paramCount = len(own_state)

    def _copy_weights(state, name, param):
        """

        :param state: list of parameter names
        :param name: the name of the parameter to be loaded
        :param param: the parameter values for name
        :return: copy or skip; the counter to increments
        """
        copy = 0
        if hasattr(state[name], 'copy_'):  # isinstance(own_state[name], torch.Tensor):
            # print('copy_ ',name)
            if state[name].size() == param.size():
                state[name].copy_(param)
                copy = 1
            else:
                print(
                    'Invalid param size(own={} vs. source={}), skipping {}'.format(own_state[name].size(), param.size(),
                                                                                   name))
        elif hasattr(own_state[name], 'copy'):
            own_state[name] = param.copy()
            copy = 1
        else:
            print('training.utils: Warning, unhandled element type for name={}, name_raw={}'.format(name, name_raw))
            print(type(own_state[name]))
            IPython.embed()

        if copy:
            return 'copy'
        else:
            return 'skip'


    for name_raw, param in state_dict_other.items():
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if prefix is not None and not name_raw.startswith(prefix):
            #print("skipping {} because of prefix {}".format(name_raw, prefix))
            continue
        
        # remove the path of the submodule from which we load
        name = add_prefix+".".join(name_raw.split('.')[submodule:])

        if name in own_state:
            inc_me = _copy_weights(own_state, name, param)
            if inc_me == 'copy':
                copy_count += 1
            else:
                skip_count += 1
        elif own_encoder_state != 0 and name in own_encoder_state:
            inc_me = _copy_weights(own_encoder_state, name, param)
            if inc_me == 'copy':
                copy_count += 1
            else:
                skip_count += 1
        elif own_decoder_state != 0 and name in own_decoder_state:
            inc_me = _copy_weights(own_decoder_state, name, param)
            if inc_me == 'copy':
                copy_count += 1
            else:
                skip_count += 1
        else:
            skip_count += 1
            print('Warning, no match for {}, ignoring'.format(name))
            # print(' since own_state.keys() = ',own_state.keys())
            
    print('Copied {} elements, {} skipped, and {} target params without source'.format(copy_count, skip_count, paramCount-copy_count))
