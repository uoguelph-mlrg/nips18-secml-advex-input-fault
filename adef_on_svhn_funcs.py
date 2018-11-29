import os
import torch
from torch.autograd import Variable
import scipy
import numpy as np
from cnn_model_pytorch import CNN, custom_init_params


def setup_model(num_classes, num_filters, num_channels, do_bn, model_file):
    """Build the network and load saved model."""
    net = CNN(num_classes, num_filters, num_channels, do_bn)
    net.apply(custom_init_params)

    if os.path.exists( model_file ):
        net.load_state_dict( torch.load(model_file, map_location=lambda storage, loc: storage) )
        print('* * * loaded model from %s * * *'%model_file)
    else:
        print('Model not found! \nGoing to deform images w.r.t. an untrained model!')
        verbose = False
    #print('Model: ' + str(type(net)))
    return net
    

def get_sample(seed, path_to_dataset, batch_size, save_dir, identifyer=None):
    """Load the test dataset, get a random sample (batch) of images, 
       save sample images and corresponding labels as npz. """

    # ------  Load the test dataset
    test_file_img = 'prepr_test_images.npz'
    test_file_lbs = 'test_labels.npz'
    test_x = np.load(path_to_dataset + test_file_img)['a']
    test_y = np.load(path_to_dataset + test_file_lbs)['a']

    #svhn_test = torch.utils.data.TensorDataset(test_x[:-16], test_y[:-16])
    #test_loader = torch.utils.data.DataLoader(dataset=svhn_test,
    #                                          batch_size=batch_size, shuffle=True, pin_memory=False)

    # ------  Get a random sample (batch) of images
    #batch, labels = next(iter( test_loader ))
    np.random.seed(seed)
    indx_arr = np.random.choice(len(test_y), batch_size, replace=False)
    batch, labels = torch.FloatTensor(test_x[indx_arr].astype('float32')), torch.LongTensor(test_y[indx_arr])
    np.savez_compressed(save_dir + 'orig_images'+identifyer+'.npz', a=test_x[indx_arr])
    np.savez_compressed(save_dir + 'orig_labels'+identifyer+'.npz', a=test_y[indx_arr])

    return batch, labels


def check_misclassified(net, batch, labels):
    """Check for misclassified digits in the selected sample,
       return an array with predicted labels."""
    x = Variable( batch )
    Fx = net.forward(x)
    maxval, pred_labels = torch.max( Fx.data, 1 )

    if (pred_labels != labels).any():
        num_misclassf = np.sum( pred_labels.numpy()!=labels.numpy() )
        print('There are %i misclassified example/s.'%num_misclassf )

    return pred_labels.numpy()


def plot_and_save_imgs(batch_size, batch, def_batch, vector_fields, pred_labels, labels, path_to_model):
    """Plot and save original imgs and deformed imgs."""
    fig, axs = plt.subplots( 2, batch_size )
    fig.set_size_inches(28,8)
    for im_no in range(batch_size):
        im = batch[ im_no, 0 ].numpy()
        def_im = def_batch[ im_no, 0 ].numpy()

        axs[ 0, im_no ].imshow( im, cmap='Greys', vmin=0, vmax=1 )
        draw_vector_field( axs[ 0, im_no ], vector_fields[ im_no ], amp=3 )
        if not pred_labels[im_no] == labels[im_no]:
            axs[ 0, im_no ].set_title( 'Misclf. as %d' % pred_labels[im_no], color='red' )
        else:
            axs[ 0, im_no ].set_title( '%d' % pred_labels[im_no] )
        axs[ 1, im_no ].imshow( def_im, cmap='Greys', vmin=0, vmax=1 )
        axs[ 1, im_no ].set_title( '%d' % def_labels[im_no] )

    axs[0,0].set_ylabel('Original')
    axs[1,0].set_ylabel('Deformed')

    plt.savefig(path_to_model+'svhn_max_norm_%.1f.png'%max_norm, dpi=80, bbox_inches='tight')
    plt.close()


def load_batch_and_labels(path_to_model, file_name_prefix, iter_type=None, iter_val=None):
    """Load saved batch of (deformed) images and the corresponding labels."""

    if file_name_prefix=='orig':
        file_name_batch  = file_name_prefix + '_images'+iter_type+'.npz'
        file_name_labels = file_name_prefix + '_labels'+iter_type+'.npz'
    elif file_name_prefix=='adef':
        file_name_batch  = file_name_prefix + '_images'+iter_type+'_%.1f.npz'%iter_val
        file_name_labels = file_name_prefix + '_labels'+iter_type+'_%.1f.npz'%iter_val       
    else:
        print('error - not a valid prefix!')

    batch  = np.load(path_to_model + file_name_batch)['a']
    labels = np.load(path_to_model + file_name_labels)['a']

    return batch, labels

def load_batch(path_to_model, file_name_prefix, iter_type=None, iter_val=None):
    """Load saved batch of images."""

    if file_name_prefix=='orig':
        file_name  = file_name_prefix + '_images'+iter_type+'.npz'

    elif file_name_prefix=='adef' and iter_type=='_norm':
        file_name  = file_name_prefix + '_images'+iter_type+'_%.1f.npz'%iter_val

    elif file_name_prefix=='adef' and iter_type=='_iter':
        file_name  = file_name_prefix + '_images'+iter_type+'_%i.npz'%iter_val

    else:
        print('error - fix load_batch func!')

    batch  = np.load(path_to_model + file_name)['a']
    return batch
    

def load_def_data(path_to_model, file_name_prefix, iter_type=None, iter_val=None):
    """Load saved def_data for the batch."""
    if iter_type=='_norm':
        file_name_data   = file_name_prefix + '_data' +iter_type+'_%.1f.npz'%iter_val
    elif iter_type=='_iter':
        file_name_data   = file_name_prefix + '_data' +iter_type+'_%i.npz'%iter_val        
    else:
        print('error - fix load_batch func!')

    data = scipy.load(path_to_model+file_name_data)['a'][()]

    return data


def evaluate_acc(model,batch,labels,device='cpu'): # former: evaluate
    """Evaluate model's prediction accuracy on given batch of data."""
    images = torch.FloatTensor(batch).to(device)
    labels = torch.LongTensor(labels).to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc=(correct/total)
    return acc


def get_prediction(model, batch, device):
    """Get predicted labels for given input batch and model."""
    images = torch.tensor(batch, dtype=torch.float).to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    return predicted
