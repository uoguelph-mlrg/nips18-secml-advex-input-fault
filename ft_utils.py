import numpy as np
import torch


def snr(S, N):
    '''
    Returns the signal to noise ratio for
    @param S: the signal
    @param N: the noise
    '''
    temp = 20 * np.log10(1 + np.linalg.norm(np.squeeze(S), axis=(1, 2)) /
                         np.linalg.norm(np.squeeze(N), axis=(1, 2)))
    # filter inf values
    return np.mean(temp[np.invert(np.isinf(temp))])


def evaluate_acc(model, images, labels):
    """Evaluate model's prediction accuracy on given batch of data."""
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc=(correct/total)
    return acc


def arr21hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def mi(T, Y, num_classes=10):
    """
    Computes the mutual information I(T; Y) between predicted T and true labels Y
    as I(T;Y) = H(Y) - H(Y|T) = H_Y - H_cond_YgT
    @param T: vector with dimensionality (num_instances,)
    @param Y: vector with dimensionality (num_instances,)
    @param num_classes: number of classes, default=10
    """
    Y = Y.detach().cpu().numpy()
    T = T.detach().cpu().numpy()

    epsilon = 1e-4 # to prevent divide by zero
    num_instances = Y.shape[0]
    py    = np.zeros(num_classes) # p(y)
    pt    = np.zeros(num_classes) # p(t)
    pygt  = np.zeros(num_classes) # p(y|t)
    H_YgT = np.zeros(num_classes) # H(Y|T)

    # Compute H(Y)
    for i in range(num_classes):
        py[i] = np.sum(Y == i) / float(num_instances)
        pt[i] = np.sum(T == i) / float(num_instances)
        
    H_Y = -np.dot( py, np.log2(py + epsilon) ) # H(Y)

    # Compute H(Y | T)
    for t in range(num_classes):
        t_idx = T == t 
        for y in range(num_classes):
            y_idx = Y == y
            pygt[y] = np.sum(y_idx[t_idx])

        # convert counts to probabilities
        c = np.sum(pygt)
        if c > 0:
            pygt /= c
            H_YgT[t] = -np.dot( pygt, np.log2(pygt + epsilon) )
    
    H_cond_YgT = np.dot( pt, H_YgT )

    return H_Y - H_cond_YgT


def pygt(labels_o, labels_p, num_classes=10):
    """
    Computes the conditional probability p(y|t)
    for true label or targeted label y given predicted label t.
    @param labels_o: true (original) labels
    @param labels_p: predicted labels
    @param num_classes: number of classes, default=10
    """
    labels_o = labels_o.detach().cpu().numpy()
    labels_p = labels_p.detach().cpu().numpy()

    epsilon = 1e-4 # to prevent divide by zero
    num_instances = labels_o.shape[0]
    pygt = np.zeros(( num_classes,num_classes )) # p(y|t)

    # Compute p(y|t)
    for t in range(num_classes):
        t_idx = labels_p == t
        for y in range(num_classes):
            y_idx = labels_o == y
            pygt[y,t] = np.sum(y_idx[t_idx])

        # convert counts to probabilities
        c = np.sum(pygt[:,t])
        if c > 0:
            pygt[:,t] /= c

    return pygt


def build_targeted_dataset(X_test, Y_test, indices, num_classes, device):
    """
    Build a dataset for targeted attacks, each source image is repeated num_classes-1 times, 
    and target labels are assigned that do not overlap with true label.
    :param X_test: clean source images
    :param Y_test: true labels for X_test
    :param indices: indices of source samples to use
    :param num_classes: number of classes in classification problem
    """
    num_samples = len(indices)
    num_target_classes = num_classes - 1

    X = X_test[indices]
    Y = Y_test[indices]
    img_shape = np.array(X.shape[1:])

    adv_inputs  = np.repeat(X, num_target_classes, axis=0)
    true_labels = np.repeat(Y, num_target_classes, axis=0)
    adv_inputs  = torch.FloatTensor(adv_inputs).to(device)
    true_labels = torch.LongTensor(true_labels).to(device)

    a = np.repeat([np.arange(num_classes)], len(Y), axis=0)
    target_labels = torch.LongTensor(a[a != np.array(Y)[:, None]]).to(device)
    
    
    return adv_inputs, true_labels, target_labels



def build_targeted_dataset_1hot(X_test, Y_test, indices, num_classes):
    """
    Build a dataset for targeted attacks, each source image is repeated num_classes-1 times, 
    and target labels are assigned that do not overlap with true label.
    :param X_test: clean source images
    :param Y_test: true labels for X_test, in 1-hot format
    :param indices: indices of source samples to use
    :param num_classes: number of classes in classification problem
    """
    num_samples = len(indices)
    num_target_classes = num_classes - 1

    X = X_test[indices]
    Y = Y_test[indices]
    img_shape = np.array(X.shape[1:])

    adv_inputs = np.repeat(X, num_target_classes, axis=0)
    #dims = tuple(np.hstack((num_samples * num_target_classes, img_shape)))
    #adv_inputs = adv_inputs.reshape((dims))

    true_labels_1hot = np.repeat(Y, num_target_classes, axis=0)
    #dims = tuple(np.hstack((num_samples * num_target_classes, num_classes)))
    #true_labels = true_labels.reshape((dims))

    diag = np.eye(num_target_classes)
    target_labels_1hot=np.zeros((1,num_classes))
    for pos in np.argmax(Y, axis=1):
        target_labels_1hot=np.vstack((target_labels_1hot, np.insert(diag, pos, 0, axis=1) ))
    target_labels_1hot=target_labels_1hot[1:]

    return adv_inputs, true_labels_1hot, target_labels_1hot


def evaluate(sess, training, acc, loss, x_, y_, x_np, y_np, feed=None):
    feed_dict = {x_: x_np, y_: y_np, training: False}
    if feed is not None:
        feed_dict.update(feed)
    return sess.run([acc, loss], feed_dict)


# Init result var
def evaluate_model(sess, training, acc, loss, x_, data_x, y_, data_y, batch_size):

    nb_examples = data_x.shape[0]
    nb_batches = int(np.ceil(float(nb_examples) / batch_size))
    #print('nb_batches=%d' % nb_batches)
    assert nb_batches * batch_size >= nb_examples
    loss_np = 0.
    accuracy_np = 0.
    for test_batch in range(nb_batches):
        start = test_batch * batch_size
        end = min(nb_examples, start + batch_size)
        cur_batch_size = end - start
        batch_xs = data_x[start:end]
        batch_ys = data_y[start:end]
        cur_acc, cur_loss = evaluate(sess, training, acc, loss,
                                     x_, y_, batch_xs, batch_ys)
        accuracy_np += (cur_batch_size * cur_acc)
        loss_np += (cur_batch_size * cur_loss)
    accuracy_np /= nb_examples
    loss_np /= nb_examples
    return accuracy_np, loss_np


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix
