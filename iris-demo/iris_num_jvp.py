import os
import logging
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from inimsolve import InImNetNumGrad

num_epochs = 1000

# Strictly increasing, non-empty; p_points[-1] = q
p_train = (0., 1.)
p_test = (0., 1.)
double_mlp_on = True
triple_mlp_on = False
inflation_factor = 4
bias_on = True
test_activation = nn.ReLU()
integration_method = 'dopri5'
leaning_rate = torch.tensor(0.01)
test_freq = 1  # Check accuracy every test_freq epochs

# Chose from integration_method:
# Adaptive: 'dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun'
# Fixed point: 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'
# Backwards compatibility: 'fixed_adams', 'scipy_solver'

# Loss regularisation
use_loss_regularisation = False
regularisation_coeff = 0.0001
use_gradient_clipping = False
grad_clipping_value = 100

# Plotting options
save_plots = True
view_plots = False
plot_loss_and_success_on = True

# Logging initiation
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
logger.addHandler(logging.StreamHandler())

# Import data
iris = load_iris()
x_raw_inputs = iris['data']
y_targets = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# Scale data to mean 0 and variance 1
scaler = StandardScaler()
x_inputs = scaler.fit_transform(x_raw_inputs)

# Split data into training and testing sets of type 'numpy.ndarray'
xtrain, xtest, ytrain, ytest = train_test_split(
    x_inputs, y_targets, test_size=0.2, random_state=2
)


class InImClassifier(InImNetNumGrad):
    """Choose output method for model."""

    def __init__(self, num_target_classes, **kwargs):
        super(InImClassifier, self).__init__(**kwargs)
        self.num_classes = num_target_classes

    def output(self, zq):
        nfeats = torch.tensor(zq.shape[:-1]).prod()
        out = torch.sigmoid(zq.view(nfeats, -1)[:, :self.num_classes])
        return out.view(*zq.shape[:-1], -1)


# Instantiate InImNet model
inimodel = InImClassifier(num_target_classes=3,
                          input_dim=xtrain.shape[1],
                          batch_dim=xtrain.shape[0],
                          cost_fn=nn.MSELoss(reduction='mean'),
                          activation=test_activation,
                          double_mlp=double_mlp_on,
                          triple_mlp=triple_mlp_on,
                          dim_multiplier=inflation_factor,
                          bias_on=bias_on,
                          integration_method=integration_method,
                          )

# Parameter optimisation
optimiser = torch.optim.Adam(inimodel.parameters(), lr=leaning_rate)

# Transform [0,1,2] to [[1,0,0], [0,1,0], [0,0,1]]
enc = OneHotEncoder()
ytrain = enc.fit_transform(np.array(ytrain.flatten())[:, np.newaxis]).toarray()
ytest = enc.fit_transform(np.array(ytest.flatten())[:, np.newaxis]).toarray()

# Move data from Numpy to Torch
xtrain = torch.tensor(xtrain).float()                       # Size([120, 4])
ytrain = torch.tensor(ytrain).float()                       # Size([120, 3])
xtest = torch.tensor(xtest).float()                         # Size([30, 4])
ytest = torch.tensor(ytest).float()                         # Size([30, 3])
ytest_argmax = torch.argmax(ytest, 1)  # Unencoded output, values in 0, 1, 2

# Update logger
logger.info('Size of training set: {}'.format(xtrain.shape[0]))
logger.info('Size of testing set: {}'.format(xtest.shape[0]))
logger.info('Input dimension: {}'.format(xtest.shape[1]))
logger.info('Input names:\n\t {} \n\t {} \n\t {} \n\t {}'
            .format(feature_names[0],
                    feature_names[1],
                    feature_names[2],
                    feature_names[3]))
logger.info('Target number: {}'.format(len(names)))
logger.info('Target names: {}'.format(names))
logger.info(inimodel)
logger.info('Number of parameters: {}'.format(
    sum(p.numel() for p in inimodel.parameters())
))

# Performance record
record_training_loss = np.zeros((num_epochs, len(p_test)))
record_testing_success = np.zeros((num_epochs, len(p_test)))

# Train parameters over num_epochs runs
for epoch in tqdm.trange(num_epochs):

    # Compute training loss
    inimodel.requires_grad_(False)
    param_grads = inimodel.augmented_grads(p_train, xtrain, ytrain)[1]
    param_grads = param_grads.permute(1, 0, 2, 3)[0].sum(1)  # No pturb at [0]
    inimodel.requires_grad_(True)

    # Update parameters for each p in p_train[1:]
    for p_loss in param_grads[1:]:
        pp = 0  # Pointer for slicing the vector for each parameter
        for param in inimodel.parameters():
            param_length = param.numel()  # Length of parameter
            # Slice, reshape and insert grads from param_grads
            param.grad = param_grads[pp:pp + param_length].view_as(param).data
            pp += param_length  # Increment pointer
        optimiser.step()

        # Manual method
        #param_vec = parameters_to_vector(inimodel.parameters())
        #param_vec -= leaning_rate * p_loss
        # Replace parameters in inimodel
        #vector_to_parameters(param_vec, inimodel.parameters())

    # Recorded training loss
    ztrain = inimodel(p_test, xtrain)
    for ii, zi in enumerate(ztrain):
        record_training_loss[epoch, ii] = inimodel.cost_fn(
            inimodel.output(zi[0]), ytrain
        )

    # Recorded testing success
    if epoch % test_freq == 0:
        ztest = inimodel(p_test, xtest)
        for ii, zi in enumerate(ztest):
            ypred = inimodel.output(zi[0])
            ypred_argmax = torch.argmax(ypred, 1)
            record_testing_success[epoch:(epoch + test_freq), ii] = (
                    ypred_argmax == ytest_argmax
            ).float().mean()


# Remove NaN losses from the plots
record_testing_success[np.isnan(record_testing_success)] = float('NaN')

# Saving plots
if save_plots:
    if not os.path.exists('iris_plots'):
        os.makedirs('iris_plots')

# Plot loss and accuracy graphs on one plot
if plot_loss_and_success_on:
    testing_success = np.array(record_testing_success)
    training_loss = np.array(record_training_loss)
    cmap = plt.cm.coolwarm
    rcParams['axes.prop_cycle'] = cycler(
        color=cmap(np.linspace(0, 1, len(p_test)))
    )

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex='all')
    plt.title(
        'q = {q}; p_train = {p_train}; p_test = {p_test}; '
        'lr = {lr:.1e}; num_epochs = {ep}; method = {im}'
        .format(q=p_train[-1],
                p_train=p_train,
                p_test=p_test,
                lr=leaning_rate,
                ep=num_epochs,
                im=integration_method,
                )
    )

    accu_lines = ax1.plot(testing_success)
    loss_lines = ax2.plot(training_loss)

    ax1.set_ylabel('Testing success')
    ax2.set_ylabel('Training loss')
    ax2.set_xlabel('Number of epochs')

    labels = list()
    for p in reversed(p_test):
        labels.append('p = {}'.format(p))
    ax1.legend(labels)
    ax2.legend(labels)

    if view_plots:
        plt.show()

    if save_plots:
        plt.savefig(
            'iris_plots/num_grads_loss_sucess_lr_exp{lr}_ep{ep}'
            .format(lr=int(np.log10(leaning_rate)), ep=num_epochs)
        )
