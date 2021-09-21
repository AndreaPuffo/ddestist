import pymc3 as pm

import theano.tensor as T


def construct_nn(net, ann_in, ann_out, n_input, n_hidden, n_output):

    # Initialize random weights between each layer
    init_1 = net.fc1.weight.data.numpy().T
    init_2 = net.fc2.weight.data.numpy().T
    init_out = net.fc3.weight.data.numpy().T
    b_1 = net.fc1.bias.data.numpy()
    b_2 = net.fc2.bias.data.numpy()
    b_3 = net.fc3.bias.data.numpy()

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        ann_input = pm.Data("ann_input", ann_in)
        ann_output = pm.Data("ann_output", ann_out)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", init_1, sigma=0.001, shape=(n_input, n_hidden), testval=init_1)
        bias_1 = pm.Normal('b1', b_1, sigma=0.001, shape=(n_hidden,), testval=b_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", init_2, sigma=0.001, shape=(n_hidden, n_hidden), testval=init_2)
        bias_2 = pm.Normal('b2', b_2, sigma=0.001, shape=(n_hidden,), testval=b_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", init_out, sigma=0.001, shape=(n_hidden, n_output), testval=init_out)
        bias_3 = pm.Normal('b3', b_3, sigma=0.001, shape=(n_output,), testval=b_3)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1) + bias_1)
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + bias_2)
        act_out = T.nnet.softmax(pm.math.dot(act_2, weights_2_out) + bias_3)

        # Binary classification -> Bernoulli likelihood
        # General classification -> Categorical likelihood
        out = pm.Categorical(
            "out",
            p=act_out,
            observed=ann_output,
            # total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network
