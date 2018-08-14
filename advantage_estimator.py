import numpy as np

class adv_nn:

    def __init__(self, learning_rate, weight_scale, epochs):
        self.learning_rate = learning_rate
        self.weight_scale = weight_scale
        self.epochs = epochs


    def train(self, dataset, targets):
        nentries = dataset.shape[0]
        ndims = dataset.shape[1]
        n_batches = int(nentries/16)
        data_left = nentries%16
        for epoch in np.arange(self.epochs):

            # Get a randomized batch of data
            dandt = np.concatenate(dataset, targets)
            np.random.shuffle(dandt)
            batch_no = np.random.randint(0, n_batches-2)
            batch = dataset[batch_no*16:(batch_no+1)*batch_no]
            batch_data = batch[:,:5]
            batch_target = batch[:,5:]
            # Get forward output
            score = self.forward_pass(batch)

            # Calculate the loss
            loss = self.loss(score, batch_target)

            # Do backward pass and then update the weights



    def forward_pass(self, inputs):
        """
        :param inputs: (batch_size x dims)
        :return: Y: (scalar)
        """

        units = 256
        batch_size = inputs.shape[0]
        dims = inputs.shape[1]

        w_1 = np.random.rand(dims, units)
        b1 = np.random.rand(units)
        acache1 = (np.matmul(w_1, inputs.T) + b1).T
        rcache1 = acache1[acache1 > 0]

        w_2 = np.random.rand(units, units)
        b2 = np.random.rand(units)
        acache2 = 


    def loss(self, score, targets):
        """
        :param score: (batch_size, 3)
        :param targets: (batch_size, 3)
        :return:
        """

        loss = score - targets
        loss = np.reduce_sum(loss)
        return loss




