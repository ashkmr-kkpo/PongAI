from bhv_cloning import state, cache, affine_fwd, affine_bwd, relu_fwd, relu_bwd, test_nn
from support import read_data
import matplotlib.pyplot as plt
import numpy as np

def adv_init_params(unit_num, weight_scale):
    W1 = (np.random.rand(5, unit_num) * weight_scale)
    W2 = (np.random.rand(unit_num, unit_num) * weight_scale)
    W3 = (np.random.rand(unit_num, unit_num) * weight_scale)
    W4 = (np.random.rand(unit_num, 3) * weight_scale)

    b1 = np.zeros(unit_num)
    b2 = np.zeros(unit_num)
    b3 = np.zeros(unit_num)
    b4 = np.zeros(3)

    return W1, W2, W3, W4, b1, b2, b3, b4


def mean_sq_loss_fn(F, Y):
    """
    :param F: (num_samples, num_actions) - predicted adv values per state and per action
    :param Y: (num_samples, num_actions) - actual adv values per state and per action
    :return loss: - scalar that represents total square loss
    :return dF: - (num_samples, num_actions) - gradient of the loss
    """
    n = F.shape[0]
    loss = np.sum(np.sum((F - Y)**2, axis=1))/(2 * n)
    dF = (F - Y)/n
    return loss, dF


def adv_train_nn(X, W1, W2, W3, W4, b1, b2, b3, b4, y, l_rate):
    inst_num = X.shape[0]
    global acc_count
    global conf_mat
    global epoch_count
    Z1, acache1 = affine_fwd(X, W1, b1)
    A1, rcache1 = relu_fwd(Z1)
    # print("affine1\n",A1)
    Z2, acache2 = affine_fwd(A1, W2, b2)
    A2, rcache2 = relu_fwd(Z2)
    # print("affine2\n",A2)
    Z3, acache3 = affine_fwd(A2, W3, b3)
    A3, rcache3 = relu_fwd(Z3)
    # print("affine3\n",A3)
    F, acache4 = affine_fwd(A3, W4, b4)
    # print("affine4\n", F)

    loss, dF = mean_sq_loss_fn(F, y)

    decision = F.argmax(1)
    if epoch_count > 99:
        pass
       # print(decision)
    for i in np.arange(inst_num):
        pred = int(decision[i])
        act = int(np.argmax(y[i, :]))
        conf_mat[act][pred] += 1
        if pred == act:
            # print("correct")
            acc_count += 1

    dA3, dW4, db4 = affine_bwd(dF, acache4)
    dZ3 = relu_bwd(dA3, rcache3)

    dA2, dW3, db3 = affine_bwd(dZ3, acache3)
    dZ2 = relu_bwd(dA2, rcache2)

    dA1, dW2, db2 = affine_bwd(dZ2, acache2)
    dZ1 = relu_bwd(dA1, rcache1)

    dX, dW1, db1 = affine_bwd(dZ1, acache1)

    W1 = W1 - l_rate * dW1
    W2 = W2 - l_rate * dW2
    W3 = W3 - l_rate * dW3
    W4 = W4 - l_rate * dW4

    b1 = b1 - l_rate * db1
    b2 = b2 - l_rate * db2
    b3 = b3 - l_rate * db3
    b4 = b4 - l_rate * db4

    return loss, W1, W2, W3, W4, b1, b2, b3, b4

def adv_minibatch_gd(X, y, l_rate, batch_size, unit_num, n_epoch):
    global conf_mat
    global acc_count
    global epoch_count
    epoch_count = 0
    conf_mat = np.zeros((3, 3))
    acc_count = 0
    loss_vec = np.zeros(n_epoch)
    acc_vec = np.zeros(n_epoch)
    W1, W2, W3, W4, b1, b2, b3, b4 = adv_init_params(unit_num, 0.01)
    curr_batch = np.zeros((batch_size, 5))
    curr_targ = np.zeros(batch_size)
    for round in np.arange(n_epoch):
        epoch_count += 1
        tot_loss = 0
        ind_list = [i for i in range(10000)]
        ln_rate = l_rate
        if round > n_epoch / 2:
            ln_rate = l_rate / (round - (n_epoch / 2) + 1)
        # print(ind_list)
        np.random.shuffle(ind_list)
        # print(ind_list)
        X_shuffled = X[ind_list, :]
        y_shuffled = y[ind_list]
        for batch in np.arange(10000 / batch_size):
            batch_start = int(batch * batch_size)
            batch_end = int((batch + 1) * batch_size)
            curr_batch = X_shuffled[batch_start:batch_end, :]
            curr_targ = y_shuffled[batch_start:batch_end]
            # print(curr_batch)
            # print(curr_batch.shape, curr_targ.shape)
            loss, W1, W2, W3, W4, b1, b2, b3, b4 = adv_train_nn(curr_batch, W1, W2, W3, W4, b1, b2, b3, b4, curr_targ,
                                                            ln_rate)
            tot_loss += loss
        accuracy = acc_count / (10000 * (round + 1))
        loss_vec[round] = tot_loss
        acc_vec[round] = accuracy
        if round == n_epoch - 1:
            print(tot_loss, accuracy, "\n")

    return W1, W2, W3, W4, b1, b2, b3, b4, loss_vec, acc_vec, conf_mat


adv_exp_state, adv_exp_decision = read_data("advantages.txt")
mean = np.mean(adv_exp_state, axis=0)
sdv = np.std(adv_exp_state, axis=0)
# print(mean, sdv)

adv_exp_state = adv_exp_state - mean
adv_exp_state = adv_exp_state / sdv
# print(adv_exp_state)

w1, w2, w3, w4, b1, b2, b3, b4, loss, acc, conf = adv_minibatch_gd(adv_exp_state, adv_exp_decision, 0.5, 100, 256, 1000)
plt.plot(loss)
plt.ylabel('advantage Loss')
plt.xlabel('advantage epoch')
plt.show()
plt.plot(acc)
plt.ylabel('advantage Accuracy')
plt.xlabel('advantage epoch')
plt.show()
# conf_sum = np.sum(conf, axis=1)
# for i in np.arange(3):
#     for j in np.arange(3):
#         conf[i][j] /= conf_sum[i]
# print(conf)

scores = np.zeros(200)
for num_games in np.arange(200):
    initial_state = state(0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)
    score = 0
    while initial_state.running == 1:
        game_state = [initial_state.b_x, initial_state.b_y, initial_state.v_x, initial_state.v_y, initial_state.p_y]
        #print(game_state)
        game_state -= mean
        game_state /= sdv
        game_state = np.asarray(game_state)
        in_state = np.zeros((1,5))
        in_state[0,:] = game_state
        decision = test_nn(in_state, w1, w2, w3, w4, b1, b2, b3, b4)
        initial_state.actions(decision[0])
        score = initial_state.updatestate(score)
    scores[num_games] = score
    #print(score)

plt.plot(scores)
plt.ylabel('advantage Bounce')
plt.xlabel('advantage game')
plt.show()
print(np.mean(scores))
