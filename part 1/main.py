from pong2 import *
import json
import pickle
import matplotlib.pyplot as plt

game = state(0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)
#game = state(0.5, 0.5, -0.03, 0.01, 0.5 - 0.2 / 2)

if __name__ == '__main__':
        #TRAIN
     while True:
        game.qtd_train()
        if game.end_train:
            with open('qlearnd3.txt', 'wb') as handle:
                pickle.dump(game.q_table, handle)
            with open('x.txt', 'wb') as handle:
                pickle.dump(game.x, handle)
            with open('y.txt', 'wb') as handle:
                pickle.dump(game.x, handle)
            plt.plot(game.x, game.y, 'b', label='')
            plt.ylabel('Mean Episode Rewards')
            plt.xlabel('Episodes')
            plt.show()

           #     TEST

    # with open('qlearnd.txt', 'rb') as handle:
    #     game.q_table = pickle.loads(handle.read())
    # while True:
    #     game.test_update()
    #     if game.end_train:
    #         exit()

    # # 1.2 DOESNT WORK
    # # with open('qlearnd.txt', 'rb') as handle:
    # #     game.q_table = pickle.loads(handle.read())
    # while True:
    #     game.other_train()
    #     if game.end_train:
    #         exit()
