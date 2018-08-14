import math
import random

# from pong import state
import random

ALPHA = 4
GAMMA = 0.85
EPSILON = 0.03


class state:
    def __init__(self, ball_x, ball_y, velocity_x, velocity_y, paddle_y):
        self.b_x = ball_x
        self.b_y = ball_y
        self.v_x = velocity_x
        self.v_y = velocity_y
        self.p_y = paddle_y
        self.p_x = 1
        self.p_h = 0.2
        self.scores = []
        self.total = 0
        self.rounds = 0
        self.MER=0
        self.score = 0
        self.curr_state = None
        self.curr_action = None
        self.terminate = False
        self.running = 1
        self.end_train = 0
        self.q_table = {}
        self.N = {}
        self.actions = [-1, 0, 1]
        self.x = [0]
        self.y = [0]
        print(ALPHA, GAMMA, EPSILON)

    def hit_paddle(self):
        if (self.b_x >= 1):
            if self.p_y <= self.b_y <= self.p_y + 0.2:
                self.score += 1
                self.b_x = 2 * self.p_x - self.b_x
                U = random.uniform(-0.015, 0.015)
                V = random.uniform(-0.03, 0.03)
                self.v_x = -self.v_x + U
                self.v_y += V
                if abs(self.v_x < 0.03):
                    if self.v_x > 0:
                        self.v_x = 0.03
                    else:
                        self.v_x = -0.03
                if self.v_x > 1:
                    self.v_x = 1
                if self.v_y > 1:
                    self.v_y = 1
                return 1
            else:
                self.terminate = True
                # self.terminated()
                return -1
        return 0

    def terminated(self):
        self.running = 0
        self.scores.append(self.score)
        self.score = 0
        self.rounds += 1
        if self.rounds % 500 == 0:
            total = (sum(self.scores))/ 500
            self.x.append(self.rounds)
            self.y.append(total)
            print(total, self.rounds)
            self.scores=[]
        if self.rounds == 100000:
            self.end_train = 1

    def get_state(self):
        # DISCRETIZE
        if self.terminate:
            return (0, 0, 0, 0, 0)
        else:
            if self.v_x > 0:
                dvx = 1
            else:
                dvx = -1
            if self.v_y > 0.015:
                dvy = 1
            elif abs(self.v_y) <= 0.015:
                dvy = 0
            else:
                dvy = -1
            dbx = min(11, int(math.floor(12 * self.b_x)))
            dby = min(11, int(math.floor(12 * self.b_y)))
            dpy = int(math.floor(12 * self.p_y / (1 - self.p_h)))
            if self.p_y == 0.8:
                dpy = 11
            return (dbx, dby, dvx, dvy, dpy)

    def restart(self):
        self.b_x = 0.5
        self.b_y = 0.5
        self.v_x = 0.03
        self.v_y = 0.01
        self.p_y = 0.4
        self.terminate = False
        self.curr_state = None

    def bounce(self):
        if self.b_x < 0:
            self.b_x = -self.b_x
            self.v_x = -self.v_x
        elif self.b_y < 0:
            self.b_y = -self.b_y
            self.v_y = -self.v_y
        elif self.b_y > 1:
            self.b_y = 2 - self.b_y
            self.v_y = -self.v_y

    def move_ball(self):
        self.b_x += self.v_x
        self.b_y += self.v_y

    def move_paddle(self, action):
        self.p_y += action * 0.04
        if self.p_y < 0:
            self.p_y = 0
        if self.p_y > 0.8:
            self.p_y = 0.8

    def Q_Action(self, state):
        x=[]
        if random.random() < EPSILON:
            return random.choice(self.actions)
        else:
            for a in self.actions:
                x.append(self.q_table.get((state, a), 0.0))
            Q_max = max(x)
            if x.count(Q_max) > 1:
                best = [i for i in range(3) if x[i] == Q_max]
                action = self.actions[random.choice(best)]
                return action
            else:
                return self.actions[x.index(Q_max)]

    def learn_QTD(self, state, action, reward, new_state):
        x=[]
        for a in self.actions:
            x.append(self.q_table.get((new_state, a), 0.0))
        new_Q= max(x)
        if (state, action) not in self.N:
            self.N[(state, action)] = 0
        self.N[(state, action)] += 1
        prev_Q = self.q_table.get((state, action), None)
        if prev_Q == None:
            self.q_table[(state, action)] = reward
        else:
            self.q_table[(state, action)] = prev_Q + (
                (float(ALPHA) / float(ALPHA + self.N[(state, action)])) * (reward + (GAMMA * new_Q) - prev_Q))

    def learn_SARSA(self, state, action, reward, new_state):
        new_Q= self.q_table.get((new_state, action), 0.0)
        if (state, action) not in self.N:
            self.N[(state, action)] = 0
        self.N[(state, action)] += 1
        prev_Q = self.q_table.get((state, action), None)
        if prev_Q == None:
            self.q_table[(state, action)] = reward
        else:
            self.q_table[(state, action)] = prev_Q + (
                (float(ALPHA) / float(ALPHA + self.N[(state, action)])) * (reward + (GAMMA * new_Q) - prev_Q))

    def qtd_train(self):

        reward = self.hit_paddle()
        new_state = self.get_state()
        self.MER+=reward
        # print(state)
        # print (self.b_x, self.b_y, self.v_x, self.v_y, self.hit)
        if self.curr_state is not None:
            self.learn_QTD(self.curr_state, self.curr_action, reward, new_state)
            #self.learn_SARSA(self.curr_state, self.curr_action, reward, new_state)

        if self.terminate:
            self.terminated()
            self.restart()

        action = self.Q_Action(new_state)

        self.curr_state = new_state
        self.curr_action = action

        self.move_paddle(action)
        self.move_ball()
        self.bounce()

    def test_update(self):
        reward = self.hit_paddle()
        state = self.get_state()
        if self.terminate:
            self.terminated_test()
            self.restart()
        action = self.Q_Action(state)
        self.move_paddle(action)
        self.move_ball()
        self.bounce()

    def terminated_test(self):
        self.scores.append(self.score)
        self.total += self.score
        self.score = 0
        self.rounds += 1
        if self.rounds == 200:
            print(self.total / 200)
            self.end_train=1