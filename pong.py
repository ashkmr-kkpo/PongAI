import numpy
import random


class state:
    def __init__(self, ball_x, ball_y, velocity_x, velocity_y, paddle_y):
        self.b_x = ball_x
        self.b_y = ball_y
        self.v_x = velocity_x
        self.v_y = velocity_y
        self.p_y = paddle_y
        self.p_x = 1
        self.p_h = 0.2


def actions(self, input, state):
    if input == 0:
        print("nothing")
    elif input == 1:
        print("up")
        state.p_y += 0.4
        if state.p_y <= 0:
            state.p_y = 0
    elif input == 2:
        print("down")
        state.p_y -= 0.4
        if state.p_y >= 0.8:
            state.p_y = 0.8


def terminate():
    print("end")


def updatestate(self, state, score):
    state.b_x += state.v_x
    state.b_y += state.v_y

    if state.b_x == 1:
        if state.p_y <= state.b_y <= state.p_y + 0.2:
            score += 1
            state.b_x = 2 * state.p_x - state.b_x
            U = random.uniform(-0.015, 0.015)
            V = random.uniform(-0.03, 0.03)
            state.v_x = -state.v_x + U
            state.v_y += V
            if abs(state.v_x < 0.03):
                if state.v_x > 0:
                    state.v_x = 0.03
                else:
                    state.v_x = -0.03
            if state.v_x > 1:
                state.v_x = 1
            if state.v_y > 1:
                state.v_y = 1
        else:
            score -= 1
            terminate()
    elif state.b_y < 0:
        state.b_y = -state.b_y
    elif state.b_y > 1:
        state.b_y = 2 - state.b_y
        state.v_y = -state.v_y


grid = numpy.zeros((12, 12))
initial_state = state(0.5, 0.5, 0.03, 0.01, 0.5 - 0.2 / 2)
