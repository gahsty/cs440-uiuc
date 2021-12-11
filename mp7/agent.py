import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)

        state: (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom,
        adjoining_body_left, adjoining_body_right)
        '''

        s_prime = self.generate_state(environment)
        #print(environment,self.a,self.s)
        if self.train:
            if not dead:

                if self.s is None:
                        self.s = s_prime
                        self.a = self.chose(self.Q,self.Ne,self.N,s_prime)
                        self.N[self.s][self.a] += 1
                        return self.a
                else:
                    if points - self.points ==1:
                        reward = 1
                        self.points = points

                    else:
                        reward = -0.1


                    best_a = self.best(self.Q, s_prime)

                    self.Q[self.s][self.a] += self.C / (self.C + self.N[self.s][self.a]) * (
                                reward + self.gamma * self.Q[s_prime][best_a] - self.Q[self.s][self.a])
                    chosen_a = self.chose(self.Q, self.Ne, self.N, s_prime)
                    self.N[s_prime][chosen_a] += 1

                    self.a = chosen_a
                    self.s = s_prime
                    return self.a
            else:
                #print(environment,s_prime,"++++")
                self.Q[self.s][self.a] = self.Q[self.s][self.a] + self.C / (self.C + self.N[self.s][self.a]) * (
                        - 1 + self.gamma * self.max_q(self.Q,s_prime) - self.Q[self.s][self.a])
                self.reset()

                return self.actions[3]


        else:
            Q_value = -999999
            for i in [3,2,1,0]:
                temp_Q = self.Q[s_prime][self.actions[i]]
                if temp_Q > Q_value:
                    Q_value = temp_Q
                    self.a = self.actions[i]
            return self.a



    def best(self, Q,state):
        temp_q = -99999
        ans = 3
        for i in [3, 2, 1, 0]:
            f = Q[state][i]
            if temp_q < f:
                temp_q = f
                ans = i
        return ans
    def max_q(self,Q,state):
        return max(self.Q[state][0],self.Q[state][1],self.Q[state][2],self.Q[state][3])

    def chose(self, Q, ne, N, state):
        temp_q = -100
        ans = 3
        for i in [3, 2, 1, 0]:
            if N[state][i] < ne:
                f = 1
            else:
                f = Q[state][i]
            if temp_q < f:
                temp_q = f
                ans = i
        return ans

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment 
        diplay_size = utils.DISPLAY_SIZE
        wall_size = utils.WALL_SIZE
        grid_size = utils.GRID_SIZE
        small_limit = wall_size / grid_size
        big_limit = (diplay_size - wall_size) / grid_size
        state = np.zeros(8, dtype=int)

        headcoordi = (environment[0] / grid_size, environment[1] / grid_size)
        foodcoordi = (environment[3] / grid_size, environment[4] / grid_size)
        if headcoordi[0] > foodcoordi[0]:
            state[0] = 1
        elif headcoordi[0] < foodcoordi[0]:
            state[0] = 2
        if headcoordi[1] > foodcoordi[1]:
            state[1] = 1
        elif headcoordi[1] < foodcoordi[1]:
            state[1] = 2

        if headcoordi[0] == small_limit:
            state[2] = 1
        elif headcoordi[0] == big_limit-1:
            state[2] = 2

        if headcoordi[1] == small_limit:
            state[3] = 1
        elif headcoordi[1] == big_limit-1:
            state[3] = 2

        if len(environment[2]) > 0:

            for i in range(len(environment[2])):

                temp_coordi = (environment[2][i][0] / grid_size, environment[2][i][1] / grid_size)
                if temp_coordi[0] == headcoordi[0] - 1 and temp_coordi[1] == headcoordi[1]:
                    state[6] = 1
                if temp_coordi[0] == headcoordi[0] + 1 and temp_coordi[1] == headcoordi[1]:
                    state[7] = 1

                if temp_coordi[0] == headcoordi[0] and temp_coordi[1] == headcoordi[1] - 1:
                    state[4] = 1

                if temp_coordi[0] == headcoordi[0] and temp_coordi[1] == headcoordi[1] + 1:
                    state[5] = 1


        return tuple(state)
