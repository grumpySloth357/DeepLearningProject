# Built on The Pacman AI projects: http://ai.berkeley.edu/project_overview.html

import numpy as np
import random
import util
import time
import sys

from pacman import Directions
from game import Agent
import game
from collections import deque
import tensorflow as tf
from SimpleQ import *

params = {
    # Model backups
    'load_file': None,#'./models/PacmanDQN_capsuleClassic_h8_ep100000',
    'save_file': None,
    'save_interval' : 5000, 
    
    'history':1,

    'train_start': 5000,    # Episodes before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate
    'rms_decay': 0.99,      # RMS Prop decay
    'rms_eps': 1e-6,        # RMS Prop epsilon

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}                     



class SimpleQman(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        self.params['history'] = args['numHistory']
        self.params['save_file'] =sys.argv[2]+'_'+ sys.argv[8]+'_h'+str(self.params['history'])+'_x2'
        print ("HistoryLen: ",self.params['history'])
        print ("FN: ",self.params['save_file'])
        
        # Start Tensorflow session
        n = 2 #number of nodes requested
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0/n)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options, intra_op_parallelism_threads=n-1))
        self.qnet = SimpleQ(self.params)

        # Q and cost
        self.Q_global = []

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.last_reward = 0.

        self.current_state = np.zeros((self.params['height'],self.params['width'],self.params['history']))
        self.replay_mem = deque() #replay memmory
        self.last_scores = deque()


    def getMove(self, state):
        # Take epsilon greedy action
        if np.random.rand() > self.params['eps']:
            # Take action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['height'], self.params['width'], self.params['history'])), 
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred)) #GetQval
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))
            #Break tie..
            if len(a_winner) > 1:
                move = self.get_direction(a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    #mapping dxn -->action index    
    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    #mapping action index -> dxn.        
    def get_direction(self, value): 
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST
            
    def update_experience(self, state):
        if self.last_action is not None:
            # Update each step
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getState(state)
            #Update rewards
            self.last_reward = state.data.score - self.last_score
            self.last_score = state.data.score
            #Make experience
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            #Add experience
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()
            #Start training...
            self.train()
        # Next
        self.local_cnt += 1
        self.frame += 1
        #Linear epsilon decay
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))
        return state    


    

    #Pacman comes here during its final round..    
    def final(self, state):
    
        self.terminal = True
        self.update_experience(state)
        self.won = state.isWin()
        #Stats out
        sys.stdout.write("# %4d,  l_r: %12f " %
                         (self.numeps, self.last_score))
        sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

        # Save model
        if(params['save_file']):
            if self.local_cnt > self.params['train_start'] and self.numeps % self.params['save_interval'] == 0:
                self.qnet.save('./models/'+params['save_file'] + '_ep' + str(self.numeps))
                print('Model saved')

    def train(self):
        # Train
        if (self.local_cnt > self.params['train_start']):
            #Get random batch of experiences from replay memory
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            states = [] # States (s)
            rewards = [] # Rewards (r)
            actions = [] # Actions (a)
            nstates = [] # Next states (s')
            terminals = [] # Terminal state (t)

            for i in batch:
                states.append(i[0])
                rewards.append(i[1])
                actions.append(i[2])
                nstates.append(i[3])
                terminals.append(i[4])
            states = np.array(states)
            rewards = np.array(rewards)
            #actions = self.get_onehot(np.array(actions))
            nstates = np.array(nstates)
            terminals = np.array(terminals)
            #Pass onto the learner...
            self.cnt, self.cost_disp = self.qnet.train(states, rewards, actions, nstates, terminals)


    def get_onehot(self, actions):
        # make 1 hot action vector
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   


    #Get grayscaled agents in 2D matrix form    
    def getState(self, state):
        # Create observation matrix of all the agents
        width, height = self.params['width'], self.params['height']
        matrix = np.zeros((height, width))
        matrix.dtype = int

        #Grayscaled value of each agent..
        wall = 0.125
        ghost = 0.250
        food = 0.375
        capsule = 0.500
        scaredg = 0.625
        pac = 0.998
    
        #Add walls...
        walls_matrix = state.data.layout.walls
        for i in range(walls_matrix.height):
            for j in range(walls_matrix.width):
                cell = 1 if walls_matrix[j][i] else 0
                matrix[-1-i][j] = cell*wall

        #Add agents.. ghost/scare_ghost/pacman...
        for agentState in state.data.agentStates:
            pos = agentState.configuration.getPosition()
            if not agentState.isPacman: #check for ghost
                if agentState.scaredTimer > 0: #scared ghost..
                    matrix[-1-int(pos[1])][int(pos[0])] = scaredg
                else: #regular ghost
                    matrix[-1-int(pos[1])][int(pos[0])] =  ghost
            else: #pacman..
                matrix[-1-int(pos[1])][int(pos[0])] = pac                   
                
        #Add food 
        food_matrix = state.data.food
        for i in range(food_matrix.height):
            for j in range(food_matrix.width):
                cell = 1 if food_matrix[j][i] else 0
                matrix[-1-i][j] = cell*food  

        #add capsule           
        capsule_matrix = state.data.food
        for i in range(capsule_matrix.height):
            for j in range(capsule_matrix.width):
                cell = 1 if capsule_matrix[j][i] else 0
                matrix[-1-i][j] = cell*capsule 

        n = self.params['history']
        
        #Stack histories together..
        observation = np.dstack([matrix]*n)
        if (self.last_state!=None and n>1):
            observation[:,:,0:n-2] = self.current_state[:,:,1:n-1]
            observation[:,:,n-1] = obs
        #print ("OBS:")
        #print (observation.shape)
        return observation
        #return observation
    
    #Start of each episode
    def registerInitialState(self, state):

        # Reset rewards
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.

        # Reset states
        self.last_state = None
        self.current_state = self.getState(state)

        # Reset actions
        self.last_action = None

        # Reset values
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    #Perfom legal actions else STOP    
    def getAction(self, state):
        move = self.getMove(state)
        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move
