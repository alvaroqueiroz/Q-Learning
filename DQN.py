# Author: Álvaro Queiroz
#
#
# To train agent use
# $ python DQN.py train
#
# To play a round of five games use
# $ python DQN.py play
#
# To play a round of five games using weights in model_Agent_DQN.h5 file
# $ python DQN.py loadandplay
#
# If you get crashes or the code is using too much RAM, try rducing NumGamesEp


import keras
from keras import layers
from keras import models
from keras.optimizers import adam
import tensorflow as tf
import random
import numpy as np
from keras.models import load_model
import cv2
import time
import sys
import gym
from keras.callbacks import Callback

class ComputeMetrics(Callback):
    """
    Class that inherits from keras Callback to add log fields mean_pointsep, mean_qvaluesep and epsilon to log
    used in Tensorboard
    """

    def on_epoch_end(self, epoch, logs):

        # I'm using global variables to acess the values calculated in LearnToPlay method
        global mean_pointsep, mean_qvaluesep

        # Those fieds can be acessed in Tensorboard during and after training
        # They are very important to see if the training is allright and corverging
        logs['mean_pointsep'] = mean_pointsep
        logs['mean_qvaluesep'] = mean_qvaluesep
        logs['epsilon'] = getattr(Agent, 'epsilon')


class Agent:
    """
    Class Agent includes Agent Neural Network, enviroment and functions to train the agent
    Aditional Functions:
    setModel, getEpsilon, setEpsilon, load and save.
    Model wights are saved in model_Agent_DEQN.h5 file
    """

    def __init__(self, envName):
        """
        param envName: name of gym enviroment
        """

        # Compile enviroment
        self.env = self.MakeEnvironment(envName)
        # This will get the number of possible actions for the current enviroment
        self.action_size = self.env.action_space.n
        # Each image is 84x84, each state is composed by 3 images so the NN can recognize patters in movement
        self.state_size = [84, 84, 3]
        # Rate of consideration for future rewards for the agent
        self.gamma = 0.9
        # Initial rate at which the Agent Takes random action
        self.epsilon = 1 
        # Final rate at which the Agent Takes random action
        self.epsilon_min = 0.1 
        # Rate of decay of epsilon for each game episode
        self.epsilon_decay = 0.995
        # Learning rate for the optimizer ( Normally Adam)
        self.learning_rate_optimizer = 0.0001
        # Learning rate for the Agent see Q-Learning equation https://en.wikipedia.org/wiki/Q-learning
        self.learning_rate = 0.3
        # Agent NN is set by defoult, but can also be set using setModel method
        self.model = self._build_model() 
        # Number of games for each episode before training
        self.NumGamesEp = 15
        # number of epochs in training phase
        self.NumEpochs = 3
        # Save weight during training
        self.SaveWeights_inTraining = True
        # Record matrics in logs folder for use in Tensorboard
        self.RecordMetrics = True

    def setModel(self, model):
        """
        param: model
        Set compiled model as the new neural networf of the Agent
        """
        self.model = model
        # I've had some problem in recording metrics with custom models, to avoid that...
        self.RecordMetrics = False

    def getEpsilon(self):
        """
        Returns Epsilon
        """
        return self.epsilon

    def setEpsilon(self, epsilon):
        """
        param: epsilon
        Sets Epsilon for Agent
        """
        self.epsilon = epsilon

    def MakeEnvironment(self, envName):
        """
        param envName: name of gym enviroment
        returns enviroment 'compiled'
        Also sets seeds for enviroment and numpy for consistency
        """
        env = gym.make(envName)
        np.random.seed(123)
        env.seed(123)

        return env

    def load(self):
        """
        Loads saved wights for the Agent in model_Agent_DQN.h5
        file should be in the same folder as DQN.py
        """
        self.model = load_model('model_Agent_DQN.h5')

    def save(self):
        """
        Save wights for the Agent in model_Agent_DQN.h5 file
        file will be in the same folder as DQN.py
        """
        self.model.save('model_Agent_DQN.h5')
    
    def _build_model(self):

        def create_network(x):

            # Firt convlution layer
            x = layers.Conv2D(32, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
            x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)


            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024)(x)
            # Dropout can help with over-fitting
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(1024)(x)
            x = layers.Dropout(0.5)(x)
            # Output layer has the size of the action space possible in the game
            x = layers.Dense(self.action_size)(x)

            return x

        # the input of the NN has the sice of the state 84,84,3
        image_tensor = layers.Input(shape=(self.state_size))
        network_output = create_network(image_tensor)
        
        model = models.Model(inputs=[image_tensor], outputs=[network_output])

        # clipvalue will not let the gradient get biger/lower than -1,1
        optimizer = adam(lr = self.learning_rate_optimizer, clipvalue = 1)
        
        model.compile(optimizer=optimizer, 
                    loss='mean_squared_error',
                    metrics=['mae'])
        return model
    

    def NormalizeImage(self, observation):
        """
        Takes image as numpy array and put it in gray scale and recises it and normalize it /255
        param observation: image numpy array
        returns normalized image with size 84x84x1 as numpy array
        """
        # The image is first transformed from RGB to grey scale
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        # Some parts of the image can be put away (No information is lost)
        observation = observation[26:110,:]
        _, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        # Each pixel ranges from 0-255, NN can converge easily if all input range from 0-1
        # So we divede the entire numpy array by 255
        return np.reshape(observation,(84, 84, 1))/255

    def updateActionQValues(self, PlayMemory):
        """
        Takes  list Playmemory and returns data for training
        param List PlayMemory: list with state_frame, N_action, Qvalues, reward, state
        Returns 2 numpy arrays states and qvalues
        """
        # New list to append states
        StatesList = []
        # List to append qvalues as numpy arrays
        # This way we add one dimension to the training data  (batch dim)
        QValuesList = []
    
        # Loop over Playmemory to calculate new Q-values
        # Will not loop over the two first playmemory rows as they have dummy data
        # The last state of the game is discarded, it cannot be updated using the Q-learning equation
        for i in range(len(PlayMemory)-1):
            
            # [state_frame, N_action, Qvalues, reward, state]
            Qvalues = PlayMemory[i][2]
            ActionTaken = PlayMemory[i][1]
            Reward = PlayMemory[i][3]
            # Max Qvalue for the next frame
            MaxNextState = np.max(PlayMemory[i+1][2])
            
            # Q-Learning iterative equation https://en.wikipedia.org/wiki/Q-learning
            Qvalues[ActionTaken] =  Qvalues[ActionTaken] + self.learning_rate*(Reward + self.gamma*MaxNextState - Qvalues[ActionTaken])

            StatesList.append(PlayMemory[i][4])
            QValuesList.append(np.array(Qvalues))
        
        # For training, data must be in numpy arrays
        return np.array(StatesList), np.array(QValuesList)

    def PlayGame(self, renderEnv):
        """
        Funtion to play gym game and fill Playmemory list
        param bool renderEnv: render enviroment or not
        returns List Playmemory with state_frame, N_action, Qvalues, reward, state
        """
        # Create list to append game data [state_frame, N_action, Qvalues, reward, state]
        PlayMemory = []

        # Reset enviroment to start new game
        self.env.reset()

        # Inicial Q values to start Playmemory and make first state (Dummy values)
        initQvalues = [0, 0, 0, 0]

        # First dummy value with random action
        # choose actions randomly
        # We need to dummy frames to get the first state in the loop
        N_action = np.random.randint(low=0, high=self.action_size)
        # Get image frame, reward
        img, reward, end_episode, _ = self.env.step(action=N_action)
        # preprocess and normalize image
        state_frame = np.array(self.NormalizeImage(img))
        dummystate = np.dstack([state_frame, state_frame, state_frame])

        # Append the dummy row
        PlayMemory.append([state_frame, N_action, initQvalues, reward, dummystate])

        # Another one
        N_action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, _ = self.env.step(action=N_action)
        state_frame = np.array(self.NormalizeImage(img))
        dummystate = np.dstack([state_frame, state_frame, state_frame])

        PlayMemory.append([state_frame, N_action, initQvalues, reward, dummystate])

        # Sum reward to log in callback
        reward_total = reward

        # Loop over until game ends
        while not (end_episode):

            # Render enviroment to screen if renderEnv is True
            if renderEnv:
                self.env.render()
                # sleep so the rendered image does not disappear too fast
                time.sleep(0.01)

            # Get the lenght of the list to get the last state
            num_states = len(PlayMemory)-1

            # the state will be the stack of three frames, the three last ones are the current state
            state = np.dstack([PlayMemory[num_states-2][0],
            PlayMemory[num_states-1][0], 
            PlayMemory[num_states][0]])

            # Expand dimension so we can use the model to predict the Qvalues for the current state
            state_pred = np.expand_dims(state, axis=0)

            # Predict Qvalues
            Qvalues = self.model.predict(state_pred)[0]

            # Epsilon-Greedy strategy
            # We start by exploring and gradually exploit more the good actions we find while exploring
            # At the end the value of epsilon decreases, so after some time the agent
            # start to use more and more the actions it have found to be good
            if np.random.rand() >= self.epsilon:

                # Chooses the action with the biggest Qvalue
                N_action = np.argmax(Qvalues)
            else:

                # Chooses random action
                N_action = np.random.randint(low=0, high=self.action_size)

            # Step eviroment
            # Takes the action and observe reward
            img, reward, end_episode, _ = self.env.step(action=N_action)
            state_frame = np.array(self.NormalizeImage(img)) 

            reward_total += reward

            PlayMemory.append([state_frame, N_action, Qvalues, reward, state])

        # Return PlayMemory and some data to be displayed while training
        # First two rows are discarded as they are dummy data
        return PlayMemory[2:len(PlayMemory)], reward_total, Qvalues

    def LearnToPlay(self, renderEnv):
        """
        Learn to play, it uses an infinity loop
        param bool renderEnv: render enviroment or not

        """

        # Set Tensorboard callback to see the logs while training model
        TensorboadCallback = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

        # Global variables to be acessed inside ComputeMetrics class
        global mean_pointsep, mean_qvaluesep

        print()
        print("Starting training, weights are saved in model_Agent_DQN.h5")
        print()

        while(True):

            # Each episode reward and Qvalues total are set to zero to calculate the mean
            reward_ep_total = 0
            Qvalues_ep_total = np.array([0, 0, 0, 0])

            # List to append Q matrix of each game
            TotalPlayMemory = []

            # Each game episode generates the training data for one training section
            for _ in range(self.NumGamesEp):

                # Play game and get data
                PlayMemory, reward_ep, Qvalues_ep = self.PlayGame(renderEnv)

                # sum to get the mean after
                reward_ep_total += reward_ep
                Qvalues_ep_total = np.add(Qvalues_ep_total, Qvalues_ep)

                # We will append the play memory to the total
                # The TotalPlayMemory will be used to train the NN after we update its Qvalues
                TotalPlayMemory += PlayMemory

            # Calculate de the mean points and qvalues for the episode
            mean_pointsep = reward_ep_total/self.NumGamesEp
            mean_qvaluesep = np.mean(Qvalues_ep_total)/self.NumGamesEp

            # Here we update the Qvalues using the Q-Learning iterative equation
            #  and prepare data for training
            States, ActionsValues = self.updateActionQValues(TotalPlayMemory)


            print()
            print("Epsilon : {} Median Points : {} Median QValues : {}".format(round(self.epsilon,2),
            round(mean_pointsep,2),
            round(mean_qvaluesep,2)))
            print()
            print("Optimizing Model")
            print()
            if self.RecordMetrics:
                self.model.fit(x = States, y = ActionsValues, epochs = self.NumEpochs,
                callbacks=[ComputeMetrics(), TensorboadCallback])
            else:
                self.model.fit(x = States, y = ActionsValues, epochs = self.NumEpochs)
            print()
            print("Done playing {} game(s)! Will start another round. Press CTRL + C to stop training".format(self.NumGamesEp))
            print()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if self.SaveWeights_inTraining:
                self.model.save('model_Agent_DQN.h5')

    def JustPlay(self, renderEnv, NumGames):
        for _ in range(NumGames):

            _  = self.PlayGame(renderEnv)

def main(Agent):

    if sys.argv[1] == 'train':

        Agent.LearnToPlay(renderEnv = False)

    elif sys.argv[1] == 'play':

        Agent.JustPlay(NumGames = 5, renderEnv = True)

    elif sys.argv[1] == 'loadandplay':

        try:
            Agent.load()
        except:
            print('Could not load weights in model_Agent_DQN.h5')

        Agent.JustPlay(NumGames = 5, renderEnv = True)

    else:
        raise Exception('Invalid argument, choose train play or loadandplay (need wights in file model_Agent_DQN.h5)')

if __name__ == "__main__":

    envName = 'Breakout-v0'
    Agent = Agent(envName)
    main(Agent)