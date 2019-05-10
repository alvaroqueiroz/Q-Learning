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

    def on_epoch_end(self, epoch, logs):

        global mean_pointsep, mean_qvaluesep

        logs['mean_pointsep'] = mean_pointsep
        logs['mean_qvaluesep'] = mean_qvaluesep
        logs['epsilon'] = getattr(Agent, 'epsilon')


class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.7 
        self.epsilon = 1 
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.1 
        self.learning_rate_optimizer = 0.00001
        self.model = self._build_model() 
        self.learning_rate_QLearning = 0.3 
        self.NumGamesEp = 10

    def getEpsilon(self):
        return self.epsilon

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon
    
    def _build_model(self):
        cardinality = 1
        def residual_network(x):
            """
            ResNeXt by default. For ResNet set `cardinality` = 1 above.
            
            """
            def add_common_layers(y):
                y = layers.BatchNormalization()(y)
                y = layers.ReLU()(y)

                return y

            def grouped_convolution(y, nb_channels, _strides):
                # when `cardinality` == 1 this is just a standard convolution
                if cardinality == 1:
                    return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
                
                assert not nb_channels % cardinality
                _d = nb_channels // cardinality

                # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
                # and convolutions are separately performed within each group
                groups = []
                for j in range(cardinality):
                    group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
                    groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
                    
                # the grouped convolutional layer concatenates them as the outputs of the layer
                y = layers.concatenate(groups)

                return y

            def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
                """
                Our network consists of a stack of residual blocks. These blocks have the same topology,
                and are subject to two simple rules:
                - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
                - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
                """
                shortcut = y

                # we modify the residual building block as a bottleneck design to make the network more economical
                y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
                y = add_common_layers(y)

                # ResNeXt (identical to ResNet when `cardinality` == 1)
                y = grouped_convolution(y, nb_channels_in, _strides=_strides)
                y = add_common_layers(y)

                y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
                # batch normalization is employed after aggregating the transformations and before adding to the shortcut
                y = layers.BatchNormalization()(y)

                # identity shortcuts used directly when the input and output are of the same dimensions
                if _project_shortcut or _strides != (1, 1):
                    # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
                    # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
                    shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
                    shortcut = layers.BatchNormalization()(shortcut)

                y = layers.add([shortcut, y])

                # relu is performed right after each batch normalization,
                # expect for the output of the block where relu is performed after the adding to the shortcut
                y = layers.ReLU()(y)

                return y

            # conv1
            x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
            x = add_common_layers(x)

            # conv2
            x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
            for i in range(2):
                project_shortcut = True if i == 0 else False
                x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(512)(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(self.action_size)(x)

            return x
        image_tensor = layers.Input(shape=(self.state_size))
        network_output = residual_network(image_tensor)
        
        model = models.Model(inputs=[image_tensor], outputs=[network_output])

        optimizer = adam(lr = self.learning_rate_optimizer)
        
        model.compile(optimizer=optimizer, 
                    loss='mean_squared_error',
                    metrics=['mae'])
        return model
    

    def NormalizeImage(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110,:]
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return np.reshape(observation,(84, 84, 1))/255

    def updateActionQValues(self, PlayMemory):
    
        Q = []
    
        for i in range(2, len(PlayMemory)-1):
            
            estimative = np.add(-(PlayMemory[i][3] + self.gamma*np.max(PlayMemory[i+1][2])), PlayMemory[i][2])
            
            Qvalue = np.add(PlayMemory[i][2], -self.learning_rate_QLearning*estimative)

            Q.append([PlayMemory[i][0], Qvalue]) 
        
        return Q

    def PlayGame(self, env, renderEnv):
        PlayMemory = []

        env.reset()

        initActions = [0, 0, 0, 0]

        N_action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, _ = env.step(action=N_action)
        state_frame = np.array(self.NormalizeImage(img))

        PlayMemory.append([state_frame, N_action, initActions, reward])

        N_action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, _ = env.step(action=N_action)
        state_frame = np.array(self.NormalizeImage(img))

        PlayMemory.append([state_frame, N_action, initActions, reward])


        reward_total = reward
        while not (end_episode):

            if renderEnv:
                env.render()
                time.sleep(0.01)

            num_states = len(PlayMemory)-1
            state = np.dstack([PlayMemory[num_states-2][0],
            PlayMemory[num_states-1][0], 
            PlayMemory[num_states][0]])

            state_pred = np.expand_dims(state, axis=0)

            Qvalues = self.model.predict(state_pred)[0]

            if np.random.rand() <= self.epsilon:
                N_action = np.argmax(Qvalues)
            else:
                N_action = np.random.randint(low=0, high=self.action_size)

            img, reward, end_episode, _ = env.step(action=N_action)
            state_frame = np.array(self.NormalizeImage(img)) 

            reward_total += reward

            PlayMemory.append([state_frame, N_action, Qvalues, reward])

        return PlayMemory, reward_total, Qvalues

    def LearnToPlay(self, env, renderEnv):

        TensorboadCallback = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)

        global mean_pointsep, mean_qvaluesep

        print()
        print("Starting training, weights are saved in model_Agent_DQN.h5")
        print()

        while(True):

            reward_ep_total = 0
            Qvalues_ep_total = np.array([0, 0, 0, 0])

            Q = []

            for _ in range(self.NumGamesEp):

                PlayMemory, reward_ep, Qvalues_ep = self.PlayGame(env, renderEnv)

                reward_ep_total += reward_ep
                Qvalues_ep_total = np.add(Qvalues_ep_total, Qvalues_ep)

                Q1 = self.updateActionQValues(PlayMemory)

                Q+=Q1

            mean_pointsep = reward_ep_total/self.NumGamesEp
            mean_qvaluesep = np.mean(Qvalues_ep_total)/self.NumGamesEp

            x, y = self.PrepareTrainData(Q)


            print()
            print("Epsilon : {} Median Points : {} Median QValues : {}".format(round(self.epsilon,2),
            round(mean_pointsep,2),
            round(mean_qvaluesep,2)))
            print()
            print("Optimizing Model")
            print()
            self.model.fit(x=x, y=y, epochs = 4, callbacks=[ComputeMetrics(), TensorboadCallback])
            print()
            print("Done playing {} games! weights saved. Will start another round. Press CTRL + C to stop training".format(self.NumGamesEp))
            print()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.model.save('model_Agent_DQN.h5')

            del Q

    def JustPlay(self, env, renderEnv, NumGames):
        for _ in range(NumGames):

            _  = self.PlayGame(env, renderEnv)

    def PrepareTrainData(self, Q):
        statelist = []
        actionslist = []

        state0 = Q[0][0]
        state = np.dstack([state0, state0, state0])
        statelist.append(state)

        state0 = Q[1][0]
        state = np.dstack([state0, state0, state0])
        statelist.append(state)

        action0 = Q[0][1]
        actionslist.append(np.array(action0))
        action0 = Q[1][1]
        actionslist.append(np.array(action0))

        for i in range (2,len(Q)):
            state = np.dstack([Q[i-2][0], Q[i-1][0], Q[i][0]])
            statelist.append(state)

            action0 = Q[i][1]
            actionslist.append(np.array(action0))
            
        x = np.array(statelist[3:len(Q)])
        y = np.array(actionslist[3:len(Q)])

        return x,y

    def load(self):
        self.model = load_model('model_Agent_DQN.h5')

    def save(self):
        self.model.save('model_Agent_DQN.h5')

def main(Agent, env):

    if sys.argv[1] == 'train':

        Agent.LearnToPlay(env, False)

    elif sys.argv[1] == 'play':

        Agent.JustPlay(env, 5, True)

    elif sys.argv[1] == 'loadandplay':

        try:
            Agent.load()
        except:
            print('Could not load weights in model_Agent_DQN.h5')

        Agent.JustPlay(env, 5, True)

    else:
        raise Exception('Invalid argument, choose train play or loadandplay (need wights in file model_Agent_DQN.h5)')

if __name__ == "__main__":

    env_name = 'Breakout-v0'
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    img_height = 84
    img_width = 84
    img_channels = 3

    state_dim = [img_height,img_width,img_channels]
    Agent = Agent(state_dim, nb_actions)
    main(Agent, env)