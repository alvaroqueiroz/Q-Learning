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

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.5 # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        self.epsilon = 1 # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon_decay = 0.999 # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_min = 0.01 # minimum amount of random exploration permitted
        self.learning_rate = 0.05 # rate at which NN adjusts models parameters via Adam to reduce cost 
        self.model = self._build_model() # private method 
    
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

        optimizer = adam(lr = self.learning_rate, clipvalue=1)
        
        model.compile(optimizer=optimizer, 
                    loss='mean_squared_error',
                    metrics=['mae'])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        #self.memory.append((state, action, reward, next_state, done)) # list of previous experiences, enabling re-training later
        pass

    def NormalizeImage(self, observation):
        observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
        observation = observation[26:110,:]
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,(84,84,1))/255

    def updateActionQValues(self, Q, reward_total):
    
        Qret = Q
        
        Q[0][1] = [0] * (self.action_size)
        Q[1][1] = [0] * (self.action_size)
    
        for i in range(2,len(Q)-1):
            
            state = np.dstack([Q[i-2][0], Q[i-1][0], Q[i][0]])
        
            state_pred = np.expand_dims(state, axis=0)
            actions_pred = self.model.predict(state_pred)[0]
            
            action_values = [0] * (self.action_size)
            action_values[Q[i][1]] = reward_total*(0.98)**(len(Q)-i)  + self.gamma*Q[i+1][1]
            
            actions_values = []

            for num1,num2 in zip(actions_pred, action_values):
                actions_values.append(num1+num2)
            Qret[i][1] = actions_values
        
        return Qret[2:len(Q)-2]

    def PlayGame(self, env):
        Q = []

        env.reset()

        action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, info = env.step(action=action)
        state_frame = np.array(self.NormalizeImage(img))

        Q.append([state_frame, action])

        action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, info = env.step(action=action)
        state_frame = np.array(self.NormalizeImage(img))

        Q.append([state_frame, action])

        action = np.random.randint(low=0, high=self.action_size)
        img, reward, end_episode, info = env.step(action=action)
        state_frame = np.array(self.NormalizeImage(img))

        Q.append([state_frame, action])

        reward_total = reward
        while not (end_episode):

            num_states = len(Q)-1
            state = np.dstack([Q[num_states-2][0], Q[num_states-1][0], Q[num_states][0]])

            state_pred = np.expand_dims(state, axis=0)

            if np.random.rand() <= self.epsilon:
                action = np.argmax(self.model.predict(state_pred)[0])
            else:
                action = np.random.randint(low=0, high=self.action_size)

            img, reward, end_episode, info = env.step(action=action)
            state_frame = np.array(self.NormalizeImage(img)) 

            Q.append([state_frame, action])   

            reward_total += reward

        print("Points : {}, Epsilon : {}".format(reward_total, round(self.epsilon,2)) )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        Q = self.updateActionQValues(Q, reward_total)
    
        return Q

    def LearnToPlay(self, env):
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)
        while(True):
            Q = []
            for _ in range(15):
                Q1 = self.PlayGame(env)
                Q+=Q1
            x,y = self.PrepareTrainData(Q)

            print()
            print("Optimizing Model")
            print()
            self.model.fit(x=x,y=y,epochs = 4,callbacks=[tbCallBack])
            print()
            del Q

    def JustPlay(self, env, NumGames):
        for _ in range(NumGames):

            Q = []

            env.reset()

            action = np.random.randint(low=0, high=self.action_size)
            img, reward, end_episode, info = env.step(action=action)
            state_frame = np.array(self.NormalizeImage(img))

            Q.append([state_frame, action])

            action = np.random.randint(low=0, high=self.action_size)
            img, reward, end_episode, info = env.step(action=action)
            state_frame = np.array(self.NormalizeImage(img))

            Q.append([state_frame, action])

            action = np.random.randint(low=0, high=self.action_size)
            img, reward, end_episode, info = env.step(action=action)
            state_frame = np.array(self.NormalizeImage(img))

            Q.append([state_frame, action])

            reward_total = reward

            while not (end_episode):
                env.render()

                num_states = len(Q)-1
                state = np.dstack([Q[num_states-2][0], Q[num_states-1][0], Q[num_states][0]])

                state_pred = np.expand_dims(state, axis=0)
                action = np.argmax(self.model.predict(state_pred)[0])
                img, reward, end_episode, info = env.step(action=action)  

                reward_total += reward

        print("Points : {}".format(reward_total))

    def PrepareTrainData(self, Q):
        statelist = []
        actionslistaaa = []

        state0 = Q[0][0]
        state = np.dstack([state0, state0, state0])
        action0 = Q[2][1]
        actionslistaaa.append(np.array(action0))
        action0 = Q[2][1]
        actionslistaaa.append(np.array(action0))

        statelist.append(state)
        state0 = Q[1][0]
        state = np.dstack([state0, state0, state0])
        statelist.append(state)

        for i in range (2,len(Q)):
            state = np.dstack([Q[i-2][0], Q[i-1][0], Q[i][0]])
            statelist.append(state)
            action0 = Q[i][1]
            actionslistaaa.append(np.array(action0))
        x = np.array(statelist)
        y = np.array(actionslistaaa)
        return x,y


    # def act(self, state):
    #     if np.random.rand() <= self.epsilon: # if acting randomly, take random action
    #         return random.randrange(self.action_size)
    #     act_values = self.model.predict(state) # if not acting randomly, predict reward value based on current state
    #     return np.argmax(act_values[0]) # pick the action that will give the highest reward (i.e., go left or right?)

    # def replay(self, batch_size): # method that trains NN with experiences sampled from memory
    #     minibatch = random.sample(self.memory, batch_size) # sample a minibatch from memory
    #     for state, action, reward, next_state, done in minibatch: # extract data for each minibatch sample
    #         target = reward # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
    #         if not done: # if not done, then predict future discounted reward
    #             target = (reward + self.gamma * # (target) = reward + (discount rate gamma) * 
    #                       np.amax(self.model.predict(next_state)[0])) # (maximum target Q based on future action a')
    #         target_f = self.model.predict(state) # approximately map current state to future discounted reward
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0) # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

    def load(self):
        self.model = load_model('model_Agent_DQN.h5')

    def save(self):
        self.model.save('model_Agent_DQN.h5')