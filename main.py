import tensorflow as tf

from tensorflow import keras

import gym

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation

from collections import deque

from gym import wrappers

from tensorflow.keras import backend as K

def record_epochs(index):
    return index%100==0

class obs_wrapper(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        que_length = 200
        self.outs = list(np.zeros((que_length,2)))
        self.outs = deque(self.outs,maxlen=que_length)
        
    def state_preprocess(self,state):
        state[0]=state[0]/1.2
        state[1]=state[1]/0.07
        state = np.array(state)
        self.outs.append(state)
        return np.array(list(self.outs))

    def observation(self,obs):
        obs = self.state_preprocess(obs)
        return obs

    def reset(self):
        que_length = 200
        main = super().reset()
        self.outs = list(np.zeros((que_length,2)))
        self.outs = deque(self.outs,maxlen=que_length)
        self.outs.append(main[-1])

        return np.array(list(self.outs))

class reward_wrapper(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.env = env
    
    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        reward += np.absolute(next_state[1])
        return next_state, reward, done, info

env = wrappers.Monitor(obs_wrapper(reward_wrapper(gym.make("MountainCar-v0"))),"video",record_epochs,mode="training")
state = env.reset()

def func_model():
    inp = keras.layers.Input((state.shape[0],state.shape[1]))
    gru = keras.layers.GRU(30,unroll=True)(inp)
    first = keras.layers.Conv1D(32,2,1)(inp)
    second = keras.layers.Conv1D(64,4,2)(first)
    third = keras.layers.Conv1D(128,8,4)(second)
    gru_second = keras.layers.GRU(30,unroll=True)(third)
    together = keras.layers.concatenate([gru,gru_second])
    first_layer = keras.layers.Dense(30,activation="elu")(together)
    out_layer = keras.layers.Dense(3)(first_layer)
    return keras.Model(inp,out_layer)

model = func_model()
print(model.summary())

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    else:
        #print(state.shape)
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

"""We will also need a replay memory. It will contain the agent's experiences, in the form of tuples: `(obs, action, reward, next_obs, done)`. We can use the `deque` class for that:"""

replay_memory = deque(maxlen=10000)

"""And let's create a function to sample experiences from the replay memory. It will return 5 NumPy arrays: `[obs, actions, rewards, next_obs, dones]`."""

def add_experience(experiences,experience,p,m):
    if p == False:
        if m == True:
            experiences=experiences.append(experience)
        else:
            for e in range(len(experiences)):
                if experiences[e][2] < experience[2]:
                    break
            experiences = experiences[:e]+[experience]+experiences[e:]
    else:
        experiences.pop()
        if True:
            for e in range(len(experiences)):
                if experiences[e][2] < experience[2]:
                    break
            experiences = experiences[:e]+[experience]+experiences[e:]
    return experiences

def sample_experiences(batch_size):
    #replay_memory = list(replay_memory).sort(key=lambda x:x[2])
    indices = list(range(batch_size))
    #print(len(replay_memory))
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

"""Now we can create a function that will use the DQN to play one step, and record its experience in the replay memory:"""

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

"""Lastly, let's create a function that will sample some experiences from the replay memory and perform a training step:

**Note**: the first 3 releases of the 2nd edition were missing the `reshape()` operation which converts `target_Q_values` to a column vector (this is required by the `loss_fn()`).
"""

batch_size = 2**10
discount_rate = 0.99
lr=1e-5
optimizer = keras.optimizers.Nadam(lr=lr)
loss_fn = keras.losses.logcosh

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, 3)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    print(loss)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

"""And now, let's train the model!"""

env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

rewards = [] 
best_score = np.inf
best = False
for episode in range(1000):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 200, 0.001)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    rewards.append(step) # Not shown in the book
    if step<best_score:
        best = True
    if step <= best_score: # Not shown
        best_weights = model.get_weights() # Not shown
        best_score = step # Not shown
    
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}, Best Score: {}".format(episode, step + 1, epsilon,best_score), end="") # Not shown
    if episode %50 ==0 and episode>0:
        replay_memory = list(replay_memory)
        replay_memory.sort(key=lambda x:np.absolute(x[2]))
        #print(len(replay_memory))
        #print(replay_memory)
        replay_memory = deque(replay_memory[::-1],maxlen=15000)
        training_step(batch_size)
        if best ==True:
            lr=lr/10
            optimizer=keras.optimizers.Nadam(lr=lr)

model.set_weights(best_weights)

plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()

env.seed(42)
state = env.reset()

frames = []

while True:
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim
   
plot_animation(frames)