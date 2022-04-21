import gym
import numpy as np
import tensorflow as tf
from collections import deque
from preprocess import stack_frames
from memory import Memory
import matplotlib.pyplot as plt

def agent(state_shape,action_shape,lr):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=(4,4),activation='relu',input_shape=state_shape),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),activation='relu'),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(action_shape,activation=None)#no activation to give probabilities as output
    ])
    model.compile(loss=tf.keras.losses.Huber(),optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    return model

def alternative_agent(state_shape,action_shape,lr):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=(4,4),activation='relu',input_shape=state_shape),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),activation='relu'),
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(256,activation='relu'),
        tf.keras.layers.Dense(action_shape,activation=None)#no activation to give probabilities as output
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.Adam(learning_rate=lr),metrics=[tf.keras.metrics.Accuracy()])
    return model

def policy(epsilon_start,epsilon_stop,decay_rate,decay_step,state,actions,model):
    """epsilon greedy policy this particular policy uses a decaying epsilon
    value over time steps. This allows for grater exploration at the start 
    and greater exploitation once the network has learnt the best actions"""
    threshold=np.random.rand()
    epsilon=epsilon_stop + (epsilon_start - epsilon_stop) * np.exp(-decay_rate * decay_step)
    
    if threshold<epsilon:
        action=np.random.choice(actions)
        explore=True
    else:
        Qs=model.predict(state)
        action=np.argmax(Qs)
        explore=False
        
        
    return action,epsilon,explore

def main():
    """initialise important variables"""
    memory=Memory(100000)
    global_step=0
    epsilon=1.0
    
    gamma=0.9 # discounting rate
    
    learning_rate=0.0005
    epsilon_min=0.01
    epsilon_max=1.0
    decay_rate=0.00002#exponential decay rate for exploration probability
    
    BATCH_SIZE=32 #learning batch size
    total_episodes=200
    window_size=4

    history=[]
    
    """sets up the environment"""
    env=gym.make('ALE/SpaceInvaders-v5',render_mode='human')
    height,width,channels=env.observation_space.shape
    actions=env.action_space.n
    env.unwrapped.get_action_meanings()
    
    """make observations of the environment"""
    env.unwrapped.get_action_meanings()
    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((110,84), dtype=np.int32) for i in range(4)], maxlen=4)
    q_network=agent([110,84,4],actions,learning_rate)
    
    for i in range(total_episodes):
        done=False
        observation=env.reset()
        epoch=0
        episode_reward=0
        explore_count=0
        exploit_count=0
        episode_loss=[]
        
        #start by stacking frames
        observation,stacked_frames=stack_frames(stacked_frames,observation,True)
        
        while not done:
            action,epsilon,explore=policy(epsilon_max,epsilon_min,decay_rate,global_step,observation,actions,q_network)
            if explore:
                explore_count+=1
            else:
                exploit_count+=1
            next_obs,reward,done,_=env.step(action)
            
            next_obs,stacked_frames=stack_frames(stacked_frames,next_obs,False)
            
            memory.add([observation,action,next_obs,reward,done])
            
            """train the network every 4 steps so frames can be stacked"""
            if global_step%window_size==0 and global_step>0:
                batch_obs,batch_act,batch_next_obs,batch_rew,batch_done=memory.sample_memories(BATCH_SIZE)
                
                for i in range(len(batch_obs)):
                    o_obs=batch_obs[i]
                    o_next_act=q_network.predict(batch_next_obs[i])
                    y=batch_rew[i] + gamma * np.max(o_next_act, axis=-1) * (1-batch_done[i])
                    train_loss=q_network.train_on_batch(o_obs,y)
                    episode_loss.append(train_loss)
                    
            observation = next_obs
            epoch += 1
            global_step += 1
            episode_reward += reward
            
        next_obs=np.zeros(observation.shape)
        memory.add([observation, action, next_obs, reward, done])
        
        history.append(episode_reward)
        print("Epochs per episode:", epoch, "Episode Reward:", episode_reward,"Episode number:", len(history), " Times Explored:",explore_count," Times Exploited:",exploit_count)
        
    plt.plot(history)
    plt.show()
    
 
main()