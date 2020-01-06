#!/usr/bin/env python3
import sys, gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

def createNetwork( activation, nLayers, nNodes ):
    """
    The acrobot has has 6 inputs: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
    where theta1 and 2 are the two rotational joint angles, and thetaDot1 and 2 are the joint angular velocities
    The acrobot has 3 actions: applying +1, 0 or -1 torque on the joint between the two pendulum links
    """
    # https://www.tensorflow.org/api_docs/python/tf/keras
    model = keras.models.Sequential( )
    model.add( keras.layers.InputLayer( batch_input_shape=(1, 6) ) )
    for i in range(nLayers):
        model.add( keras.layers.Dense( nNodes, activation=activation ) )
    model.add( keras.layers.Dense( 3, activation="softmax" ) )
    model.compile( loss="mse",
                   optimizer="adam",
                   metrics=[ "mae" ] )
    return model

def trainModel(model, env, render, nEpochs):
    # keep gamma large (0.95 - 0.99) but not 1.0.
    # this allows reward from a win to trickle back to earlier states.
    gamma = 0.95
    # starting percentage of chance of random action
    epsilon = 0.25
    decayFactor = 0.98
    for i in range(nEpochs):
        if render:
            print( "epoch:", i )
        s = env.reset()
        # the environment's state is a list [ cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2 ]
        # we reshape it to be useful input to tensorflow to be [ [ cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2 ] ]
        s = np.reshape( s, [ 1, 6 ] )
        # reduce probability of random actions.
        epsilon *= decayFactor
        # finish preparations for the epoch
        done = False
        reward = 0
        while not done:
            if render:
                env.render()
            # choose action randomly, or based on the model's predicted value.
            if np.random.random( ) < epsilon:
                a = env.action_space.sample( )
            else:
                a = np.argmax( model.predict( s ) )
            s1, r, done, _ = env.step( a )

            #
            # reward engineering
            #
            tipHeight = -np.cos(env.state[0]) - np.cos(env.state[1] + env.state[0]) 
            # initial reward as the height of the tip of the second link (range of: (-2, 1), where 1 is the goal)
            myReward = tipHeight - .5
            if r == 0: # if the goal has been reached, give a significant reward
                myReward += 25
            theta1Cos = s1[0]
            if theta1Cos <= 0 and tipHeight > 0: # if the first link is parallel to the goal line and the tip of the second link is above the second link
                # give a decent reward, since this state is required for reaching the goal
                myReward += 5
            thetaDot2 = s1[-1]
            # provide a reward to get the second link moving as fast as possible to reach the goal
            myReward += abs(thetaDot2)

            # Again, reshape the state vector to be correct for tensorflow's format
            s1 = np.reshape( s1, [ 1, 6 ] )
            # This is what we want the utility for action a in state s to be.
            target = myReward + gamma * np.max( model.predict( s1 ) )
            # This is the current array of utilities for all actions in state s.
            target_vec = model.predict( s )[ 0 ]
            # Change for action a's value to be target
            target_vec[ a ] = target
            # Cause the network to update its weights to attempt to give this target value.
            # There are 3 values (one for each action) in target_vect.  We reshape it to be correct for tensorflow's purposes.
            model.fit( s, target_vec.reshape( -1, 3 ), epochs=1, verbose=0 )
            # tally the total reward for the epoch, and update the current state
            reward += r
            s = s1
            if render and done:
                print( "reward:", reward )
                print( "state:", s )
        if render:
            env.render()
    env.close( )

def testModel(model, env, render, nEpochs):
    totalRewards = 0
    for i in range( nEpochs ):
        s = env.reset()
        done = False
        epochReward = 0
        while not done:
            if render:
                env.render()
            s = np.reshape( s, [ 1, 6 ] ) #reshape the state vector to be correct for tensorflow's format
            a = np.argmax( model.predict( s ) )
            s, r, done, _ = env.step( a )
            epochReward += r 
        totalRewards += epochReward
        if render:
            env.render()
    env.close( )
    return totalRewards/nEpochs

def evaluateModelOptions(activation, nLayersOpt, nNodesOpt, env, render, nTrainingEpochs, nTestingEpochs):
    print("n training epochs:", nTrainingEpochs, ", n testing epochs:", nTestingEpochs)
    for nLayers in nLayersOpt:
        for nNodes in nNodesOpt:
            # Create a new network
            model = createNetwork(activation, nLayers, nNodes)
            trainModel(model, env, render, nTrainingEpochs)
            averageReward = testModel(model, env, render, nTestingEpochs)
            print("n layers:", nLayers,", Nodes per layer:", nNodes, ", Average reward per epoch:", averageReward)
    
def main( argv ):
    # Choose whether to render the simulation or not
    render = False
    if "render" in argv:
        render = True
        argv.remove( "render" )

    # Choose the activation function to use, linear by default
    activations = set( [ "elu", "exponential", "selu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear" ] )
    if len( argv ) > 1:
        activation = argv[ 1 ]
        if activation not in activations:
            print( str( activation ) + " is not a known activation: " + str( activations ) )
            sys.exit( 1 )
    else:
        activation = "linear"
    print(activation)

    # Choose an environment
    env = gym.make("Acrobot-v1")
    nLayersOpt = [1, 2, 3, 4]
    nNodesOpt = [10, 20, 40, 80]
    evaluateModelOptions(activation, nLayersOpt, nNodesOpt, env, render, 150, 500)

    return

if __name__ == "__main__":
    main( sys.argv )