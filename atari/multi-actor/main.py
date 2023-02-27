import multiprocessing, threading, os
from atari import Atari
from agent import LearnerAgentPolicy, LearnerAgentRND, LearnerAgentEmbedding, ActorAgent
from memory import ExperienceReplayMemory
# do not import tensorflow here, as it will break the multiprocessing library

def train(game, agentName, loadPolicy, loadRND, loadEmbedding, cpuCount, replayMemSize, expChanCap, render, actionSpace, enableLearnerGPU):

    # each actor gets access to a central shared best score tracker
    oracleScore = multiprocessing.Value( "i", 0 )
    # each actor gets access to the experience queue
    expChan = multiprocessing.Queue( expChanCap )
    # each actor gets access to a fresh weights queue
    weightsChan = multiprocessing.Queue( 1 )

    for actorID in range( cpuCount ):
        actor = ActorAgent(game, expChan, weightsChan, actorID, cpuCount, oracleScore, 
                           render=(actorID == (cpuCount-1) and render), actionSpace=actionSpace)

        proc = multiprocessing.Process(target=actor.explore)
        # start actor process upon creation
        proc.start()

        print("Actor", actorID, "started")

    # create and start loading memory from actors (policy)
    learnerMemory = ExperienceReplayMemory(expChan, size=replayMemSize)
    memLoadThread = threading.Thread(target=learnerMemory.load_nolock)
    memLoadThread.start()

    learnerPolicy = LearnerAgentPolicy(learnerMemory, agentName, weightsChan, load=loadPolicy, actionSpace=actionSpace, enableGPU=enableLearnerGPU)
    learnerEmbedding = LearnerAgentEmbedding(learnerMemory, agentName, weightsChan, load=loadEmbedding, actionSpace=actionSpace, enableGPU=enableLearnerGPU)
    learnerRND = LearnerAgentRND(learnerMemory, agentName, weightsChan, load=loadRND, actionSpace=actionSpace, enableGPU=enableLearnerGPU)

    # start policy learner
    threadPolicy = threading.Thread(target=learnerPolicy.learn)
    threadPolicy.start()
    # start embedding learner
    threadEmbedding = threading.Thread(target=learnerEmbedding.learn)
    threadEmbedding.start()
    # start rnd learner
    threadRND = threading.Thread(target=learnerRND.learn)
    threadRND.start()

    # join with policy learner
    threadPolicy.join()

def main():
    # A list of all possible Atari games
    games = [
        "Adventure", # 0
        "AirRaid", # 1
        "Alien", # 2
        "Amidar", # 3
        "Assault", # 4
        "Asterix", # 5
        "Asteroids", # 6
        "Atlantis", # 7
        "BankHeist", # 8
        "BattleZone", # 9
        "BeamRider", # 10
        "Berzerk", # 11
        "Bowling", # 12
        "Boxing", # 13
        "Breakout", # 14
        "Carnival", # 15
        "Centipede", # 16
        "ChopperCommand", # 17
        "CrazyClimber", # 18
        "Defender", # 19
        "DemonAttack", # 20
        "DoubleDunk", # 21
        "ElevatorAction", # 22
        "Enduro", # 23
        "FishingDerby", # 24
        "Freeway", # 25
        "Frostbite", # 26
        "Gopher", # 27
        "Gravitar", # 28
        "Hero", # 29
        "IceHockey", # 30
        "Jamesbond", # 31
        "JourneyEscape", # 32
        "Kangaroo", # 33
        "Krull", # 34
        "KungFuMaster", # 35
        "MontezumaRevenge", # 36
        "MsPacman", # 37
        "NameThisGame", # 38
        "Phoenix", # 39
        "Pitfall", # 40
        "Pong", # 41
        "Pooyan", # 42
        "PrivateEye", # 43
        "Qbert", # 44
        "Riverraid", # 45
        "RoadRunner", # 46
        "RobotTank", # 47
        "Seaquest", # 48
        "Skiings", # 49
        "Solaris", # 50
        "SpaceInvaders", # 51
        "StarGunner", # 52
        "Tennis", # 53
        "TimePilot", # 54
        "Tutankham", # 55
        "UpNDown", # 56
        "Venture", # 57
        "VideoPinball", # 58
        "WizardofWor", # 59
        "Zaxxon" # 60
    ]
    option = 14
    game = Atari( games[option]+"Deterministic-v4" )
    agentName = "atari_agent_" + games[option]

    actionSpace = game.getActionSpace()

    # set to None if no model to load
    loadPolicy = None
    #loadPolicy = "models/atari_agent_" + games[option] + "_best.h5"
    #loadPolicy = "atari_agent_" + games[option] + "_policy.h5"
    loadRND = None
    #loadRND = "models/atari_agent_" + games[option] + "_rnd_best.h5"
    #loadRND = "atari_agent_" + games[option] + "_rnd.h5"
    loadEmbedding = None
    #loadEmbedding = "models/atari_agent_" + games[option] + "_embedding_best.h5"
    #loadEmbedding = "atari_agent_" + games[option] + "_embedding.h5"

    render = False # Currently broken, do not enable
    enableLearnerGPU = True
    cpuCount = os.cpu_count()
    # accounts for majority of the memory used by program
    replayMemSize = 2**17
    # simply ensures that the experience chan doesn't keep growing infinitely
    expChanCap = 256

    train(game, agentName, loadPolicy, loadRND, loadEmbedding, cpuCount, replayMemSize, expChanCap, render, actionSpace, enableLearnerGPU)


if __name__ == "__main__":
    main()