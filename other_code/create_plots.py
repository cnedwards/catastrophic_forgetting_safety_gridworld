
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=100): #used this approach https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

    
exec(open("pytorch_dqn_agent_mylava_taskA.py").read())
steps_at_episode1A = steps_at_episode
episode_return1A = episode_return
exec(open("pytorch_dqn_agent_mylava_taskB.py").read())
steps_at_episode1B = steps_at_episode
episode_return1B = episode_return
EWC_losses1 = EWC_losses
exec(open("pytorch_dqn_agent_mylava_taskA.py").read())
steps_at_episode2A = steps_at_episode
episode_return2A = episode_return
exec(open("pytorch_dqn_agent_mylava_taskB.py").read())
steps_at_episode2B = steps_at_episode
episode_return2B = episode_return
EWC_losses2 = EWC_losses
exec(open("pytorch_dqn_agent_mylava_taskA.py").read())
steps_at_episode3A = steps_at_episode
episode_return3A = episode_return
exec(open("pytorch_dqn_agent_mylava_taskB.py").read())
steps_at_episode3B = steps_at_episode
episode_return3B = episode_return
EWC_losses3 = EWC_losses
    

plt.figure()
#plt.plot(file1[file1.files[0]], file1[file1.files[1]])
plt.plot(steps_at_episode1A[99:], moving_average(episode_return1A))
plt.plot(steps_at_episode2A[99:], moving_average(episode_return2A))
plt.plot(steps_at_episode3A[99:], moving_average(episode_return3A))
plt.xlabel("Steps")
plt.ylabel("Return")
plt.savefig('print_code/outA_.png')

plt.figure()
#plt.plot(file1[file1.files[0]], file1[file1.files[1]])
plt.plot(steps_at_episode1B[99:], moving_average(episode_return1B))
plt.plot(steps_at_episode2B[99:], moving_average(episode_return2B))
plt.plot(steps_at_episode3B[99:], moving_average(episode_return3B))
plt.xlabel("Steps")
plt.ylabel("Return")
plt.savefig('print_code/outB_.png')

plt.figure()
plt.plot(EWC_losses1)
plt.plot(EWC_losses2)
plt.plot(EWC_losses3)
plt.xlabel("Loss Calculations")
plt.ylabel("EWC Loss")
plt.savefig('print_code/EWC_.png')