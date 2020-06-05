import numpy as np
import matplotlib.pyplot as plt


def main():
    pass
alpha = 0.5

dict_action = {
    1: 'left',
    2: 'hold_still',
    3: 'right'
}

# reward defination

def left(s):
    if s == -10:
        s = -10
        reward = -11
    elif s == 1:
        s = 0
        reward = 99
    else:
        s = s - 1
        reward = -1 
    return s, reward

def right(s):
    if s == 10:
        s = 10
        reward = -11
    elif s == -1:
        s = 0
        reward = 99
    else:
        s = s + 1
        reward = -1 
    return s, reward

def hold_still(s):
    if s == 0:
        return s, 99
    else:
        return s, -1

def get_q_array(v): # after solving the Bellman equation, we get v(s) = reward + alpha * v(s') 
    q_array = np.zeros([21, 3]) # -10 to 10
    coords = [k for k in range(-10, 11)]
    for i in range(21):
        for j in range(3): # j = 0, 1, 2 to match the dictionary, we have num = j + 1
            s = coords[i]
            num = j + 1
            if num == 1: # left
                s_prime, reward = left(s)
            elif num ==2: # hold_still
                s_prime, reward = hold_still(s)
            else: # right
                s_prime, reward = right(s)
            q_array[i, j] = reward + alpha * v[coords.index(s_prime)]
    return q_array

def get_policy(q_array):
    return np.argmax(q_array, axis=1) + 1 # 1: left, 2: hold_still, 3: right, find the max value in each row
def update_v(pi, q_array):
    v = np.zeros([21, 1])
    for i in range(21):
        v[i] = q_array[i, pi[i]-1]
    return v


def main():
    # initialize
    q_array = np.zeros([21, 3])
    v = np.zeros([21, 1]) # action values

    diff_norm = np.float(100) # a very large number 
    eps = np.float(10**-6)
    count = 0
    print('value-based method bgeins!')
    while(diff_norm >= eps):
        v_pre = v
        q_array = get_q_array(v)
        pi = get_policy(q_array) # pi is policy
        v = update_v(pi, q_array)
        diff_norm = np.linalg.norm(v - v_pre) # we use Euclidean distance as the loss evaluator
        count = count + 1


    print('done!')
    print('action value matrix: ')
    print(q_array)
    print('state values:')
    print(v)
    print('the optimal policy is: ')
    print(pi)
    print('time of iterations:', count, ' alpha =', alpha)

    fig = plt.figure()
    x = [k for k in range(-10, 11)]
    plt.scatter(x, pi)
    plt.title('value_based_method')
    plt.ylabel('action')
    plt.xlabel('s')
    plt.show()

if __name__ == '__main__':
    main()


