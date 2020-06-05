import numpy as np
import matplotlib.pyplot as plt
import time


import value_based, policy_based, MC_method

def main():
    value_based.main()
    time.sleep(1)
    print('\n')
    policy_based.main()
    time.sleep(1)
    print('\n')
    MC_method.main()


if __name__ == '__main__':
    main()