'''
Utility functions implemented in the rest of the project.
'''

#################################################### IMPORTS ###########################################################
import math

#################################################### UTILS #############################################################
# Turns cumulative log probabilities into relative probabilities
def standardize(probabilities):
    exps = [math.exp(x) for x in probabilities]
    total = sum(exps)
    return [x / total for x in exps]
