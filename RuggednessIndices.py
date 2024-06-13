import math
import numpy as np


################################################################################
def signal_mean(sequence: list):
  return float(sum(sequence))/len(sequence)

def signal_std_dev(sequence: list):
  mean = signal_mean(sequence)
  variance = sum([((x - mean) ** 2) for x in sequence]) / float(len(sequence)-1)
  return variance ** 0.5

def subtract_avg_signal(sequence: list):
  avg_signal = float(sum(sequence))/len(sequence)
  return [signal - avg_signal for signal in sequence]


################################################################################
def first_differences_signal(sequence: list):
  return [sequence[i+1]-sequence[i] for i in range(len(sequence)-1)]

def normalized_number_of_sign_changes_in_first_differences(sequence: list):

  first_differences = first_differences_signal(sequence)

  sign_changes_counter = 0
  for i in range(len(first_differences)-1):
    if first_differences[i]*first_differences[i+1] < 0:
      sign_changes_counter += 1

  return sign_changes_counter/float(len(sequence)-2)


################################################################################
def normalizeSample_DFT_medianEnergy(sequence: list, energy_perc):

  DFT = np.fft.fft(sequence)
  mod_DFT = abs(DFT)
  half_mod_DFT = [mod_DFT[i] for i in range(math.ceil(len(mod_DFT)/2))]
  median_energy = sum(half_mod_DFT) * energy_perc

  temporary_sum = 0
  for i, sample in enumerate(half_mod_DFT):
    temporary_sum = temporary_sum + sample
    if temporary_sum >= median_energy:
      return i/float(len(half_mod_DFT)-1)


################################################################################
symbols = ["d", "e", "i"]

def get_information_string(sequence: list, epsilon_perc):
  string = ""
  epsilon = epsilon_perc * abs(signal_mean(sequence))
  for i in range(len(sequence)-1):
    difference = sequence[i+1]-sequence[i]
    if difference < -epsilon:
      string = string + "d"
    if abs(difference) <= epsilon:
      string = string + "e"
    if difference > epsilon:
      string = string + "i"
  return string

def get_blocks_frequency(sequence: list, epsilon_perc):
  # By "block" we mean a sequece of two symbols
  information_string = get_information_string(sequence, epsilon_perc)
  n = len(sequence)
  P = np.zeros((len(symbols),len(symbols)))
  for i,p in enumerate(symbols):
    for j,q in enumerate(symbols):
      P[i,j] = information_string.count(p+q)/n
  return P

def information_content_measure(sequence: list, epsilon_perc):
  information_content = 0
  P = get_blocks_frequency(sequence, epsilon_perc)
  for i,p in enumerate(symbols):
    for j,q in enumerate(symbols):
      p_ij = P[i,j]
      # p must be different from q
      # Furthermore, if p_ij=0, the log operation raise an error
      if p!=q and p_ij != 0:
        information_content += p_ij * math.log(p_ij, 6)
  return -information_content


################################################################################
def topography_ruggedness(sequences: list[list]):
  center_fitness = sequences[0][0]
  ruggedness = 0
  total_samples = 0
  for sequence in sequences:

    if sequence[0]!=center_fitness:
      raise ValueError("The fitness of the central point isn't equal for all the neighbors!!")

    for i,sample in enumerate(sequence):
      if i==0: continue

      total_samples = total_samples + 1
      ruggedness = ruggedness + (center_fitness - sample)**2

  return (ruggedness/total_samples)**0.5