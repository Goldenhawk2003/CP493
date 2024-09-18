import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the device
dev = qml.device('default.qubit', wires=5, shots=1024)

# Define the parameter list
theta_list = [0.01, 0.02, 0.03, 0.04, 0.05, 1.31, 1.32, 1.33, 1.34, 1.35]

# Define the quantum circuit
@qml.qnode(dev)
def circuit(theta):
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=4)
    qml.U3(theta, np.pi, np.pi, wires=1)
    qml.U3(theta, np.pi, np.pi, wires=4)
    qml.CSWAP(wires=[2, 1, 4])
    qml.Hadamard(wires=2)
    return qml.sample(qml.PauliZ(2))

# Function to execute the circuit and get the results
def execute_circuit(theta):
    result = circuit(theta)
    probability_1 = np.mean(result == 1)
    return probability_1

# List to store distances
distances = []

# Loop to compute distances for each theta value
for theta in theta_list:
    probability_1 = execute_circuit(theta)
    distance = 1 - 2 * probability_1  # Distance computation (based on measurement outcome)
    distances.append(distance)

    print(f'theta: {theta}, distance: {distance}')

# Plotting the distances
plt.figure(figsize=(10, 7))
plt.scatter(theta_list, distances, c='blue')
plt.xlabel('Theta')
plt.ylabel('Distance')
plt.title('Distances for Individual Theta Values')
plt.show()