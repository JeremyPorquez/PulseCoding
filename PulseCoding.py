from scipy.linalg import hadamard, inv
import numpy as np

"""
Generate Simplex matrix from Hadamard matrix.

Based on 
    M. D. Jones, Using Simplex Codes to Improve OTDR Sensitivity,
    IEEE Photonics Technology Letters Vol. 15, No. 7, 822-824 (1993)

DOI Link:
    https://doi.org/10.1109/68.229819
    
Did modifications on scipy.linalg.hadamard algorithm to normalize.
A verication of the simplex generated code can be found in figure 1 of
    S. L. Floch, Colour Simplex coding for Brillouin distributed sensors,
    Proceedings of SPIE, 8794 (2013)
    
"""


def generateSimplex(n=2, withInverse=True):
    """
    :param n: order
    :type n: int
    :return: simplex matrix
    :rtype: array
    """
    n = int(n)
    hadamardMatrix = hadamard(2 ** n)

    # Modification on the scipy.linalg.hadamard code
    hadamardMatrix[hadamardMatrix == 1] = 0
    hadamardMatrix[hadamardMatrix == -1] = 1

    # Remove first rows and columns
    simplexMatrix = hadamardMatrix[1:, 1:]
    if withInverse:
        return simplexMatrix, inv(simplexMatrix)
    else:
        return simplexMatrix


def reconstructBGS(rawData, inverseMatrix, index=0):
    """
    :param rawData: raw oscilloscope data
    :type rawData: 2-D array
    :param inverseMatrix: inverse simplex matrix
    :type inverseMatrix: 2-D array
    :param index: index where BGS is extracted.
    :type index: int
    :return: Extracted BGS.
    :rtype: 1-D array
    """

    Sinv = inverseMatrix[index][np.newaxis, :]  # Makes inverse matrix a row matrix
    return np.matmul(Sinv, rawData)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # generate the simplex code
    n = 2
    s, s_inv = generateSimplex(n)

    # generate the test data parameters
    number_of_averages = 1
    number_of_bits = (2 ** n) - 1
    print(f"Number of bits: {number_of_bits}")
    data_points = 500
    pulse_width = 1
    amplitude = 3
    max_time = 100
    start_time = 0

    # simulate pulse
    bit_separation = (max_time - start_time) / number_of_bits
    pulse_locations = bit_separation * np.arange(s.shape[1])

    # initialize the test data
    test_data = np.zeros((s.shape[0], data_points))
    time_axis = np.linspace(start_time, max_time, data_points)

    for _n in range(number_of_bits):
        for bit_index, bit in enumerate(s[_n]):
            for a in range(number_of_averages):
                for idx, t in enumerate(time_axis):
                    test_data[_n][idx] += (amplitude * bit) * np.exp((-(t - pulse_locations[bit_index]) ** 2) / ((2 * pulse_width) ** 2)) + np.random.random()
    test_data /= number_of_averages

    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Pulse coding raw data. Bit length {number_of_bits}')
    for _n in range(number_of_bits):
        ax1.plot(time_axis, test_data[_n] + _n*amplitude)

    ax2 = fig.add_subplot(312)
    ax2.set_title(f"Pulse coding reconstructed")
    reconstructed = reconstructBGS(test_data, s_inv, index=0)
    ax2.set_ylim(0, amplitude * 1.5)
    ax2.plot(time_axis, reconstructed[0])

    ax3 = fig.add_subplot(313)
    ax3.set_title(f'Averaging {number_of_averages * number_of_bits} times')
    averaged = np.zeros((number_of_averages * number_of_bits, data_points))
    for i in range(0, number_of_averages * number_of_bits):
        averaged[i] = amplitude * np.exp((-(time_axis - 0) ** 2) / ((2 * pulse_width) ** 2)) + np.random.random(np.zeros((data_points)).shape)
    # averaged /= number_of_averages * number_of_bits
    ax3.set_ylim(0, amplitude * 1.5)
    ax3.plot(time_axis, np.sum(averaged,axis=0)/averaged.shape[0])

    print(f"SNR 1 meaurement: {np.mean(averaged[0]) / np.std(averaged[0])}")
    print(f"SNR Pulse coding: {np.mean(reconstructed[0])/np.std(reconstructed[0])}")
    print(f"SNR Averaging: {np.mean(averaged) / np.std(averaged)}")
    plt.show()
