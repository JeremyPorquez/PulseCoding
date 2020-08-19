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
    number_of_combinations = (2 ** n) - 1
    data_points = 500
    pulse_width = 1
    amplitude = 3
    max_time = 100
    start_time = 0

    # simulate pulse
    bit_separation = (max_time - start_time) / number_of_combinations
    pulse_locations = bit_separation * np.arange(s.shape[1])

    # initialize the test data
    test_data = np.zeros((s.shape[0], data_points))
    time_axis = np.linspace(start_time, max_time, data_points)

    for _n in range(number_of_combinations):
        for bit_index, bit in enumerate(s[_n]):
            for idx, t in enumerate(time_axis):
                test_data[_n][idx] += (amplitude * bit) * np.exp((-(t - pulse_locations[bit_index]) ** 2) / ((2 * pulse_width) ** 2)) + np.random.random()


    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    for _n in range(number_of_combinations):
        ax1.plot(time_axis, test_data[_n] + _n*amplitude)

    ax2 = fig.add_subplot(212)
    reconstructed = reconstructBGS(test_data, s_inv, index=0)
    ax2.plot(time_axis, reconstructed[0])

    print(f"SNR : {np.mean(reconstructed[0])/np.std(reconstructed[0])}")

    plt.show()
