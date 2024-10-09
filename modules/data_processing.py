import re
import numpy as np
from numpy.polynomial import Chebyshev
# import matplotlib.pyplot as plt

"""
Opens the file 'filename', checks its validity, formats it into an 2D numpy array and returns this array.
"""

def get_data(filename):
    return format_data(validate_data(load_file(filename)))


"""
Opens the file 'filename' and returns an array containing the lines of the file.
"""
def load_file(filename):
    if filename == "":
        raise Exception("Error in function 'load_file': filename is empty")
    with open(filename, "r") as file:
        lines = []
        for line in file.readlines():
            lines.append(line)
        return lines


"""
Validates the entries of the array 'data', removes the invalid lines and returns the valid entries.
The entries must contain two decimal numbers separated by either ' ', '\t' or ',' to be counted as valid.
"""
def validate_data(data):
    if data == []:
        raise Exception("Error in function 'validate_data': data array is empty")

    length = len(data)
    for i in range(length):
        if re.match("^\s*\d+\.?\d*(?:\s*|\,*|\t*)-?\d+\.?\d*(?:[eE][+\-]?\d+)?\s*(\\n?)$", data[length - i - 1]) is None:
            data.pop(length - i - 1)
    if len(data) == 0:
        raise Exception("Error in function 'validate_data': array has no valid entry")
    return data


"""
Formats the entries of the array 'data' by splitting it into two decimal numbers, creates a 2D numpy array, 
fills the array with the decimal numbers and returns the array.
"""
def format_data(data):
    formatted_data = np.array([[0.0, 0.0] for _ in range(len(data))])
    for i in range(len(data)):
        formatted_data[i] = re.split('[\s*|\,*|\t*]', data[i].strip())
    return formatted_data


"""
Fits the array 'data' with a Chebyshev polynom of degree 'degree' and returns the fitted function.
"""
def fit_data(data, degree=5):
    try:
        return Chebyshev.fit(data[:, 0], data[:, 1], degree)
    except Exception as e:
        print("Error in 'fit_data()': {}".format(e))


"""
Interpolates the function 'fit' by evaluating it either at the entries of xdata or at every delta_x, from x_min to x_max
and returns the interpolated data in a 2D numpy array.
"""
def interpolate_data(fit, xdata, x_min=None, x_max=None, delta_x=None):
    if xdata is not None or (x_min is not None and x_max is not None and delta_x is not None):
        if xdata is not None:
            interpolated_data = np.array([[0.0, 0.0] for _ in range(len(xdata))])
            for i in range(len(xdata)):
                interpolated_data[i] = [xdata[i], fit(xdata[i])]
        else:
            length = int(((x_max - x_min) / delta_x) + 1)
            interpolated_data = np.array([[0.0, 0.0] for _ in range(length)])
            for i in range(length):
                x = x_min + i * delta_x
                interpolated_data[i] = [x, fit(x)]
        return interpolated_data
    else:
        raise Exception("Error in 'interpolate_data()': xdata oder x_min, x_max, delta_x muss übergeben werden")

def save_data(path, data, x_legend="", y_legends=[]):
    # Check if saving was aborted
    if(path == ""):
        return
    
    x_array = data[0]
    y_arrays = data[1:]
    with open(path, "w") as file:
        legend = x_legend
        if (legend != ""):
            legend += "\t"
        for y_legend in y_legends:
                legend += f"{y_legend}\t"
        legend = legend.strip()
        legend +="\n"
        file.write(legend)
        # file.write(f"{x_legend}")
        # if (len(y_arrays) != 0):
        #     for y_legend in y_legends:
        #         file.write(f"\t{y_legend}")
        #     file.write(f"\n")

        for i in range(max([max([len(col) for col in data]), len(x_array)])):
            if (len(x_array) > i):
               file.write(f"{x_array[i]}")
            for y_array in y_arrays:
                if (len(y_array) > i):
                    file.write(f"\t{y_array[i]}")
                else:
                    file.write(f"\t")
            file.write(f"\n")

def save_params(path, para_dict):
    # Check if saving was aborted
    if(path == ""):
        return
    
    with open(path, "w") as file:
        for key in para_dict:
            file.write(f"{key}\t{para_dict[key]}\n")
"""
Shows the x and y data given in '*args'.
"""


# def plot_data(*args):
#     if len(args) % 2 == 1:
#         print("Error in 'plot_data()': Number of *args must be even")
#         return
#     for i in range(0, len(args), 2):
#         plt.plot(args[i], args[i + 1])
#     plt.show()


if __name__ == "__main__":
    string = "a\t\t   ,\tb"
    print(string.split())
    # # Examples:
    # data = get_data("test.dat")
    # c = fit_data(data, degree=2)
    # # new_data = interpolate_data(c, None, 300, 500, 50)
    # new_data = interpolate_data(c, data[:, 0])
    # # plot_data(data[:, 0], data[:, 1], data[:, 0], c(data[:, 0]), new_data[:, 0], new_data[:, 1])
