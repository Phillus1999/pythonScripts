from time import time
from datetime import datetime

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


def read_csv(filename):
    #read log.csv first column as timestamps
    data = np.genfromtxt(filename, delimiter=',', usecols=0, dtype=str, skip_header=1)
    #convert timestamps to datetime objects
    data_converted = [datetime.strptime(x,  '"%M:%S.%f"') for x in data]
    #convert datetime objects to milliseconds
    data_converted_ms = [convert_to_milliseconds(x) for x in data_converted]
    return data_converted_ms
def convert_to_milliseconds(timestamp):
    return timestamp.minute * 60000 + timestamp.second * 1000 + timestamp.microsecond / 1000

def plot_graph():
    data = read_csv("log.csv")
    time_differences_seconds = evaluate_data(data)
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Set the desired figure size and aspect ratio
    fig = plt.figure(figsize=(20, 6))  # Width: 12 inches, Height: 6 inches
    ax = fig.add_subplot(111)
    ax.set_aspect('auto')

    #count datapoints above and below 8 seconds
    below = 0
    above = 0
    for x in time_differences_seconds:
        if x < 8:
            below += 1
        else:
            above += 1

    # display the counts in the top left corner
    ax.text(0.05, 0.95, 'below 8 seconds: ' + str(below), transform=ax.transAxes, fontsize=12,
            verticalalignment='top')
    ax.text(0.05, 0.90, 'above 8 seconds: ' + str(above), transform=ax.transAxes, fontsize=12,
            verticalalignment='top')


    # Add a horizontal red line at y = 6
    ax.axhline(y=8, color='red')

    # Plot the data with increased line width
    ax.plot(time_differences_seconds, linewidth=2.5)

    # Increase the font size of axis labels
    ax.set_xlabel('#consecrations', fontsize=12)
    ax.set_ylabel('time between uses', fontsize=12)

    # Increase the font size of tick labels
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.savefig('plot.png')
    plt.show()


def evaluate_data(data):
    # calculate time differences to the next entry
    time_differences = [data[i + 1] - data[i] for i in range(len(data) - 1)]
    print(time_differences)
    time_differences_seconds = [x / 1000 for x in time_differences]
    # remove first entry
    time_differences_seconds.pop(0)
    # remove all entries over 20 seconds
    time_differences_seconds = [x for x in time_differences_seconds if x < 20]
    return time_differences_seconds


def plot_histogramm():
    data = read_csv("log.csv")
    time_differences_seconds = evaluate_data(data)
    # Create a figure and axes
    fig, ax = plt.subplots()
    mu, sigma = st.norm.fit(time_differences_seconds)
    plt.plot(st.norm(mu, sigma).pdf(np.linspace(np.min(time_differences_seconds), np.max(time_differences_seconds), 30)))
    plt.hist(time_differences_seconds, bins=30, density=True)

    plt.xlabel('time between uses in seconds')
    plt.ylabel('probability density')

    plt.savefig('histogramm.png')
    plt.show()

if __name__ == "__main__":
    plot_histogramm()