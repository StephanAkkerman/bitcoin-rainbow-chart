import matplotlib.pyplot as plt
from plot import create_plot

from data import get_data


def main():
    # Load data
    raw_data, fitted_ydata, popt = get_data("data/bitcoin_data.csv")

    # Create plot
    create_plot(raw_data, fitted_ydata, popt)

    # Show plot
    plt.show()


if __name__ == "__main__":
    main()
