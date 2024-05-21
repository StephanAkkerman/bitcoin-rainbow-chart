import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit

# Define colors and labels for the rainbow bands
colors_labels = {
    "#c00200": "Maximum bubble territory",
    "#d64018": "Sell. Seriously, SELL!",
    "#ed7d31": "FOMO Intensifies",
    "#f6b45a": "Is this a bubble?",
    "#feeb84": "HODL!",
    "#b1d580": "Still cheap",
    "#63be7b": "Accumulate",
    "#54989f": "BUY!",
    "#4472c4": "Fire sale!",
}
# Colors and labels from https://www.coinglass.com/pro/i/bitcoin-rainbow-chart


# this is your log function
def logFunc(x, a, b, c):
    return a * np.log(b + x) + c


def get_data():
    # Load data
    raw_data = pd.read_csv("bitcoin_data.csv")

    raw_data["Date"] = pd.to_datetime(
        raw_data["Date"]
    )  # Ensure that the date is in datetime or graphs might look funny
    raw_data = raw_data[
        raw_data["Value"] > 0
    ]  # Drop all 0 values as they will mess up the regression bands

    # getting your x and y data from the dataframe
    xdata = np.array([x + 1 for x in range(len(raw_data))])
    ydata = np.log(raw_data["Value"])

    # here we ar fitting the curve, you can use 2 data points however I wasn't able to get a graph that looked as good with just 2 points.
    popt, _ = curve_fit(logFunc, xdata, ydata)
    # This is our fitted data, remember we will need to get the ex of it to graph it
    fittedYData = logFunc(xdata, popt[0], popt[1], popt[2])

    return raw_data, fittedYData


# Create a wider plot with a dark background
fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("#0d1117")  # Set figure face color
ax.set_facecolor("#0d1117")  # Set axes face color

# Draw the rainbow bands and create legend handles
legend_handles = []

raw_data, fittedYData = get_data()


def plot_rainbow(
    ax, fittedYData, raw_data, num_bands=9, band_width=0.3, legend_handles=[]
):
    for i in range(num_bands):
        # Higher values decreases the bands (so the rainbow is lower on the plot)
        i_decrease = 1.5
        lower_bound = np.exp(fittedYData + (i - i_decrease) * band_width - band_width)
        upper_bound = np.exp(fittedYData + (i - i_decrease) * band_width)
        color = list(colors_labels.keys())[::-1][i]
        label = list(colors_labels.values())[::-1][i]
        ax.fill_between(
            raw_data["Date"],
            lower_bound,
            upper_bound,
            alpha=1,  # 0.8
            color=color,
            label=label,
        )
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=color, alpha=1, label=label)
        )


plot_rainbow(ax, fittedYData, raw_data, legend_handles=legend_handles)


def plot_price(ax, raw_data):
    # Plot in a with long Y axis and set the color of the Bitcoin price data
    (btc_line,) = ax.semilogy(
        raw_data["Date"], raw_data["Value"], color="white", label="BTC price"
    )  # Change color to white for contrast

    return btc_line


btc_line = plot_price(ax, raw_data)


# Custom formatter function to display y-axis labels as $0.01, $0.1, $1, $10, $1K, $10K, etc.
def y_format(y, _):
    if y < 1:
        return f"${y:.2f}"
    elif y < 10:
        return f"${y:.1f}"
    elif y < 1_000:
        return f"${int(y):,}".replace(",", ".")
    elif y < 1_000_000:
        return f"${y/1_000:.1f}K".replace(".0K", "K").replace(".", ",")
    else:
        return f"${y/1_000_000:.1f}M".replace(".0M", "M").replace(".", ",")


# Apply the custom formatter to the y-axis
formatter = FuncFormatter(y_format)
ax.yaxis.set_major_formatter(formatter)

# Set the minimum y-axis value to $0.01
ax.set_ylim(bottom=0.01)

# Set the x-axis limits to fit the data without gaps
ax.set_xlim([raw_data["Date"].min(), raw_data["Date"].max()])

# Change the color of the tick labels
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")

# Add the legend above the plot, centered, in a single row
handles = [btc_line] + legend_handles[::-1]  # Reversed order for legend
ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=len(handles),
    frameon=False,
    fontsize="small",
    labelcolor="white",
)

# Show the plot in wide format
plt.show()
