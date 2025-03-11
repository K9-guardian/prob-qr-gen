#!/usr/bin/env python3
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pprint import pp
from tabulate import tabulate
from IPython.display import display, Math, Latex


# In[ ]:


fake_to_real = {
    "amavon": "amazon",
    "chajgpt": "chatgpt",
    "facibook": "facebook",
    "gooale": "google",
    "insjagram": "instagram",
    "redbit": "reddit",
    "whajsapp": "whatsapp",
    "wikepedia": "wikipedia",
    "yahio": "yahoo",
    "youjube": "youtube"
}


# In[ ]:


# Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Cleanup
df = df.rename(columns={"Brand": "Fake"})
df.insert(1, 'Real', df['Fake'].map(fake_to_real))

# Display the DataFrame
print(df)


# In[ ]:


# Merge the trials into a single array
grouped = df.groupby(['Real', 'Scanner ID', 'Module Size', 'Luminosity'])

rows = []
for _, group in grouped:
    trial = group['Outcome (T/F)'].map({'T': 1, 'F': 0}).to_numpy()
    trials = pd.Series([trial], index=['Trials'])
    row = pd.concat([group.iloc[0].drop(['Trial Number', 'Outcome (T/F)']), trials])
    rows.append(row)

df_comp = pd.DataFrame(rows)
print(df_comp)


# In[ ]:


# Probability of Fake Scans between URLs

# Prepare data for box plots
data = []
labels = []

urls = df_comp.groupby('Real')

for url, group in urls:
    prob_fake = group['Trials'].map(lambda x: (1 - x).mean()).to_numpy()
    prob_fake = prob_fake + 1e-10  # Add a small offset to avoid log(0)
    data.append(prob_fake)
    labels.append(url)

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels)
plt.title('Probability of Fake Scans between URLs')
plt.xlabel('URL')
plt.ylabel('Probability Fake')
plt.yscale('log')
plt.xticks(rotation=45)  # Rotate labels if they are long
plt.grid(True)
plt.show()


# In[ ]:


# Probability of URLs between Scanners

table = {}

grouped = df_comp.groupby(['Scanner ID', 'Real'])

for (scanner, url), group in grouped:
    if scanner not in table:
        table[scanner] = {}
    prob_fake = group['Trials'].map(lambda x: (1 - x).mean()).to_numpy().mean()
    # prob_fake = prob_fake + 1e-10  # Add a small offset to avoid log(0)
    table[scanner][url] = prob_fake

# Function to format values (bold if 0.0000)
def format_value(value):
    return f"\\textbf{{{value:.4f}}}" if value < 0.05 else f"{value:.4f}"

# Print table header
s = ""
s += "\\begin{table}\n"
s += "\\begin{tabular}{lccccccccccc}\n"
s += "\\toprule\n"
s += "\\textbf{Scanner} & \\textbf{amazon} & \\textbf{chatgpt} & \\textbf{facebook} & \\textbf{google} & \\textbf{instagram} & \\textbf{reddit} & \\textbf{whatsapp} & \\textbf{wikipedia} & \\textbf{yahoo} & \\textbf{youtube} \\\\\n"
s += "\\midrule\n"

# Print table rows
for app, values in table.items():
    row = [app.replace('&', '\&')]
    for key in ['amazon', 'chatgpt', 'facebook', 'google', 'instagram', 'reddit', 'whatsapp', 'wikipedia', 'yahoo', 'youtube']:
        row.append(format_value(values[key]))
    s += " & ".join(row) + " \\\\\n"

# Print table footer
s += "\\bottomrule\n"
s += "\\end{tabular}\n"
s += "\\caption{Mean Probabilities of Fake Scans of Apps Between URLs}\n"
s += "\\label{tab:camera_app_performance}\n"
s += "\\end{table}"

print(s)


# In[ ]:


# TODO: Investigate why chatgpt and youtube don't follow the rest. Could be localized to a scanner, or based on burst error


# In[ ]:


# Number of probabilistic black and white bits for brightness analsysis

prob_bit_counts = [
    [ "gooale"    ,1 ,( ( 'prob_black', 0 ), ( 'prob_white', 1 ) ) ],
    [ "gooale"    ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "gooale"    ,3 ,( ( 'prob_black', 1 ), ( 'prob_white', 2 ) ) ],
    [ "youjube"   ,1 ,( ( 'prob_black', 0 ), ( 'prob_white', 1 ) ) ],
    [ "youjube"   ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "youjube"   ,3 ,( ( 'prob_black', 2 ), ( 'prob_white', 1 ) ) ],
    [ "facibook"  ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "facibook"  ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "facibook"  ,3 ,( ( 'prob_black', 2 ), ( 'prob_white', 1 ) ) ],
    [ "insjagram" ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "insjagram" ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "insjagram" ,3 ,( ( 'prob_black', 0 ), ( 'prob_white', 3 ) ) ],
    [ "whajsapp"  ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "whajsapp"  ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "whajsapp"  ,3 ,( ( 'prob_black', 2 ), ( 'prob_white', 1 ) ) ],
    [ "wikepedia" ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "wikepedia" ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "wikepedia" ,3 ,( ( 'prob_black', 1 ), ( 'prob_white', 2 ) ) ],
    [ "chajgpt"   ,1 ,( ( 'prob_black', 0 ), ( 'prob_white', 1 ) ) ],
    [ "chajgpt"   ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "chajgpt"   ,3 ,( ( 'prob_black', 3 ), ( 'prob_white', 0 ) ) ],
    [ "redbit"    ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "redbit"    ,2 ,( ( 'prob_black', 1 ), ( 'prob_white', 1 ) ) ],
    [ "redbit"    ,3 ,( ( 'prob_black', 2 ), ( 'prob_white', 1 ) ) ],
    [ "yahio"     ,1 ,( ( 'prob_black', 0 ), ( 'prob_white', 1 ) ) ],
    [ "yahio"     ,2 ,( ( 'prob_black', 0 ), ( 'prob_white', 2 ) ) ],
    [ "yahio"     ,3 ,( ( 'prob_black', 1 ), ( 'prob_white', 2 ) ) ],
    [ "amavon"    ,1 ,( ( 'prob_black', 1 ), ( 'prob_white', 0 ) ) ],
    [ "amavon"    ,2 ,( ( 'prob_black', 2 ), ( 'prob_white', 0 ) ) ],
    [ "amavon"    ,3 ,( ( 'prob_black', 3 ), ( 'prob_white', 0 ) ) ]
]

prob_bit_df = pd.DataFrame(prob_bit_counts)
prob_bit_df[0] = prob_bit_df[0].map(fake_to_real)
print(prob_bit_df)


# In[ ]:


grouped = prob_bit_df.groupby(2)

def line_title(t):
    return f'{t[0][1]}B:{t[1][1]}W'

data = {}
for name, group in grouped:
    points = np.zeros(3)
    for row in group.itertuples():
        vals = df_comp.query(f'Real == "{row._1}" & `Module Size` == {row._2}')

        # Get mean probs for all 3 luminosities
        prob_fake_0 = vals.query('Luminosity == 0')['Trials'].map(lambda x: (1 - x).mean()).to_numpy().mean()
        prob_fake_1 = vals.query('Luminosity == 1')['Trials'].map(lambda x: (1 - x).mean()).to_numpy().mean()
        prob_fake_2 = vals.query('Luminosity == 2')['Trials'].map(lambda x: (1 - x).mean()).to_numpy().mean()

        points[0] += prob_fake_0
        points[1] += prob_fake_1
        points[2] += prob_fake_2
    points = points / len(group)
    data[line_title(name)] = points

# X-axis values (only 3 points)
x_values = [0, 1, 2]

# Plotting
plt.figure(figsize=(10, 6))

for label, values in data.items():
    plt.plot(x_values, values, label=label)

# Set x-axis ticks to only show 0, 1, 2
plt.xticks(x_values)

# Add labels, title, and legend
plt.xlabel("Luminosity")
plt.ylabel("Probability Fake")
plt.title("Probability of Fake Scans at Different Luminosities")
plt.legend()

# Show the plot
plt.show()


# In[ ]:




