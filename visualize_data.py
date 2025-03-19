#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pprint import pp
from tabulate import tabulate
from IPython.display import display, Math, Latex
from scipy import stats


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


scanner_to_platform = {
    "Android Camera": "android",
    "Apple Camera": "apple",
    "Komorebi": "apple",
    "Mixinbox QR Code Scanner": "apple",
    "QR & Barcode Scanner (TeaCapps)": "android",
    "QR Code Reader: QR Scanner (Anha Ltd)": "android",
    "QR Scanner (PFA)": "android",
    "TeaCapps": "apple"
}


# In[ ]:


# Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')

# Cleanup
df = df.rename(columns={"Brand": "Fake"})
df.insert(1, 'Real', df['Fake'].map(fake_to_real))
df['Platform'] = df['Scanner ID'].map(scanner_to_platform)

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
    print(url, np.median(prob_fake))
    # prob_fake = prob_fake + 1e-10  # Add a small offset to avoid log(0)
    data.append(prob_fake)
    labels.append(url)

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels)
# plt.title('Probability of Fake Scans between URLs')
plt.xlabel('URL')
plt.ylabel('Probability Fake')
# plt.yscale('log')
plt.xticks(rotation=45)  # Rotate labels if they are long
plt.grid(True)

plt.savefig('visuals/prob_fake_scans_urls.png')
plt.show()


# In[ ]:


# Probability of URLs between Scanners

table = {}

grouped = df_comp.groupby(['Platform', 'Team Member', 'Scanner ID', 'Real'])

for (platform, member, scanner, url), group in grouped:
    if (platform, member, scanner) not in table:
        table[(platform, member, scanner)] = {}
    prob_fake = group['Trials'].map(lambda x: (1 - x).mean()).to_numpy().mean()
    table[(platform, member, scanner)][url] = prob_fake

# Function to format values (bold if 0.0000)
def format_value(value):
    if value < 0.05:
        return f"\\color{{red}}{value:.4f}"
    elif value > 0.5:
        return f"\\color{{blue}}{value:.4f}"
    else:
        return f"{value:.4f}"

# Print table header
s = ""
s += "\\begin{table*}\n"
s += "\\begin{adjustbox}{width=\\textwidth}\n"
s += "\\begin{tabular}{lccccccccccc}\n"
s += "\\toprule\n"
s += "\\textbf{Scanner} & \\textbf{amazon} & \\textbf{chatgpt} & \\textbf{facebook} & \\textbf{google} & \\textbf{instagram} & \\textbf{reddit} & \\textbf{whatsapp} & \\textbf{wikipedia} & \\textbf{yahoo} & \\textbf{youtube} \\\\\n"
s += "\\midrule\n"

# Print table rows
for (idx, ((platform, _, scanner), values)) in enumerate(table.items()):
    print(scanner, np.mean(np.array(list(values.values()))))
    match platform:
        case "apple":
            prefix = "\\faApple"
        case "android":
            prefix = "\\faAndroid"

    row = [prefix + "~" + scanner.replace('&', '\&')]
    for key in ['amazon', 'chatgpt', 'facebook', 'google', 'instagram', 'reddit', 'whatsapp', 'wikipedia', 'yahoo', 'youtube']:
        row.append(format_value(values[key]))
    s += " & ".join(row) + " \\\\\n"
    if idx % 2 == 1 and idx != 7:
        s += "\\midrule\n"

print()

# Print table footer
s += "\\bottomrule\n"
s += "\\end{tabular}\n"
s += "\\end{adjustbox}\n"
s += "\\caption{Mean probabilities of fake scans of apps between URLs. Each 2 scanner section represents the collective scans of 1 person. Red values are less than 0.05, and blue values are greater than 0.5. Android Camera had the lowest fake scan rate at 0.03, while Apple Camera had the highest fake scan rate 0.44.}\n"
s += "\\label{tab:mean_prob_fake_scans_url}\n"
s += "\\end{table*}"

print(s)


# In[ ]:


# Table of Mean and CI for Luminosity and Brightness Levels

table = {}

grouped = df_comp.groupby('Luminosity')

for lum, group in grouped:
    if lum not in table:
        table[lum] = {}
    subgrouped = group.groupby('Module Size')
    for prob_bytes, group in subgrouped:
        prob_fake = group['Trials'].map(lambda x: (1 - x).mean()).to_numpy()

        # Parameters
        confidence_level = 0.95  # 95% confidence interval
        n = len(prob_fake)            # Sample size
        mean = np.mean(prob_fake)     # Sample mean
        std_dev = np.std(prob_fake, ddof=1)  # Sample standard deviation (ddof=1 for sample std)

        # Calculate standard error
        standard_error = std_dev / np.sqrt(n)

        # Determine critical value (t-distribution for small samples, z-distribution for large samples)
        critical_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)

        # Calculate margin of error
        margin_of_error = critical_value * standard_error

        # Calculate confidence interval
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)

        table[lum][prob_bytes] = (mean, confidence_interval)

# Function to format values (bold if 0.0000)
def format_value(value):
    return f"{value:.4f}"

# Print table header
s = ""
s += "\\begin{table*}\n"
s += "\\begin{adjustbox}{width=\\textwidth}\n"
s += "\\begin{tabular}{lccc}\n"
s += "\\toprule\n"
s += "& \\multicolumn{3}{c}{\\textbf{Probabilistic Bytes}} \\\\\n"
s += "\\cmidrule(lr){2-4}\n"
s += "\\textbf{Luminosity} & \\textbf{1} & \\textbf{2} & \\textbf{3} \\\\\n"
s += "\\midrule\n"

# Print table rows
for lum, rest in table.items():
    match lum:
        case 0:
            lum_word = "Bright"
        case 1:
            lum_word = "Medium"
        case 2:
            lum_word = "Dark"

    row = [lum_word]
    for prob_bytes, value in rest.items():
        row.append(f'{format_value(value[0])}~({format_value(value[1][0])}, {format_value(value[1][1])})')

    s += " & ".join(row) + " \\\\\n"

# Print table footer
s += "\\bottomrule\n"
s += "\\end{tabular}\n"
s += "\\end{adjustbox}\n"
s += "\\caption{Mean and 95\\% CI for probabilities of fake scans for luminosity and probabilistic bytes  using a student's \\(t\\)-distribution.}\n"
s += "\\label{tab:mean_ci_fake_scans_prob}\n"
s += "\\end{table*}"

print(s)


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

# X-axis tick labels
x_labels = ["Bright", "Medium", "Dark"]

# Plotting
plt.figure(figsize=(10, 6))

for label, values in data.items():
    plt.plot(x_values, values, label=label)

# Set x-axis ticks to only show 0, 1, 2
plt.xticks(x_values, x_labels)

# Add labels, title, and legend
plt.xlabel("Luminosity")
plt.ylabel("Probability Fake")
# plt.title("Probability of Fake Scans at Different Luminosities")
plt.legend()

# Show the plot
plt.savefig('visuals/prob_fake_scans_lum.png')
plt.show()


# In[ ]:


# Mann-Whitney U rank tests between luminosities and prob byte count

grouped = df_comp.groupby('Luminosity')

print("Comparing overall luminosity distributions")
for lum_0, group_0 in grouped:
    for lum_1, group_1 in grouped:
        if lum_0 < lum_1:
            print(f'luminosity {lum_0} vs luminosity {lum_1}')
            prob_fake_0 = group_0['Trials'].map(lambda x: (1 - x).mean()).to_numpy()
            prob_fake_1 = group_1['Trials'].map(lambda x: (1 - x).mean()).to_numpy()
            print(stats.mannwhitneyu(prob_fake_0, prob_fake_1))

print()
grouped = df_comp.groupby('Module Size')

print("Comparing overall prob byte count distributions")
for prob_byte_count_0, group_0 in grouped:
    for prob_byte_count_1, group_1 in grouped:
        if prob_byte_count_0 < prob_byte_count_1:
            print(f'{prob_byte_count_0} prob bytes vs {prob_byte_count_1} prob bytes')
            prob_fake_0 = group_0['Trials'].map(lambda x: (1 - x).mean()).to_numpy()
            prob_fake_1 = group_1['Trials'].map(lambda x: (1 - x).mean()).to_numpy()
            print(stats.mannwhitneyu(prob_fake_0, prob_fake_1))


# In[ ]:




