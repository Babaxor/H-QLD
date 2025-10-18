import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

# --- 1. Simulation Setup ---
np.random.seed(42)
n_items = 30
n_takers = 500

# Create item profiles with a mix of types
item_profiles = []
for i in range(n_items):
    if i < 10:  # Lexically challenging
        l2, l1, l3 = np.random.uniform(0.7, 0.95), np.random.uniform(0.1, 0.4), np.random.uniform(0.2, 0.6)
    elif i < 20: # Logically challenging
        l2, l1, l3 = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4), np.random.uniform(0.7, 0.95)
    else: # Balanced
        l2, l1, l3 = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
    item_profiles.append([l2, l1, l3])
item_profiles = np.array(item_profiles)

# --- 2. Simulate Response Data ---
# True abilities
true_abilities = np.random.normal(0, 1, n_takers)

# H-QLD parameters
w2, w1, w3 = 1, 1, 1
tau2, tau1 = 0.5, 0.4

# Calculate H-QLD difficulty scores
hqld_scores = []
for l2, l1, l3 in item_profiles:
    if l2 > tau2:
        score = w2 * l2
    elif l1 > tau1:
        score = w2 * l2 + w1 * l1
    else:
        score = w2 * l2 + w1 * l1 + w3 * l3
    hqld_scores.append(score)

# Generate responses based on H-QLD model
response_matrix = np.zeros((n_takers, n_items))
for j in range(n_takers):
    for i in range(n_items):
        prob = 1 / (1 + np.exp(-(true_abilities[j] - hqld_scores[i])))
        response_matrix[j, i] = np.random.binomial(1, prob)

# --- 3. Estimate Difficulties from Both Models ---
# Estimate H-QLD difficulties (we already have them, they are the "true" scores in this sim)
hqld_difficulties = np.array(hqld_scores)

# Estimate Rasch difficulties using joint maximum likelihood (simplified)
rasch_difficulties = np.zeros(n_items)
for i in range(n_items):
    model = LogisticRegression()
    model.fit(true_abilities.reshape(-1, 1), response_matrix[:, i])
    rasch_difficulties[i] = -model.intercept_[0]

# --- 4. Create the Scatter Plot ---
plt.style.use('seaborn-whitegrid') 
fig, ax = plt.subplots(figsize=(8, 7))

# Color points based on their primary complexity
colors = []
for l2, l1, l3 in item_profiles:
    if l2 > 0.7: colors.append('red') # Lexically hard
    elif l3 > 0.7: colors.append('blue') # Logically hard
    else: colors.append('green') # Balanced

scatter = ax.scatter(rasch_difficulties, hqld_difficulties, c=colors, alpha=0.7, edgecolors='k')

# Add a reference line (y=x)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)

ax.set_xlabel('Estimated Rasch Difficulty ($\\delta_i$)', fontsize=12)
ax.set_ylabel('H-QLD Difficulty Score ($S_{ij}$)', fontsize=12)
ax.set_title('Comparison of Item Difficulty Estimates: Rasch vs. H-QLD', fontsize=14)

# Create a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Lexically Challenging', markerfacecolor='r', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Logically Challenging', markerfacecolor='b', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Balanced', markerfacecolor='g', markersize=10),
    plt.Line2D([0], [0], color='k', linestyle='--', label='y = x (Perfect Agreement)')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()

# Save the figure to a file
plt.savefig('rasch_vs_hqld_comparison.png', dpi=300)
print("Plot saved as rasch_vs_hqld_comparison.png")

# Display the figure in the output
plt.show()

# --- Optional: Show the file path ---
cwd = os.getcwd()
image_path = os.path.join(cwd, 'rasch_vs_hqld_comparison.png')
print(f"Image also saved at: {image_path}")

# Also save the data for the appendix
pd.DataFrame({
    'Item_ID': range(1, n_items + 1),
    'Rasch_Difficulty': rasch_difficulties,
    'H QLD_Difficulty': hqld_difficulties,
    'Lexical_Complexity': item_profiles[:, 0],
    'Syntactic_Complexity': item_profiles[:, 1],
    'Logical_Complexity': item_profiles[:, 2]
}).to_csv('simulation_data.csv', index=False)
print("Data saved as simulation_data.csv")