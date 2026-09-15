# %% [markdown]
# # Week 1 — Load the data & prove the bias exists
#
# In VS Code, each `# %%` block is a runnable cell (a "Run Cell" link appears above it).
# Work top to bottom. Goal for this week: load the Last.fm data, confirm it looks right,
# and draw the long-tail plot (your Figure 1).

# %%
# --- Cell 1: imports ---
# First, in the VS Code terminal: pip install -r requirements.txt
import sys
sys.path.append("..")          # lets us import the code in ../src
from src import data_loading, popularity

print("imports OK")

# %%
# --- Cell 2: load the dataset (first run downloads ~176 MB, then it's cached) ---
artists, users, plays = data_loading.load_raw()
print("plays shape:", plays.shape)
print("num artists:", len(artists))
print("num users:  ", len(users))
# Sanity check: expect roughly 292,000 artists and 359,000 users.
# (plays is artists x users — we'll flip it next.)
import numpy as np
user_items = data_loading.to_user_items(plays)
listeners = popularity.artist_popularity(user_items)
top10 = np.argsort(listeners)[::-1][:10]
for i in top10:
    print(artists[i], "-", int(listeners[i]), "listeners")
# %%
# --- Cell 3: peek at real data so it feels concrete ---
# TODO: print a few artist names, e.g. artists[:10], to see what's inside.
print(artists[:50])
# %%
# --- Cell 4: YOUR FIRST DELIVERABLE — the long-tail plot ---
# Steps (fill these in using the stubs in src/):
#   1. user_items = data_loading.to_user_items(plays)
#   2. pop        = popularity.artist_popularity(user_items)   # listeners per artist
#   3. popularity.long_tail_plot(pop)                          # saves results/fig1_longtail.png
#
# When it works you should see a steep spike on the left (a few megastars) and a long
# flat tail on the right (thousands of barely-heard artists). That shape IS the problem.
#
# TODO: implement the three stub functions above, then call them here.
user_items = data_loading.to_user_items(plays)
listeners = popularity.artist_popularity(user_items)
popularity.long_tail_plot(listeners, save_path="../results/fig1_longtail.png")
# %%
