import os

dataDir = 'data'

positiveReviews = []
for filename in os.listdir(os.path.join(dataDir, 'train/pos')):
    with open(os.path.join(dataDir, 'train/pos', filename), 'r', encoding='utf-8') as f:
        positiveReviews.append(f.read())

print(positiveReviews[0])
