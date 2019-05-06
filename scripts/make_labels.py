import glob
import pandas as pd

NUM_CLASSES = 964
NUM_EXAMPLES = 20
NUM_TRAIN_EXAMPLES = 18

train_labels = []
val_labels = []
for i in range(1, NUM_CLASSES + 1):
    for j in range(1, NUM_EXAMPLES + 1):
        class_label = i - 1
        class_label_str = f'{class_label+1:04d}'
        character_id = f'{j:02d}'
        example_id = f'{class_label_str}_{character_id}'
        if j > NUM_TRAIN_EXAMPLES:
            val_labels.append([example_id, class_label])
        else:
            train_labels.append([example_id, class_label])

train_df = pd.DataFrame(train_labels, columns=['example_id', 'label'])
train_df = train_df.sample(frac=1)
val_df = pd.DataFrame(val_labels, columns=['example_id', 'label'])
val_df = val_df.sample(frac=1)

train_df.to_csv('data/train_labels.csv', index=False)
val_df.to_csv('data/val_labels.csv', index=False)

