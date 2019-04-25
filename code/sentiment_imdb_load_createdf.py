# Only run this the first time
# find it much easier to process text within Python, rather than helper functions
#   processing text with Python is straight forward, not that hard


CLASSES = ['neg', 'pos', 'unsup']

def get_rawtexts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        print( (path/label))
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_rawtexts(PATH/'train')
val_texts,val_labels = get_rawtexts(PATH/'test')

display(trn_labels)
len(trn_texts),len(val_texts)

# Only run this the first time
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))

trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]
trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]

print(len(trn_texts), len(trn_labels))


# Only run this the first time
# CLASSIFIER PATH
#
# Create Data Frames ... DONT RUN THIS UNLESS NEED TO RECREATE DataFrames and SAVE
# create dataframes and save to file
# Standardized format for NLP
# datarames used later

df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

display(df_trn.head())

# save data frame
# Classifier data ... no header by default
# after removing unsupervised (unlabeled classes = 2) there will be 25 K positive and 25 K negative
df_trn[df_trn['labels']!=2].to_csv(CLAS_PATH/'train.csv', header=False, index=False)
df_val.to_csv(CLAS_PATH/'test.csv', header=False, index=False)

(CLAS_PATH/'classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)
print((CLAS_PATH/'classes.txt').open().readlines())
