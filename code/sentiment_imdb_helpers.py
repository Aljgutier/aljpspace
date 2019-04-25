re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    BOS = 'xbos'  # beginning-of-sentence tag
    FLD = 'xfld'  # data field tag
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)  # BOS beginning of text for a doc
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)  # multiple fields
    texts = list(texts.apply(fixup).values)
      # tokenize with process all multiprocessor ... tokenize slow, but speed up with multiple cores
      # SpaCy gret, but slow and with multi processor its much better
      # number of sublists is number of cores on your computer ... each part of the list will be tokenized on different core
      #   on Jeremy's machine 1.5 hours without mulitprocessing, a couple of minutes with multiprocessing
      #   e.g. we all have multicores on our laptops
    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
       # go through each chunck (each is a dataframe) and call get_texts
       #    get_texts will grab labels make them into ints and grb texts 
       #    before including the text get_text includes BOS function.
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels


def make_md(trn_clas, trn_labels, val_clas, val_labels, bs):
    min_lbl = trn_labels.min()
    trn_labels -= min_lbl
    val_labels -= min_lbl
    trn_ds = TextDataset(trn_clas, trn_labels)
    val_ds = TextDataset(val_clas, val_labels)
    trn_samp = SortishSampler(trn_clas, key=lambda x: len(trn_clas[x]), bs=bs//2)
    val_samp = SortSampler(val_clas, key=lambda x: len(val_clas[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(PATH, trn_dl, val_dl)
    return md

def get_rnn_clf(bptt, vs, em_sz, nh, nl):
    
    c=int(trn_labels.max())+1
    # dps = np.array([0.4, 0.5, 0.05, 0.3, 0.1])
    # change 20*70 to 10*70 ... running out of memory with 20 * 70 ... see notes/comments below
    dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5

    m = get_rnn_classifer(bptt, 10*70, c, vs, emb_sz=em_sz, n_hid=nh, 
                      n_layers=nl, pad_token=1,
                      layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
                      dropouti=dps[0], wdrop=dps[1],        
                      dropoute=dps[2], dropouth=dps[3])
    #opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
    # https://forums.fast.ai/t/lesson10-classification-part-running-out-of-memory/14667
    #Paperspace, p40000
    #RAM: 30 GB
    #CPUS: 8
    #HD: 210.5 KB / 100 GB
    #GPU: 8 GB
    # Note from JH
    #    When creating the classifier there’s a param that’s set to 20*70 in the notebook. 
    #    Change it to 10*70 to halve (approx) the memory requirements.
    #    max sl param ... this is the max sequence length ... see lm_rnn.py ... get_rnn_classifier
    return m

