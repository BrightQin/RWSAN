import numpy as np
import glob
import torch.utils.data as Data
import random, re, json
import torch
from torchnlp.word_to_vector import GloVe
from data.ans_punct import prep_ans


def shuffle_list(ans_list):
    random.shuffle(ans_list)


# ------------------------------
# ---- Initialization Utils ----
# ------------------------------

def img_feat_path_load(path_list):
    iid_to_path = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        iid_to_path[iid] = path

    return iid_to_path


def img_feat_load(path_list):
    iid_to_feat = {}

    for ix, path in enumerate(path_list):
        iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
        img_feat = np.load(path)
        img_feat_x = img_feat['x'].transpose((1, 0))
        iid_to_feat[iid] = img_feat_x
        print('\rPre-Loading: [{} | {}] '.format(ix, path_list.__len__()), end='          ')

    return iid_to_feat


def ques_load(ques_list):
    qid_to_ques = {}

    for ques in ques_list:
        qid = str(ques['question_id'])
        qid_to_ques[qid] = ques

    return qid_to_ques


def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    glove_embedding = GloVe(name='840B', dim=300)
    if use_glove:
        pretrained_emb=(glove_embedding['PAD'])
        pretrained_emb=pretrained_emb.unsqueeze(0)
        pretrained_emb=torch.cat((pretrained_emb, glove_embedding['UNK'].unsqueeze(0)), 0)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb = torch.cat((pretrained_emb, glove_embedding[word].unsqueeze(0)), 0)

    return token_to_ix, pretrained_emb


def ans_stat(json_file):
    ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

    return ans_to_ix, ix_to_ans


# ------------------------------------
# ---- Real-Time Processing Utils ----
# ------------------------------------

def proc_img_feat(img_feat, img_feat_pad_size):
    if img_feat.shape[0] > img_feat_pad_size:
        img_feat = img_feat[:img_feat_pad_size]

    img_feat = np.pad(
        img_feat,
        ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return img_feat


def proc_ques(ques, token_to_ix, max_token):
    ques_ix = np.zeros(max_token, np.int64)

    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == max_token:
            break

    return ques_ix


def get_score(occur):
    if occur == 0:
        return .0
    elif occur == 1:
        return .3
    elif occur == 2:
        return .6
    elif occur == 3:
        return .9
    else:
        return 1.


def proc_ans(ans, ans_to_ix):
    ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
    ans_prob_dict = {}

    for ans_ in ans['answers']:
        ans_proc = prep_ans(ans_['answer'])
        if ans_proc not in ans_prob_dict:
            ans_prob_dict[ans_proc] = 1
        else:
            ans_prob_dict[ans_proc] += 1

    for ans_ in ans_prob_dict:
        if ans_ in ans_to_ix:
            ans_score[ans_to_ix[ans_]] = get_score(ans_prob_dict[ans_])

    return ans_score

class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C


        # --------------------------
        # ---- Raw data loading ----
        # --------------------------
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train', 'val', 'test']:
                self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.QUESTION_PATH[split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = ans_stat('data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')


    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            ques = self.ques_list[idx]

            # Process image feature from (.npz) file
            img_feat = np.load(self.iid_to_img_feat_path[str(ques['image_id'])])
            img_feat_x = img_feat['x'].transpose((1, 0))
            img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)


        return torch.from_numpy(img_feat_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)


    def __len__(self):
        return self.data_size
