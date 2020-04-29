import os
import pandas as pd
from data import pre_process, toxic_labels

DATA_PATH = 'data/nlp/toxic-comments'


def concat_test_dfs(text_path: str, labels_path: str) -> pd.DataFrame:
    def ignored(c: pd.Series):
        return all(c[col] == -1 for col in toxic_labels)


    ignored_mask = test_label_df.apply(ignored, axis=1)
    keep_mask = np.logical_not(ignored_mask.values)
    test_df = pd.concat((test_text_df, test_label_df), axis=1).iloc[keep_mask]


train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
pre_process(df).to_csv(os.path.join(DATA_PATH, f'processed_{file}.csv'))

clean('train')
clean('test')
