import polars as pl
import numpy as np
import torch
from scipy import sparse, linalg
from scipy.sparse import csr_matrix
from datetime import timedelta

def recall_at(df_true, df_pred, k=40):
    recall =  (df_true[['node', 'cookie']].join(
               df_pred.sort('score', descending=True).group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
               how='left', on = ['cookie', 'node'])
               .select([pl.col('value').fill_null(0), 'cookie'])
               .group_by('cookie').agg([pl.col('value').sum() / pl.col('value').count()])['value'].mean())
    return recall

class Enc:
    def __init__ (self, user_key='user_id', item_key='item_id'):
        self.user_key = user_key
        self.item_key = item_key
        self.user_id_dict = {}
        self.item_id_dict = {}
        self.user_id_dict_inv = {}
        self.item_id_dict_inv = {}
        self.num_users = 0
        self.num_items = 0
    
    def _get_num(self, train_df):
        self.num_users = train_df[f'le_{self.user_key}'].max() + 1
        self.num_items = train_df[f'le_{self.item_key}'].max() + 1
    
    def get_num(self):
        return self.num_users, self.num_items
    
    def fit(self, train_df, event_weight=None):
        """ Converts DataFrame to sparse matrix """

        # inverted dicts
        self.user_id_dict = {x:i for i,x in enumerate(train_df[self.user_key ].unique().to_numpy())}
        self.item_id_dict = {x:i for i,x in enumerate(train_df[self.item_key].unique().to_numpy())}
        self.user_id_dict_inv = {v:k for k,v in self.user_id_dict.items()}
        self.item_id_dict_inv = {v:k for k,v in self.item_id_dict.items()}
        
        if event_weight=='event_weight':
            train_df = train_df.with_columns([
                pl.col(self.user_key ).replace_strict(self.user_id_dict).alias(f'le_{self.user_key}'),
                pl.col(self.item_key).replace_strict(self.item_id_dict).alias(f'le_{self.item_key}'),
            ]).select(f'le_{self.user_key}', f'le_{self.item_key}', 'event_weight')   
        else:
            train_df = train_df.with_columns([
                pl.col(self.user_key ).replace_strict(self.user_id_dict).alias(f'le_{self.user_key}'),
                pl.col(self.item_key).replace_strict(self.item_id_dict).alias(f'le_{self.item_key}'),
                pl.lit(1).alias('event_weight'),
            ]).select(f'le_{self.user_key}', f'le_{self.item_key}', 'event_weight')
        self._get_num(train_df)
        return train_df 
    
    def transform(self, train_df, event_weight=None):
        if event_weight=='event_weight':
            train_df = train_df.with_columns([
                pl.col(self.user_key ).replace(self.user_id_dict).alias(f'le_{self.user_key}'),
                pl.col(self.item_key).replace(self.item_id_dict).alias(f'le_{self.item_key}'),
                pl.lit(1).alias('event_weight'),
            ]).select(f'le_{self.user_key}', f'le_{self.item_key}', 'event_weight')
        else:
            train_df = train_df.with_columns([
                pl.col(self.user_key ).replace(self.user_id_dict).alias(f'le_{self.user_key}'),
                pl.col(self.item_key).replace(self.item_id_dict).alias(f'le_{self.item_key}'),
            ]).select(f'le_{self.user_key}', f'le_{self.item_key}', 'event_weight') 

        return train_df
    
    def inverse_transform(self, train_df):
        train_df = train_df.with_columns([
            pl.col(f'le_{self.user_key}').replace(self.user_id_dict_inv).alias(self.user_key ),
            pl.col(f'le_{self.item_key}').replace(self.item_id_dict_inv).alias(self.item_key),
            # pl.lit(1).alias('score'),
        ]).select(self.user_key, self.item_key, 'score')
        return train_df
    
def convert_to_sparse(train_df, enc):
    X = csr_matrix(
        (train_df['event_weight'].to_numpy(), 
        (train_df[f'le_{enc.user_key}'].to_numpy(), train_df[f'le_{enc.item_key}'].to_numpy())),
        shape=enc.get_num(),
        dtype=np.float32
    )
    return X


def process_in_batches(enc_eval_users, X, G, k, batch_size=100, fill_value=-1, item_col='node', mask_input = True):
    batch_dfs = []
    
    
    # есть чудики которых не смог проскорить
    enc_eval_users = [i for i in enc_eval_users if i is not None]
    for start in range(0, len(enc_eval_users), batch_size):
        end = start + batch_size
        batch_users = np.array(enc_eval_users[start:end])
        
        Xi = X[batch_users]
        predictions = Xi @ G
        if mask_input:
            predictions[Xi.nonzero()] = fill_value

        top_k_scores, top_k_ids = torch.topk(torch.tensor(predictions), k=k)
        top_k_ids, top_k_scores = top_k_ids.cpu().numpy(), top_k_scores.cpu().numpy()
        df_pred = pl.DataFrame({'le_cookie':batch_users,
                                f'le_{item_col}':top_k_ids,
                                'score':top_k_scores}).explode(['score',f'le_{item_col}'])
        batch_dfs.append(df_pred)
    
    return pl.concat(batch_dfs)


def get_dataset(recs, k_recs=200, max_pos=100, max_neg=30, seed=42, df_eval=None):
    """ 
    Sample data
    """
    if df_eval is not None:
        # only for learning binary classification
        recs_with_gt = recs.sort('score', descending=True).group_by('cookie').head(k_recs).join(
            df_eval[['node', 'cookie']].with_columns(target=1)[['node', 'cookie', 'target']],
            on=['node', 'cookie'],
            how='left',
        ).with_columns(pl.col('target').fill_null(0).alias('target'))

        recs_with_gt2 = recs_with_gt.group_by('cookie').agg(pl.sum('target'))
        active_users = (recs_with_gt2.filter(pl.col('target')>0)
                        )

        pos = recs_with_gt.filter(pl.col('target')==1).join(active_users.select('cookie').unique(), on='cookie')
        neg = recs_with_gt.filter(pl.col('target')==0).join(active_users.select('cookie').unique(), on='cookie')
        
        pos = pos.sample(fraction=1.0, shuffle=True, seed=seed).group_by('cookie').head(max_pos)
        neg = neg.sample(fraction=1.0, shuffle=True, seed=seed).group_by('cookie').head(max_neg)

        dataset = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed).sort('cookie', 'score')
    else:
        dataset = recs.sort('cookie', 'score')
    return dataset


# extra converters
def convert_to_sparse(train_df, enc):
    X = csr_matrix(
        (train_df['event_weight'].to_numpy(), 
        (train_df[f'le_{enc.user_key}'].to_numpy(), train_df[f'le_{enc.item_key}'].to_numpy())),
        shape=enc.get_num(),
        dtype=np.float32
    )
    return X

def convert(train_df, enc, col, muliplicator = 1, ):
    X = csr_matrix(
        (train_df[col].to_numpy() * muliplicator, 
        (train_df[f'le_{enc.user_key}'].to_numpy(), train_df[f'le_{enc.item_key}'].to_numpy())),
        shape=enc.get_num(),
        dtype=np.float32
    )
    return X

def truncate(G, k=20):
    if k:
        """ Мы так достанем топы i2i за один проход """
        col_indices = np.argpartition(G, -k, axis=0)[-k:]
        mask = np.zeros_like(G, dtype=bool)
        np.put_along_axis(mask, col_indices, True, axis=0)
        return np.where(mask, G, 0.0) 
    else:
        return G