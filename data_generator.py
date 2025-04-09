
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import re

from config import config
import data_helpers

print("config types:", config['np_int'], config['np_flaot'], tf.int32, tf.float32)

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
def createSimpleTokenizer(corpus_files, vocab_size, unknown_token, special_tokens):
        tokenizer = Tokenizer(models.WordLevel(unk_token=unknown_token))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordLevelTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train(corpus_files, trainer=trainer)
        tokenizer.decoder = decoders.WordPiece(prefix=" ")
        return tokenizer

def creat_team_tokenizer(df):
    teams = list(set(list(df['teams_home_team_id']) + list(df['teams_away_team_id'])))
    teams_string = [str(team) for team in teams]
    teams_string = [re.sub(r"\s", "_", item) for item in teams_string]    # replace spaces with a '_'
    teams_text = " ".join(teams_string)

    corpus_file = "./data/tokenizers/team_ids_text.txt"
    f = open(corpus_file, "w", encoding="utf-8")
    f.write(teams_text)
    f.close()

    corpus_files = [corpus_file]
    unknown_token = config['unknown_token']
    special_tokens = [unknown_token] + ["[HOME]", "[AWAY]"]
    vocab_size = len(teams_string) + len(special_tokens)

    tokenizer_team = createSimpleTokenizer(corpus_files, vocab_size, unknown_token, special_tokens)
    return tokenizer_team

def create_country_tokenizer(df):
    countries_string = list(set(list(df['league_country']) + list(df['home_team_country'])+ list(df['away_team_country'])))
    countries_string = [re.sub(r"\s", "_", item) for item in countries_string]    # replace spaces with a '_'
    # teams_string = [str(team) for team in teams]
    countries_text = " ".join(countries_string)

    corpus_file = "./data/tokenizers/country_ids_text.txt"
    f = open(corpus_file, "w", encoding="utf-8")
    f.write(countries_text)
    f.close()

    corpus_files = [corpus_file]
    unknown_token = config['unknown_token']
    special_tokens = [unknown_token] + ["[LEAGUE_COUNTRY]", "[HOME_COUNTRY]", "[AWAY_COUNTRY]"]
    vocab_size = len(countries_string) + len(special_tokens)

    tokenizer_country = createSimpleTokenizer(corpus_files, vocab_size, unknown_token, special_tokens)
    return tokenizer_country

def generate_dataset(df, fixture_id_to_ids, tokenizer_team, tokenizer_country):

    BBAB_cols = [
        'teams_home_team_id', 'teams_away_team_id',                     # 0, 1
        'fixture_date',                                                 # 2,
        'league_country', 'home_team_country', 'away_team_country',     # 3, 4, 5
        'winning_percent_home', 'winning_percent_draws', 'winning_percent_away',  # 6, 7, 8
        'outcome'                                                       # 9
    ]
    len_normal_bbab = len(BBAB_cols)
   
    def normalize_raw_bbab(bbab):
        pair_str = [str(team) for team in bbab[0:2]]
        pair_text = " ".join(pair_str)
        pair_tokens = tokenizer_team.encode(pair_text).ids
        bbab[0:2] = pair_tokens
        bbab[2] = (bbab[2].to_pydatetime() - config['baseDate']).days
        for idx in (6, 7, 8):
                bbab[idx] = data_helpers.get_odds(bbab[idx]/100, config['bookie_profit_percent'])
        countries_str = [re.sub(r"\s", "_", item) for item in bbab[3:6]]
        countries_text = " ".join(countries_str)
        countries_tokens = tokenizer_country.encode(countries_text).ids
        bbab[3:6] = countries_tokens
        # print(bbab)  
        return bbab


    def get_base_and_sequence(df, baseId, ids):
        try:
            base_bbab = list(df.loc[df['fixture_id'] == baseId, BBAB_cols].iloc[0])
            base_bbab = normalize_raw_bbab(base_bbab)
            base_bbab = np.array(base_bbab, dtype=config['np_float'])

            sequence = np.array([[]] * len_normal_bbab).T  # shape: (0, len_normal_bbab)
            for id in ids:
                bbab = list(df.loc[df['fixture_id'] == id, BBAB_cols].iloc[0])
                bbab = normalize_raw_bbab(bbab)
                sequence = np.append(sequence, [bbab], axis=0)

            return (baseId, base_bbab, sequence)
        except:
            raise Exception("Failed to get_BBAB for baseId = {}".format(baseId))    
        

    def generator():    
        for baseId, ids in fixture_id_to_ids.items():
            print(baseId, ids, end='\r')
            base_and_sequence = get_base_and_sequence(df, int(baseId), ids)
            # print(base_and_sequence)

            yield base_and_sequence

    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.int32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(()), tf.TensorShape((len_normal_bbab,)), tf.TensorShape((None, len_normal_bbab))),
        args=()
    )

    return ds

def split_and_normalize_bbas_tensor(bbas_batch):
    home, away, days, league_country, home_country, away_country, odds, outcome \
    = tf.split(bbas_batch, num_or_size_splits=[1, 1, 1, 1, 1, 1, 3, 1], axis=-1)

    # home = tf.squeeze(tf.cast(home, dtype=tf.int32))
    # away = tf.squeeze(tf.cast(away, dtype=tf.int32))
    # days = tf.squeeze(tf.cast(days, dtype=tf.int32))
    # league_country = tf.squeeze(tf.cast(league_country, dtype=tf.int32))
    # home_country = tf.squeeze(tf.cast(home_country, dtype=tf.int32))
    # away_country = tf.squeeze(tf.cast(away_country, dtype=tf.float32))
    # odds_home_win = tf.squeeze(odds_home_win)
    # odds_draw = tf.squeeze(odds_draw)
    # odds_away_win = tf.squeeze(odds_away_win)
    # outcome = tf.squeeze(tf.cast(outcome, dtype=tf.int32))
    # outcome = tf.one_hot(outcome, 3)

    home = tf.cast(home, dtype=tf.int32)
    away = tf.cast(away, dtype=tf.int32)
    days = tf.cast(days, dtype=tf.int32)
    league_country = tf.cast(league_country, dtype=tf.int32)
    home_country = tf.cast(home_country, dtype=tf.int32)
    away_country = tf.cast(away_country, dtype=tf.float32)
    outcome = tf.cast(outcome, dtype=tf.int32)
    outcome = tf.squeeze(tf.one_hot(outcome, 3, axis=-1), axis=-2)
    # print('split', outcome.shape)

    # print(odds_home_win[0], odds_draw[0], odds_away_win[0])
    # print(outcome)

    return home, away, days, league_country, home_country, away_country, odds, outcome 
         

def get_dummy_bbas_tensor(tokenizer_team, tokenizer_country):
    dummy_wining_percent = (80, 10, 10)         
    dummy_odds = [
        data_helpers.get_odds(dummy_wining_percent[0]/100, config['bookie_profit_percent']),
        data_helpers.get_odds(dummy_wining_percent[1]/100, config['bookie_profit_percent']),
        data_helpers.get_odds(dummy_wining_percent[2]/100, config['bookie_profit_percent']),
    ]
    dummy_bbab = \
        tokenizer_team.encode('[HOME] [AWAY]').ids \
        + [0] \
        + tokenizer_country.encode('[LEAGUE_COUNTRY] [HOME_COUNTRY] [AWAY_COUNTRY]').ids \
        + dummy_odds \
        + [1]
    dummy_bbab = tf.constant(dummy_bbab, dtype=tf.float32)
    return dummy_bbab

def get_dummy_bbas_tensor_2(tokenizer_team, tokenizer_country, base_days):
    print('get_dummy_bbas_2', base_days)  # expected: scalar int
    dummy_wining_percent = (80, 10, 10)         
    dummy_odds = [
        data_helpers.get_odds(dummy_wining_percent[0]/100, config['bookie_profit_percent']),
        data_helpers.get_odds(dummy_wining_percent[1]/100, config['bookie_profit_percent']),
        data_helpers.get_odds(dummy_wining_percent[2]/100, config['bookie_profit_percent']),
    ]
    dummy_bbab = \
        tokenizer_team.encode('[HOME] [AWAY]').ids \
        + [base_days] \
        + tokenizer_country.encode('[LEAGUE_COUNTRY] [HOME_COUNTRY] [AWAY_COUNTRY]').ids \
        + dummy_odds \
        + [1]
    dummy_bbab = tf.Variable(dummy_bbab, dtype=tf.float32)
    return dummy_bbab

def interprete_BB_AB(bbab, tokenizer_team, tokenizer_country):
    pair_string = tokenizer_team.decode(np.array(bbab[0:2], dtype=config['np_int']))
    days = bbab[2]
    date = config['baseDate'] + datetime.timedelta(days=int(days))
    country_string = tokenizer_country.decode(np.array(bbab[3:6], dtype=config['np_int']))
    odds = bbab[6:9]
    wp = tuple(data_helpers.get_probability(odds, config['bookie_profit_percent'])*100)
    winning_percent = (round(wp[0]), round(wp[1]), round(wp[2]))
    outcome = tuple(np.array(bbab[9:12], dtype=config['np_int']))
    # print("outcome", outcome)
    outcome = 0 if outcome == (1, 0, 0) else 1 if outcome == (0, 1, 0) else 2
    return pair_string, date, country_string, winning_percent, outcome

def interprete_dataset_item(item, tokenizer_team, tokenizer_country, id_to_ids, df):
    baseId = item[0].numpy()
    baseOutput = item[1].numpy()
    print(baseOutput)
    sequence = item[2].numpy()
    print(sequence)

    print("baseId: ", baseId)
    print("baseOutput: ", interprete_BB_AB(baseOutput, tokenizer_team, tokenizer_country))

    ids = id_to_ids[str(baseId)]
    for i in range(sequence.shape[0]):
        id = ids[i] if len(ids) > 0 else 0
        print("sequence {}, {}".format(i, id), interprete_BB_AB(sequence[i], tokenizer_team, tokenizer_country))


def creat_team_tokenizer_uk(df):
    teams = list(set(list(df['HomeTeam']) + list(df['AwayTeam'])))
    teams_string = [str(team) for team in teams]
    teams_string = [re.sub(r"\s", "_", item) for item in teams_string]    # replace spaces with a '_'
    teams_text = " ".join(teams_string)

    corpus_file = "./data/tokenizers/team_ids_text_uk.txt"  # Just make sure this file exists. Overwritten.
    f = open(corpus_file, "w", encoding="utf-8")
    f.write(teams_text)
    f.close()

    corpus_files = [corpus_file]
    unknown_token = config['unknown_token']
    special_tokens = [unknown_token] ################### + ["[HOME]", "[AWAY]"]
    vocab_size = len(teams_string) + len(special_tokens)

    tokenizer_team = createSimpleTokenizer(corpus_files, vocab_size, unknown_token, special_tokens)
    return tokenizer_team

#====================================== Keep the order ============================================
id_cols = ['id']
Div_cols = ['Div']
Date_cols = ['Date']
Team_cols = ['HomeTeam', 'AwayTeam']
# Odds_cols = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'WHH', 'WHD', 'WHA']
# Odds_cols = ['B365H', 'B365D', 'B365A', 'WHH', 'WHD', 'WHA']
Odds_cols = ['HDA0H', 'HDA0D', 'HDA0A', 'HDA1H', 'HDA1D', 'HDA1A', 'HDA2H', 'HDA2D', 'HDA2A', 'HDA3H', 'HDA3D', 'HDA3A']    # for x versions.
BB_cols = id_cols + Div_cols + Date_cols + Team_cols + Odds_cols

Half_Goal_cols = ['HTHG', 'HTAG']
Full_Goal_cols = ['FTHG', 'FTAG']
Goal_cols = Half_Goal_cols + Full_Goal_cols
Result_cols = ['HTR', 'FTR']    # A function of Goal_cols, but contribute to better representation.
Shoot_cols = ['HS', 'AS']
ShootT_cols = ['HST', 'AST']
Corner_cols = ['HC', 'AC']
Faul_cols = ['HF', 'AF']
Yellow_cols = ['HY', 'AY']    # H/A Yellow Cards, H/A Red Cards
Red_cols = ['HR', 'AR']    # H/A Yellow Cards, H/A Red Cards
AB_cols = Goal_cols + Result_cols + Shoot_cols + ShootT_cols + Corner_cols + Faul_cols + Yellow_cols + Red_cols

# underscore_prefixed lists have discontinued columns.
BBAB_cols = BB_cols + AB_cols
_Cols_List_to_Embedd = [Div_cols, Team_cols, Goal_cols, Result_cols]
_Cols_List_to_Standardize = [Odds_cols, Shoot_cols, ShootT_cols, Corner_cols, Faul_cols, Yellow_cols, Red_cols]
_Cols_List_for_Label = [Full_Goal_cols, Odds_cols]
_Label_cols = Full_Goal_cols + Odds_cols

def get_std_size():
    size = 0
    for cols in _Cols_List_to_Standardize:
        size += len(cols)
    return size

#---------------------------------------------------------------------------------------------------------------

def get_standardization_params(df):
    
    def get_mean_and_std(cols):
        array = df[cols]
        array = np.array(array)
        return (array.mean(), array.std(), np.max(array))
    
    params = {}
    for cols in _Cols_List_to_Standardize:
        params[cols[0]] = get_mean_and_std(cols)

    return params

def normalize_raw_bbab(bbab, tokenizer_team, std_params):
    #---------------------- label, before changing bbab
    label = []
    for cols in _Cols_List_for_Label:
        start = BBAB_cols.index(cols[0]); end = BBAB_cols.index(cols[0]) + len(cols)
        label += bbab[start : end]

    #----------------------- columns to embed: Div, HomeTeam, AwayTeam, HTHG, HTAG, FTHG, FTAG, HTR, FTR
    start = BBAB_cols.index(Div_cols[0])
    Div = bbab[start]
    bbab[start] = 1 if Div == 'E0' else 2 if Div == 'E1' else 3 if Div == 'E2' else 4 if Div == 'E3' else 0 # 0 for Unknown

    start = BBAB_cols.index(Team_cols[0]); end = start + len(Team_cols)
    pair_str = [str(team) for team in bbab[start : end]]    # Team names are already normalized, removing/striping spaces.
    pair_text = " ".join(pair_str)
    pair_tokens = tokenizer_team.encode(pair_text).ids
    bbab[start : end] = pair_tokens # 0 for Unknown, by tokenizer trainig.

    start = BBAB_cols.index(Goal_cols[0]); end = start + len(Goal_cols)
    bbab[start: end] = bbab[start : end]   # Goals themselves are good tokens, assuming there is no Unknown. mask_zero = False
    
    start = BBAB_cols.index(Result_cols[0]); end = start + len(Result_cols)
    bbab[start : end] = [1 if result == 'H' else 2 if result == 'D' else 3 if result == 'A' else 0 for result in bbab[start : end]]

    #--------------------- standardize
    for cols in _Cols_List_to_Standardize:
        start = BBAB_cols.index(cols[0]); end = start + len(cols)
        (mean, std, maximum) = std_params[cols[0]]
        assert 0 <= min(bbab[start : end])
        bbab[start : end] = [ (item - mean) / std for item in bbab[start : end] ]
        assert - mean/std <= min(bbab[start : end])
        assert max(bbab[start : end]) <= (maximum - mean) / std
        # print('std', bbab[start : end])

    #--------------------- columns for positional embedding
    start = BBAB_cols.index(Date_cols[0])
    bbab[start] = (datetime.datetime.combine(bbab[start], datetime.time(0,0,0)) - config['baseDate']).days

    #---------------------- bb only
    start = BBAB_cols.index(BB_cols[0]); end = start + len(BB_cols)
    bb = bbab[start : end]

    #----------------------- return
    return bbab, bb, label

def check_standardization(bbab, std_params):
    for cols in _Cols_List_to_Standardize:
        start = BBAB_cols.index(cols[0]); end = start + len(cols)
        (mean, std, maximum) = std_params[cols[0]]
        if -mean/std > min(bbab[start : end]) + 1e-5:
            print('standardization error 1', cols[0], -mean/std, bbab[start : end])
        if max(bbab[start : end]) > (maximum - mean) / std + 1e-5:
            print('standardization error 2', cols[0], bbab[start : end], (maximum - mean) / std)

def get_data_record(df, baseId, ids, tokenizer_team, std_params):
    # try:
        # base_bbab = list(df.loc[df['id'] == baseId, BBAB_cols])
        base_bbab = list(df[df['id'] == baseId][BBAB_cols].iloc[0, :])
        base_bbab, base_bb, base_label = normalize_raw_bbab(base_bbab, tokenizer_team, std_params)
        # print('2', base_bbab)
        base_bbab = tf.Variable(base_bbab, dtype=tf.float32)
        base_bb = tf.Variable(base_bb, dtype=tf.float32)
        base_label = tf.Variable(base_label, dtype=tf.float32)
        # print('3', base_bbab)
        sequence = tf.transpose(tf.Variable([[]] * len(BBAB_cols), dtype=tf.float32))   # (0, len(BBAB_cols))
        # sequence = np.array([[]] * len(BBAB_cols), dtype=config['np_float']).T
        # print('3.5', sequence)
        concat = []
        for id in ids:
            bbab = list(df[df['id'] == id][BBAB_cols].iloc[0, :])
            # print('4', bbab)
            bbab, _, _ = normalize_raw_bbab(bbab, tokenizer_team, std_params)
            # check_standardization(bbab, std_params)

            bbab = tf.Variable(bbab, dtype=tf.float32)[tf.newaxis, :]       # (1, len(BBAB_cols))
            # _bbab = bbab[0].numpy()
            # check_standardization(_bbab, std_params)

            concat.append(bbab)

        if len(concat) > 0:
            sequence = tf.concat(concat, axis=0)
            # if sequence.shape[0] > 0:
            #     bbab = sequence[0].numpy()
            #     check_standardization(bbab, std_params)

        # print('6', sequence)
        return (baseId, base_bbab, sequence, base_bb, base_label)
    # except:
    #     raise Exception("Failed to get_BBAB for baseId = {}".format(baseId))   

def generate_dataset_uk(df, fixture_id_to_ids, tokenizer_team, std_params):       
    def generator():
        count = 0
        for baseId, (tag, label, ids) in fixture_id_to_ids.items():
            baseId = int(baseId)
            # print('0', baseId, ids)
            (baseId, _, sequence, base_bb, base_label) = get_data_record(df, baseId, ids, tokenizer_team, std_params)
            print("count: {}, baseId: {}".format(count, baseId), end='\r'); count += 1
            
            # if count > 500: break

            #--------------------- verify
            # if sequence.shape[0] > 0:
            #     bbab = sequence[0, :].numpy()
            #     # print('11', bbab)
            #     for cols in _Cols_List_to_Standardize:
            #         start = BBAB_cols.index(cols[0]); end = start + len(cols)
            #         # print('.0', cols[0], start, end)
            #         (mean, std, maximum) = std_params[cols[0]]
            #         # print('.0.1', cols[0], mean, std, maximum)

            #         # if -mean/std > min(bbab[start : end]) + 1e-5:
            #         #     print('.1', cols[0], -mean/std, bbab[start : end])
            #         # if max(bbab[start : end]) > (maximum - mean) / std + 1e-5:
            #         #     print('.2', cols[0], bbab[start : end], (maximum - mean) / std)

            yield (baseId, sequence, base_bb, base_label)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.int32, tf.float32, tf.float32, tf.float32),
        output_shapes=(tf.TensorShape(()), tf.TensorShape((None, len(BBAB_cols))), tf.TensorShape((len(BB_cols),)), tf.TensorShape((len(_Label_cols),))),
        args=()
    )
    return ds

def get_dummy_bbas_tensor_uk(df, tokenizer_team, std_params):
    (baseId, base_bbab, sequence, base_bb, base_label) = get_data_record(df, min(list(df['id'])), [], tokenizer_team, std_params)
    dummy_bbab = tf.zeros_like(base_bbab, dtype=tf.float32) # All embedding fields will be zero, which means Unknown for all but goal fields.
    return dummy_bbab

