
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import pickle
import json
import re
import copy
import os
import networkx as nx

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from config import config
from dictionary_driver import Excel_Driver
from dictionary_manager import Dictionary
# import excel_driver

#============================================================================ basic ===============================================================

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def SaveJsonData(data, path):
        try:
                file = open(path, 'wt+')
                json.dump(data, file, cls = NumpyEncoder)
                file.close()
        except:
                raise Exception("Couldn't open/write/close file:" + path)


def LoadJsonData(path):
        data = None

        try:
                file = open(path, 'rt')
                data = json.load(file)
                file.close()
        except:
                #raise Exception("Couldn't open/read/close file:" + path)
                pass

        return data

def SaveBinaryData(data, path):
        try:
                file = open(path, 'wb+')
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
                file.close()
        except:
                raise Exception("Couldn't open/read/save/close file:" + path)

def LoadBinaryData(path):
        data = None
        try:
                file = open(path, 'rb')
                data = pickle.load(file)
                file.close()
        except:
                # raise Exception("Couldn't open/read/close file:" + path)
                pass

        return data

def SaveDataFrame_Excel(df, filePath):
        try:
                with pd.ExcelWriter(filePath) as writer:
                        df.to_excel(writer)
        except:
                raise Exception("Couldn't save to Excel file")
        return

def DeleteFile( path ):
        os.remove( path )

def read_excel(filename):
        # print(filename)
        # excelPath = os.path.join(countryFolder, filename + '.xlsx')

        # converters takes effect only after the excel file is read and, 
        # so, only after date-like string is converted to date type.
        df = pd.read_excel(filename, parse_dates=False) #, converters={'Date':str})
        return df

def read_csv(filename):
        # print(filename)
        # excelPath = os.path.join(countryFolder, filename + '.xlsx')
        df = pd.read_csv(filename, encoding='unicode_escape', sep=',')
        return df

def read_large_excel(data_foloder, filename):
        df = None
        binPath = os.path.join(data_foloder, filename + '.bin')
        bin = LoadBinaryData(binPath)
        if bin is not None:
            df = bin
        else:
            excelPath = os.path.join(data_foloder, filename + '.xlsx')
            df = pd.read_excel(excelPath)
            SaveBinaryData(df, binPath)
        return df

def get_normalization_params(df, cols):
    def get_mean_and_std(col):
        array = df[col].dropna()
        array = np.array(array)
        return (array.mean(), array.std(), np.max(array))
    params = {}
    for col in cols:
        params[col] = get_mean_and_std(col)
    return params

#========================================================== get_train_and_new_from_uk_football_data ===============================================

def find_1X2_odds_group(columns):
        oGroups = []
        i = 0
        while i <= len(columns)-3:
                if columns[i].endswith('H') and columns[i+1].endswith('D') and columns[i+2].endswith('A') \
                and columns[i][:-1] == columns[i+1][:-1] and columns[i+1][:-1] == columns[i+2][:-1]:
                        oGroups.append(columns[i][:-1]); i += 3
                else: i += 1
        return oGroups

def find_and_grow_with_extra_game_records_pythonic(df, df_new):
    successful = True; df_grown = None; df_extra = None

    try:
        org_ids = list(df['id'])
        df = df.drop(['id'], axis=1)
        df_new = df_new.drop(['id'], axis=1)

        # Check if there are duplicate rows. 
        # Do not drop but just check duplicate, because the org_id is there.
        assert (df.duplicated() == False).all()
        assert (df_new.duplicated() == False).all()

        # print('df', df)
        # print('df_new', df_new)

        # Check if df is a subset of df_new. 
        # In df_grown, all df rows will be placed before all df_new rows,
        # because a duplicates fall into the range of df_new.
        # We intentionally keep df to remain in df_grown, 
        # because we want to keep datasets expensively generated from df.
        df_grown = pd.concat([df, df_new]).drop_duplicates()
        assert len(df_grown) == len(df_new)
        # print('df_grown', df_grown)

        # Find extra rows. We know all extra rows are placed after all df rows.
        df_extra = df_grown.iloc[df.shape[0]:]
        # print('df_extra', df_extra)

        # Restore df ids for the df part and assign new ids to the df_extra part.
        ids = org_ids + list(range(max(list(org_ids))+1, max(list(org_ids))+1+df_extra.shape[0]))
        assert len(ids) == df_grown.shape[0]
        df_grown.insert(loc=0, column='id', value=ids)

        # df_extra can have any new ids. Preferably, assign them the same ids as in df_grown.
        ids = list(range(max(list(org_ids))+1, max(list(org_ids))+1+df_extra.shape[0]))
        assert len(ids) == df_extra.shape[0]
        df_extra.insert(loc=0, column='id', value=ids)

    except:
        print("find and grow raised an exception !!!")
        successful = False

        if successful:
                print("find and grow :", df_grown.shape[0], df_extra[0])
    return successful, df_grown, df_extra

def assign_seasonal_filenames(countryFolder):
    xlsxFiles = []
    for (dirpath, dirnames, files) in os.walk(countryFolder):     # Make sure the .csv files are renamed .xlsx.
        for filename in files:
            if '.xlsx' in filename and '~' not in filename and '-' not in filename and 'df_' not in filename:
                xlsxFiles.append((dirpath, filename))
    print('files found to rename: ', len(xlsxFiles))

    total_rows = 0
    for (_dirPath, xlsxFilename) in xlsxFiles:
            filePath = os.path.join(_dirPath, xlsxFilename)
            # df = read_csv(filePath) doesn't work with football-data.co.uk data.
            df = read_excel(filePath)
            org = df.shape[0]
            df = improve_uk_dataframe(df, dropNa=False)
            newFilePath = os.path.join(_dirPath, os.path.splitext(xlsxFilename)[0].split(' ')[0] + '-' + str(min(df['Date']).year) + '-' + str(max(df['Date']).year)+'.xlsx')
            # print(newFilePath)
            os.rename(filePath, newFilePath)
            total_rows += df.shape[0]
            del df
    print('total rows in renamed files: ', total_rows)
    return

def standardize_date_uk(rugged_date):
        if isinstance(rugged_date, datetime.datetime):  # If Regional Format == English(United Kingdom), all calls fall to this branch.
                date = rugged_date
        elif isinstance(rugged_date, str):
                print("Unexpected string date !!!!!!")
                try:
                        date = datetime.datetime.strptime(rugged_date, '%d/%m/%Y')
                except:
                        date = datetime.datetime.strptime(rugged_date, '%d/%m/%y')                                              
        else:
                print(rugged_date)
                raise Exception('unhandled date')
        
        if isinstance(date, datetime.datetime):
                date = date.date()

        return date


def standardize_dates_uk(rugged_dates):
        standardize_dates = [standardize_date_uk(d) for d in rugged_dates]
        return standardize_dates


def improve_uk_dataframe(df, dropNa=True):
        # print("checking in improve_uk_dataframe: rows ", df.shape[0])
        if dropNa:
                df.dropna(subset=df.columns, inplace=True)      # This is the ONLY line where we lose some or many rows.
        else:   # fill in missing odds
                ogs = find_1X2_odds_group(df.columns)
                og_cols = []
                for og in ogs: og_cols += [og+'H', og+'D', og+'A']
                print('1', ogs)
                
                # norm_params = get_normalization_params(df, og_cols)     # We wish the cols has few nan values.
                
                for index, row in df.iterrows():
                        nn_og = None
                        for og in ogs:
                                if not (pd.isnull(row[og+'H']) or pd.isnull(row[og+'D']) or pd.isnull(row[og+'A'])):
                                        # normal_h = (row[og+'H'] - norm_params[og+'H'][0]) / norm_params[og+'H'][1]
                                        # normal_d = (row[og+'D'] - norm_params[og+'D'][0]) / norm_params[og+'D'][1]
                                        # normal_a = (row[og+'A'] - norm_params[og+'A'][0]) / norm_params[og+'A'][1]
                                        nn_og = og; break
                                
                        if nn_og is None: continue      # give up this raw. it will be removed soon

                        for og in ogs:  # replace keeping std and mean.
                                if pd.isnull(row[og+'H']) or pd.isnull(row[og+'D']) or pd.isnull(row[og+'A']):
                                        df.loc[index, og+'H'] = row[nn_og+'H']; df.loc[index, og+'D'] = row[nn_og+'D']; df.loc[index, og+'A'] = row[nn_og+'A']
                                # col = og+'H'
                                # if pd.isnull(row[col]): (mean, std, max) = norm_params[col]; df.loc[index, col] = normal_h * std + mean
                                # col = og+'D'
                                # if pd.isnull(row[col]): (mean, std, max) = norm_params[col]; df.loc[index, col] = normal_d * std + mean
                                # col = og+'A'
                                # if pd.isnull(row[col]): (mean, std, max) = norm_params[col]; df.loc[index, col] = normal_a * std + mean

                df.dropna(subset=df.columns, inplace=True)      # This is the ONLY line where we lose some or many rows.                

        df['Date'] = standardize_dates_uk(list(df['Date']))
        # Didn't convert to a uniform case, because football-data.uk team names have no case errors,
        # although a few teams names have a space at the end.
        df['HomeTeam'] = [team.strip() for team in [re.sub(r"\s", "_", item) for item in list(df['HomeTeam'])]]
        df['AwayTeam'] = [team.strip() for team in [re.sub(r"\s", "_", item) for item in list(df['AwayTeam'])]]
        # print("checking out improve_uk_dataframe: rows ", df.shape[0])

        return df


def get_train_and_new_from_uk_football_data(gameHistoryFolderPath, countryThemeFolder, Non_Odds_cols, definitions, country_specs, skip=False):
        train_mode = definitions.TRAIN_MODE
        num_bookies = country_specs.NUMBER_BOOKIES_DATA
        oddsGroupsToExclude = country_specs.BOOKIE_TO_EXCLUDE
        preferedOrder = country_specs.PREFERED_ORDER

        binPath_grown = os.path.join(countryThemeFolder, "df_grown" + '.bin')
        excelPath_grown = os.path.join(countryThemeFolder, "df_grown" + '.xlsx')
        binPath_new = os.path.join(countryThemeFolder, "df_new" + '.bin')
        excelPath_new = os.path.join(countryThemeFolder, "df_new" + '.xlsx')
        binPath_white = os.path.join(countryThemeFolder, "df_white" + '.bin')
        excelPath_white = os.path.join(countryThemeFolder, "df_white" + '.xlsx')
        dictPath = os.path.join(countryThemeFolder, "df_grown" + '.json')

        if not train_mode or skip:      # Try to use the existing ones if in inference mode or requested to skip.
                df_grown = LoadBinaryData(binPath_grown)
                df_new = LoadBinaryData(binPath_new)
                if df_grown is not None and df_new is not None:
                        return df_grown, df_new

        #-------------- collect all .xlsx files in the target foloder.
        csvFiles = []
        for (dirpath, dirnames, files) in os.walk(gameHistoryFolderPath):     # Make sure the .csv files are renamed .xlsx.
                for filename in files:
                        # foloders (dirpath) that ends with '_' are skipped.
                        if not dirpath.endswith('_') and '.xlsx' in filename and '~' not in filename and 'df_' not in filename:
                                print(filename, end=' ')
                                csvFiles.append((dirpath, filename))
        print()
        csvFiles.sort()
        print(len(csvFiles), ' files found') 


        #-------------- read all excel files, and find odds groups, and the total of them, that have 1X2 odds.
        oGroups = []
        filename_dict = {}
        for (_dirPath, csvFilename) in csvFiles:
                filePath = os.path.join(_dirPath, csvFilename)
                # df = read_csv(filePath) doesn't work with football-data.co.uk data.
                df = read_excel(filePath)
                # print('-------------- time :',  df['Time'][0])
                columns = df.columns

                filesToSkip = []
                if set(Non_Odds_cols) <= set(columns): pass
                else:
                        print(filePath + " will be dropped out because it lacks some of the required columns !!!")
                        filesToSkip.append(csvFilename)
                # print(columns)
                ogs = find_1X2_odds_group(columns)
                # print(csvFilename, ogs)
                non_ogs = [col for col in columns if col[:-1] not in ogs]

                # drop  oddsGroupsToExclude        
                for og in oddsGroupsToExclude:
                        try:
                                df.drop(og+'H', axis=1)
                                df.drop(og+'D', axis=1)
                                df.drop(og+'A', axis=1)
                        except: pass

                filename_dict[csvFilename] = (df, non_ogs, ogs)
                # print(ogs)
                oGroups += ogs

        #-------------- Remove fils that lack some of required non-odds columns.
        for file in filesToSkip:
                filename_dict.pop(file, None)

        unique_oGroups = list(set(oGroups))
        # print(unique_oGroups)
        # print('unique oGroups: ', unique_oGroups)
        counts = [(oGroups.count(og), og) for og in unique_oGroups]
        counts.sort(reverse=True)
        # print('oGroup counts: ', counts)
        if len(counts) < num_bookies:   print("!!!!!! Too many bookies requested.")

        total_ogs_ordered = [og for (_, og) in counts]
        print('oGroups count-ordered ', total_ogs_ordered)

        # Execute preferedOrder, ignoring count order.
        indices = [i for i in range(len(total_ogs_ordered)) if total_ogs_ordered[i] in preferedOrder]
        for i in range(len(total_ogs_ordered)):
                if i in indices: total_ogs_ordered[i] = preferedOrder.pop(0)

        standard_ogs_ordered = total_ogs_ordered[:num_bookies]
        backup_ogs_ordered = total_ogs_ordered[num_bookies:]

        def get_1X2_rename_dict(local_ogs):
                # local_ogs_ordered = [og for og in total_ogs_ordered if og in local_ogs] #------- This will shift. We need a better solution.
                # assert len(local_ogs_ordered) >= num_bookies

                # if logal_ods misses any of standard_ogs_ordered, we borrow it.
                assert len(local_ogs) >= num_bookies
                local_backup_ordered = [og for og in backup_ogs_ordered if og in local_ogs]
                local_ogs_ordered = copy.deepcopy(standard_ogs_ordered)
                for id in range(len(local_ogs_ordered)):
                        if local_ogs_ordered[id] not in local_ogs:
                                local_ogs_ordered[id] = local_backup_ordered.pop(0)


                rename_plan = { local_ogs_ordered[i] : 'HDA' + str(i) for i in range(num_bookies) }
                rename_dict = {}
                for (source_og, target_og) in rename_plan.items():
                        rename_dict = rename_dict | \
                        { source_og + 'H' : target_og + 'H', source_og + 'D' : target_og + 'D', source_og + 'A' : target_og + 'A'}
                return rename_plan, rename_dict

        # Choose an initial common_cols
        common_cols = None
        for (filename, (df, non_ogs, ogs)) in filename_dict.items():
                rename_plan, rename_dict = get_1X2_rename_dict(ogs)
                df.rename(columns=rename_dict, inplace=True)
                common_cols = df.columns
                common_cols = [col for col in common_cols if col[:-1] not in oddsGroupsToExclude]
                break

        # Refine common_cols
        for (filename, (df, non_ogs, ogs)) in filename_dict.items():
                rename_plan, rename_dict = get_1X2_rename_dict(ogs)
                df.rename(columns=rename_dict, inplace=True)
                filename_dict[filename] = (df, non_ogs, ogs, rename_plan)        # keep rename_plan for later later maintainance use.
                cols = df.columns
                common_cols = [col for col in cols if col in common_cols]

        # print('common cols: ', common_cols)
        assert set(Non_Odds_cols) <= set(common_cols)

        df_built = None
        for (filename, (df, non_ogs, ogs, rename_plan)) in filename_dict.items():
                if df_built is None:
                        df_built = df[common_cols]
                else:
                        df_built = pd.concat([df_built, df[common_cols]])
                # print('df_built rows: ', df_built.shape[0], list(df_built.columns))
                # filename_dict[filename] = (non_ogs, ogs, rename_plan)
                filename_dict[filename] = (rename_plan)

        # Display rename vector, for how inconsistent the bookie rename is.
        for i in range(num_bookies):
                key = 'HDA' + str(i)
                renameVector = []
                for (filename, (rename_plan)) in filename_dict.items():
                        for (org_name, new_name) in rename_plan.items():
                                if key == new_name:
                                        renameVector.append(org_name)
                                        break
                print('Rename Vector: ', key, renameVector)
        
        # df_built.drop_duplicates()
        # Let this be the only place where df is sorted.
        # df_built = df_built.sort_values(['Date', 'Div', 'HomeTeam', 'AwayTeam'], ascending=[True, True, True, True])   # This is required for consistency between calls.
        df_built = df_built.sort_values(['Date', 'Div'], ascending=[True, True])   # This is required for consistency between calls.
        # No rows were dropped upto this point. Now permanent unique 'id's are assigned to the row.
        # A row's id assigned here will not change lifetime, UNLESS the same row was already assigned earlier.
        # 'id's will be used in place of ('Date', 'HomeTeam', 'AwayTeam'), for shorter storage space.
        baseId = config['baseGameId'] + 1   #--------------------------------------------------------------------
        ids = range(baseId, baseId+df_built.shape[0])
        df_built.insert(loc=0, column='id', value=ids)

        df_built = improve_uk_dataframe(df_built, dropNa=False)
        print('df_built cols: ', df_built.columns)

        df_grown_old = None
        df_grown_old = LoadBinaryData(binPath_grown)

        df_grown = df_new = None
        if df_grown_old is not None:
                pass # Make sure the rename_dictionaries are consistent.
                successful, df_grown, df_new = find_and_grow_with_extra_game_records_pythonic(df_grown_old, df_built)
                if successful:
                        print('grown, new: ', df_grown.shape, df_new.shape)
                        print("Successfully found new game records and grew with them", "New rows: ", df_new.shape[0])

        if df_grown_old is not None:
                if successful:
                        if train_mode:
                                # df_grown is the total and saved, df_new is the tail part of df_total and saved into the new file.
                                SaveBinaryData(df_grown, binPath_grown)
                                SaveDataFrame_Excel(df_grown, excelPath_grown)
                                SaveBinaryData(df_new, binPath_new)
                                SaveDataFrame_Excel(df_new, excelPath_new)
                                SaveJsonData(filename_dict, dictPath)
                        else:
                                # df_grown is the total and NOT saved, df_new is the tail part of df_total but saved into the WHITE file.
                                df_grown = df_grown_old         # 
                                SaveBinaryData(df_new, binPath_white)
                                SaveDataFrame_Excel(df_new, excelPath_white)  
                             
                else:
                        raise("!!! Failed to find and grow with extra game records")
        elif df_grown_old is None:      # successful is None
                if train_mode:
                        # df_grown is the total and saved, df_new is empty and saved into the new file.
                        df_grown = df_built
                        SaveBinaryData(df_grown, binPath_grown)         # df_built
                        SaveDataFrame_Excel(df_grown, excelPath_grown)  # df_built
                        SaveJsonData(filename_dict, dictPath)
                        # create an empty df_new
                        print("creating an empty df_new...")
                        df_new = df_grown.iloc[:0,:].copy()
                        SaveBinaryData(df_new, binPath_new)
                        SaveDataFrame_Excel(df_new, excelPath_new)
                else:
                        raise("!!! Inference mode requires df_grown to exist.")
                
        return df_grown, df_new


#============================================== creat_team_tokenizer_uk =========================================

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

def creat_team_tokenizer_uk(master_df, tokenizer_folder_path):
    teams = list(set(list(master_df['HomeTeam']) + list(master_df['AwayTeam'])))
    teams_string = [str(team) for team in teams]
    teams_string = [re.sub(r"\s", "_", item) for item in teams_string]    # replace spaces with a '_'
    teams_text = " ".join(teams_string)

    corpus_file = os.path.join(tokenizer_folder_path, 'team_ids_text_uk.txt')
    f = open(corpus_file, "w+", encoding="utf-8")
    f.write(teams_text)
    f.close()

    corpus_files = [corpus_file]
    unknown_token = config['unknown_token']
    special_tokens = [unknown_token] ################### + ["[HOME]", "[AWAY]"]
    vocab_size = len(teams_string) + len(special_tokens)

    tokenizer_team = createSimpleTokenizer(corpus_files, vocab_size, unknown_token, special_tokens)
    return tokenizer_team

#====================================================================  get_id_map_uk ================================================================


def get_id_map_uk(folder, id_to_ids_filename, targetLength, df_total, year_span, testcount=-1):

        def find_Electric_Flow_On_Connected_Graph(graph, source, target, inpuCurrent):
                '''
                Find the flow of electric current on each edge of 'graph' when total flow of 1.0 flows from 'source' to 'target' nodes,
                with conductance of edges stored in edge['conductance'].
                1. If 'source' and 'target' nodes are disconnected with each other, WEIRD amounts on flows on edges.
                2. If 'source' and 'target' nodes have zero conductance between them, it produces WEIRD amount of flows that violate Kirchhoff's law.
                '''
                edges = [e for e in graph.edges]        # (u, v) either u < v or u < v.
                nodes = [v for v in graph.nodes]
                edges_signs = []
                for v in nodes:
                        # We want the orientation of an edge (u, v) to be from min(u, v) to max(u, v)
                        edges_v = [(v, u) for u in nodes if (v, u) in graph.edges]      # all edges that have v as its node, in the form of (v,u)
                        edges_v_plus = [(v, u) for (v, u) in edges_v if v > u]          # (v, u < v)
                        edges_v_minus = [(v, u) for (v, u) in edges_v if v < u]         # (v, u >= v)
                        edges_v_plus = [(edges.index((u, v)) if edges.count((u, v)) > 0 else edges.index((v, u))) for (u, v) in edges_v_plus]
                        edges_v_minus = [(edges.index((u, v)) if edges.count((u, v)) > 0 else edges.index((v, u))) for (u, v) in edges_v_minus]
                        edges_signs.append((edges_v_plus, edges_v_minus))
                matrix_B = [ [ (1 if e in edges_signs[v][1] else -1 if e in edges_signs[v][0] else 0) for e in range(len(edges)) ] for v in range(len(nodes))]
                matrix_B = np.array(matrix_B, dtype=np.float32)
                vector_C = np.array( [graph.edges[e]['conductance'] for e in edges], dtype=np.float32)
                matrix_C = np.diag(vector_C)
                matrix_L = np.dot(np.dot(matrix_B, matrix_C), matrix_B.T)
                inverse_L = np.linalg.pinv(matrix_L, hermitian=True)    # Moore-Penrose pseudo-inverse
                source_node = nodes.index(source)
                target_node = nodes.index(target)
                X_vector = np.zeros((len(nodes),), dtype=np.float32)
                X_vector[source_node] = inpuCurrent
                X_vector[target_node] = - inpuCurrent
                flows = np.matmul(matrix_C, matrix_B.T)
                flows = np.matmul(flows, inverse_L)
                flows = np.matmul(flows, X_vector)

                return flows, nodes, edges   # nodes and edges, just in case list(graph.nodes) might have different order each time.

        def find_nTotalGames(gameGraph):
                nTotalGames = 0
                for e in gameGraph.edges:   nTotalGames += len(gameGraph.edges[e]['games'])
                return nTotalGames

        def reachable(graph, u, v):
                _reachable = True
                try:    nx.shortest_path_length(graph, u, v)
                except: _reachable = False
                return _reachable

        def isConnected(gameGraph):
                connected = True
                for u in gameGraph.nodes:
                        for v in gameGraph.nodes:
                                if not reachable(gameGraph, u, v):
                                        connected = False
                                        break
                return connected

        def create_conducting_game_graph_uk(game_list, baseDate, conductance365):
                # game_list is sorted in (date, div).
                graph = nx.Graph()
                alpha = pow(conductance365, -1/365)
                def conductance(_date):
                        daysAgo = (baseDate - _date).days       # > 0
                        return pow(alpha, -daysAgo)             # pow(conductance365, daysAgo/365) <= 1, as conductance365 < 1

                for id, div, home, away, dt in game_list:
                        if (home, away) not in graph.edges:
                                graph.add_edge(home, away)
                                graph.edges[home, away]['games'] = []
                        graph.edges[home, away]['games'].append((id, dt, conductance(dt)))

                for edge in graph.edges:
                        games = graph.edges[edge]['games']
                        edge_con = 0.0
                        for (_, _, con) in games: edge_con += con
                        graph.edges[edge]['conductance'] = edge_con

                return graph


        def find_games(gameGraph):
                games = []
                for e in gameGraph.edges:   games += [id for (id, date, con) in gameGraph.edges[e]['games']]
                return games

        def try_remove_lowest_flow_games(gameGraph, eFlows, edges, targetLength):
                # Eithe isConnected(gameGraph) or not. This call worsens it by removing some edges.
                emptyPairs = []
                currents = []
                for (teamA, teamB) in gameGraph.edges:  # May not: teamA < teamB
                        # get the current on the edge (teamA, teamB)
                        eId = (edges.index((teamA, teamB)) if edges.count((teamA, teamB)) > 0 else edges.index((teamB, teamA))) # Exists
                        flow = abs(eFlows[eId])
                        
                        if len(gameGraph.edges[teamA, teamB]['games']) <= 0:
                                emptyPairs.append((teamA, teamB))
                        else:
                                edge_conductance = gameGraph.edges[teamA, teamB]['conductance'] # Asserted positive, no need epsilon.
                                currentPerUnitCon = flow / edge_conductance
                                pair_games = gameGraph.edges[teamA, teamB]['games']     # Creates a set of pair expressions.
                                pair_current = [(currentPerUnitCon * cond, id, date, cond, teamA, teamB) for (id, date, cond) in pair_games]
                                currents += pair_current

                # Note: Pair representations (A, B) in currents came from [for (A, B) in gameGraph.edges]
                for (teamA, teamB) in emptyPairs:   gameGraph.remove_edge(teamA, teamB)
                assert find_nTotalGames(gameGraph) == len(currents)

                # sort currents in (curr / descending, date / descending )
                currents = [(date, curr, id, cond, teamA, teamB) for (curr, id, date, cond, teamA, teamB) in currents]
                currents.sort(reverse=True)     # later dates come first
                currents = [(curr, id, date, cond, teamA, teamB) for (date, curr, id, cond, teamA, teamB) in currents]
                currents.sort(reverse=True)    # larger current comes first

                if len(currents) <= targetLength:   pass
                else:
                        # Either zero-current edges survive or positive-current edges are removed by this cut, both leading to disconnected graph. 
                        currents = currents[ : targetLength ]   # note currents are sorted in (curr, date)
                        pairsChanged = []
                        #-----------------------------------------------------------------------------------------------------
                        #   Below, 'currents' is reflected to gameGraph. No more pairs/games are removed, except that.
                        #-----------------------------------------------------------------------------------------------------

                        #======== Find <which games on which pair> are in 'currents'
                        pairsInCurrents = list(set([(teamA, teamB) for (_,_,_,_, teamA, teamB) in currents]))
                        gamesByPairInCurrents = [ ( (teamA, teamB), [(id, date, con) for (_, id, date, con, _teamA, _teamB) in currents 
                                if _teamA == teamA and _teamB == teamB ] )      # currents and pairs_from_current share the same expressions of pair.
                                for (teamA, teamB) in pairsInCurrents]       # May not: teamA < teamB

                        #========= Remove existing pairs that are not in pairsInCurrents, that has no game at all in currents.
                        allPairs = [(teamA, teamB) for (teamA, teamB) in gameGraph.edges]   # Pair representatoin comes from [for (A, B) in gameGraph.edges]
                        pairsToRemove = [(teamA, teamB) for (teamA, teamB) in allPairs if (teamA, teamB) not in pairsInCurrents]
                        for (teamA, teamB) in pairsToRemove:  gameGraph.remove_edge(teamA, teamB)

                        #========= Replace existing games of pairsInCurrents with games found in 'currents' if appropriate.  
                        for ((teamA, teamB), games) in gamesByPairInCurrents:
                                if len(gameGraph.edges[teamA, teamB]['games']) != len(games):   # if some games were excluded.
                                        gameGraph.edges[teamA, teamB]['games'] = games  # Replace.
                                        pairsChanged.append((teamA, teamB))

                        #========= Update nTotalGames
                        nTotalGames = find_nTotalGames(gameGraph)
                        assert nTotalGames == targetLength

                return gameGraph, nTotalGames, pairsChanged, currents


        def get_historical_games_intra_div(base_id, base_date, base_div, home, away, div_sub_list, history_len, inputCurrent, conductance365):
                games = []; report = 0
                if len(div_sub_list) <= history_len:
                        games = [id for (id, _, _, _, _) in div_sub_list]
                        report = 10
                else:
                        dgg = create_conducting_game_graph_uk(div_sub_list, base_date, conductance365=conductance365)   # MUCH faster than transform_to_conducting_graph

                        if reachable(dgg, home, away):
                                flows, nodes, edges = find_Electric_Flow_On_Connected_Graph(dgg, home, away, inputCurrent)
                                flows_copy = copy.deepcopy(flows)
                                flows_copy = [abs(f) for f in flows_copy]
                                flows_copy.sort(reverse=True)

                                if  flows_copy[0] > inputCurrent/100:
                                        # Now, either isConnected(dgg) or not. This call worsens it by removing some edges.
                                        dgg, nTotalGames, pairsChanged, currents = try_remove_lowest_flow_games(dgg, flows, edges, history_len)
                                        games = find_games(dgg)
                                        assert len(games) == history_len
                                        report = 20
                                else:
                                        games = [id for (id, div, home, away, dt) in div_sub_list[- history_len : ]]  # collect latest ids
                                        report = 30

                        else: # Very rare. Few games are new edge in the conductance graph after collecting at lease HISTORY_LEN past games in the graph. They might be inter-league games.
                                # find_Electric_Flow_On_Connected_Graph(.) doesn't work here.
                                games = [id for (id, _,_,_,_) in div_sub_list[- history_len : ]]   # collect latest ids
                                report = 40

                return games, report
        
             
     
        def get_historical_games(base_id, base_date, base_div, home, away, sub_list, history_len, inputCurrent, conductance365):
                games = []; report = None
                if len(sub_list) <= history_len:
                        games = [id for (id, _, _, _, _) in sub_list]      # better than dummy games.
                        report = 100
                else:
                        gg = create_conducting_game_graph_uk(sub_list, base_date, conductance365=conductance365)   # MUCH faster than transform_to_conducting_graph

                        if reachable(gg, home, away):
                                flows, nodes, edges = find_Electric_Flow_On_Connected_Graph(gg, home, away, inputCurrent)
                                flows_abs = copy.deepcopy(flows)
                                flows_abs = [abs(f) for f in flows_abs]
                                flows_abs.sort(reverse=True)

                                if  flows_abs[0] > inputCurrent/100: # No other way to find the e flow problem was successful.
                                        # Now, either isConnected(dgg) or not. This call worsens it by removing some edges.
                                        gg, nTotalGames, pairsChanged, currents = try_remove_lowest_flow_games(gg, flows, edges, history_len)
                                        games = find_games(gg)
                                        assert len(games) == history_len
                                        report = 200
                                else:
                                        # either isConnected(gg) or not.
                                        div_sub_list = [(id, div, home, away, dt) for (id, div, home, away, dt) in sub_list if div == base_div]
                                        games, _report = get_historical_games_intra_div(base_id, base_date, base_div, home, away, div_sub_list, history_len, inputCurrent, conductance365)
                                        report = 300 + _report
                                        
                        else: # Very rare. Few games are new edge in the conductance graph after collecting at lease HISTORY_LEN past games in the graph. They might be inter-league games.
                                # find_Electric_Flow_On_Connected_Graph(.) doesn't work here.
                                div_sub_list = [(id, div, home, away, dt) for (id, div, home, away, dt) in sub_list if div == base_div]
                                games, _report = get_historical_games_intra_div(base_id, base_date, base_div, home, away, div_sub_list, history_len, inputCurrent, conductance365)
                                report = 400 + _report

                        if len(games) < history_len:
                                candi_games = [id for (id, _, _, _, _) in sub_list if id not in games]
                                games = games + candi_games[ : history_len - len(games)]
                                report += 1     # 200, 311, 320, 420, 330

                        assert len(games) == history_len

                        # if not isConnected(gg): report += 1000         # expensive

                games.sort(reverse=True)
                return games, report
        
        def sort_id_to_ids(id_to_ids):
                dates_ids = [(baseId, report, games) for (baseId, (report, games)) in id_to_ids.items()]
                dates_ids.sort()        # increasing on baseId
                # print('dates_ids', dates_ids)
                id_to_ids = {int(baseId): (report, games) for (baseId, report, games) in dates_ids}
                return id_to_ids
        
        def save_step_id_to_ids(path, step_id_to_ids, work_id_to_ids, old_step):
                save = {}
                if old_step >= 0:
                        if len(work_id_to_ids) > 0:   # Don't save anew unless we have extra id_to_ids, because this saving sometimes saves a wrong file.'"Electrical Flows 2.pdf"
                                save = step_id_to_ids | sort_id_to_ids(work_id_to_ids)
                                SaveJsonData(save, path)
                        else:
                                save = step_id_to_ids
                return save

        #----------------------------------------------- Main --------------------------------------------------------

        id_list = list(df_total['id']); div_list = list(df_total['Div']); home_list = list(df_total['HomeTeam']); away_list = list(df_total['AwayTeam']); date_list = list(df_total['Date'])
        total_list = list(zip(id_list, div_list, home_list, away_list, date_list))

        step_size = int(1E3)        # do not change.
        old_step = -1
        total_id_to_ids = {}
        step_id_to_ids = {}
        work_id_to_ids = {}

        # df_built = df_built.sort_values(['Date', 'Div'], ascending=[True, True])
        
        max_days_covered = 0; count = 0
        for (base_id, div, home, away, base_date) in total_list:      # date: yyyy-mm-dd
                if count == testcount: break
                count += 1

                step = int(base_id/step_size) * step_size   # sure step >= 0

                def build_path(step):
                        return os.path.join(folder, id_to_ids_filename + '-step-' + str(step) + '-size-' + str(step_size) + '.json')

                # Note the final step is always not saved. Save it after this loop.
                if step != old_step:    # We are turning to a new step.
                        save = save_step_id_to_ids(build_path(old_step), step_id_to_ids, work_id_to_ids, old_step)
                        total_id_to_ids = total_id_to_ids | save
                        step_id_to_ids = {}
                        path = build_path(step)
                        id_to_ids_read = LoadJsonData(path)
                        if id_to_ids_read is not None:
                                step_id_to_ids = id_to_ids_read
                        work_id_to_ids = {}
                        old_step = step
                
                if str(base_id) in step_id_to_ids.keys():  continue

                #------------------------------------------------------------------------------------------------------- Goal: get games.
                #????????????????????????????????????? Shall we limit the list to max 5 years ?????????????????????????????????????????????????
                day_span = year_span * 365
                sub_list = [(id, div, home, away, dt) for (id, div, home, away, dt) in total_list if id < base_id and (base_date-dt).days <= day_span]
                inputCurrent = 1000.0
                games, report = get_historical_games(base_id, base_date, div, home, away, sub_list, targetLength, inputCurrent, conductance365=0.9)

                if len(games) > 0: days_covered = (base_date - date_list[id_list.index(games[-1])]).days
                else: days_covered = 0
                if max_days_covered < days_covered: max_days_covered = days_covered

                print("base_id: {}, report: {}, days_span: {}, games[:10]: {}" \
                      .format(base_id, report, days_covered, games[:10]), end='\r')
                #-------------------------------------------------------------------------------------------------------

                if len(games) >= 0: work_id_to_ids[base_id] = (report, games)

        # Give a chance to the final step to save.
        save = save_step_id_to_ids(build_path(old_step), step_id_to_ids, work_id_to_ids, old_step)
        total_id_to_ids = total_id_to_ids | save

        return total_id_to_ids


#==================================================== generate_dataset_uk =========================================================


def generate_dataset_uk(df_total, fixture_id_to_ids, tokenizer_team, normalization_parms, data_columns, country_spec, max_seq_len, train_mode=True):

    def standardize_raw_bbab(bbab, tokenizer_team, normalization_parms, train_mode=True):

        label = []
        if train_mode:
            #---------------------- label, before changing bbab. They are raw full-time goals and raw odds.
            for col in data_columns._Label_cols:
                start = data_columns.BBAB_cols.index(col)
                label.append(bbab[start])

        #----------------------- 
        start = data_columns.BBAB_cols.index(data_columns.Div_cols[0])
        Div = bbab[start]
        bbab[start] = country_spec.DIVIISONS.index(Div)  # Assumes no n/a

        start = data_columns.BBAB_cols.index(data_columns.Team_cols[0]); end = data_columns.BBAB_cols.index(data_columns.Team_cols[-1]) + 1
        pair_str = [str(team) for team in bbab[start : end]]    # Team names are already normalized, removing/striping spaces.
        pair_text = " ".join(pair_str)
        pair_tokens = tokenizer_team.encode(pair_text).ids
        bbab[start : end] = pair_tokens # 0 for Unknown, by tokenizer trainig.

        #--------------------- normalize
        for col in data_columns._Cols_to_Always_Normalize:   # Odds_cols only now.
            (mean, std, maximum) = normalization_parms[col]
            start = data_columns.BBAB_cols.index(col)
            bbab[start] = (bbab[start] - mean) / std

        #--------------------- columns for positional embedding
        start = data_columns.BBAB_cols.index(data_columns.Date_cols[0])   #
        date = bbab[start]
        bbab[start] = (datetime.datetime.combine(date, datetime.time(0,0,0)) - config['baseDate']).days  # either positive or negative

        #---------------------- bb only
        start = data_columns.BBAB_cols.index(data_columns.BB_cols[0]); end = start + len(data_columns.BB_cols)     # 
        bb = bbab[start : end]

        return bbab, bb, label, date

    def getDateDetails(date):
        baseYear = config['baseDate'].year
        date_details = tf.Variable([date.year - baseYear, date.month, date.day, date.weekday()], dtype=tf.int32, trainable=False)
        return date_details     # (4,)

    filler = tf.zeros_like([0] * len(data_columns.BBAB_cols), dtype=tf.float32)

    def get_data_record(df_total, baseId, ids, tokenizer_team, normalization_parms, data_columns, train_mode=True):
        # try:
            # base_bbab = list(df_grown.loc[df_grown['id'] == baseId, BBAB_cols])
            if train_mode:
                base_bbab = list(df_total[df_total['id'] == baseId][data_columns.BBAB_cols].iloc[0, :])  # base_bbab follows BBAB. list
            else:
                base_bbab = list(df_total[df_total['id'] == baseId][data_columns.BB_cols].iloc[0, :])  # base_bbab follows BB. list

            base_bbab, base_bb, base_label, base_date = standardize_raw_bbab(base_bbab, tokenizer_team, normalization_parms, train_mode=train_mode)
            # base_bbab, base_bb, base_label, base_date
            baseId = tf.Variable(baseId, dtype=tf.int32, trainable=False)
            base_bbab = tf.Variable(base_bbab, dtype=tf.float32, trainable=False)    # (len(BBAB_cols),)
            base_bb = tf.Variable(base_bb, dtype=tf.float32, trainable=False)        # (len(BB_cols),)
            base_label = tf.Variable(base_label, dtype=tf.float32, trainable=False)  # (len(_Label_cols),)
            # print('3', base_bbab)
            # Default sequence.
            sequence = tf.transpose(tf.Variable([[]] * len(data_columns.BBAB_cols), dtype=tf.float32, trainable=False))   # (0, len(BBAB_cols))
            # sequence = np.array([[]] * len(BBAB_cols), dtype=config['np_float']).T
            # print('3.5', sequence)
            baseDateDetails = getDateDetails(base_date) # (4,)

            concat = []
            for id in ids:
                bbab = list(df_total[df_total['id'] == id][data_columns.BBAB_cols].iloc[0, :])   # bbab follows BBAB. list
                # print('4', bbab)
                bbab, _, _, _ = standardize_raw_bbab(bbab, tokenizer_team, normalization_parms, train_mode=train_mode)   # bbab follows BBAB. list
                # check_normalization(bbab, normalization_parms)

                bbab = tf.Variable(bbab, dtype=tf.float32, trainable=False)[tf.newaxis, :]       # (1, len(BBAB_cols))
                # _bbab = bbab[0].numpy()
                # check_normalization(_bbab, normalization_parms)

                concat.append(bbab)     # concat doesn't create a new axis.

            if len(concat) > 0:
                sequence = tf.concat(concat, axis=0)    # (nSequence, len(BBAB_cols))
                # if sequence.shape[0] > 0:
                #     bbab = sequence[0].numpy()
                #     check_normalization(bbab, normalization_parms)

            seq_len_org = sequence.shape[0]
            nMissings = max_seq_len - seq_len_org
            if nMissings > 0:
                patch = tf.stack([filler] * nMissings, axis=0)
                sequence = tf.concat([sequence, patch], axis=0)     # concat doesn't create a new axis. (MAX_TOKENS, len(BBAB_cols))
            base_bb = base_bb[tf.newaxis, :]    # shape: (seq_len = 1, len(BBAB_cols))
            baseDateDetails = baseDateDetails[tf.newaxis, :]
            # mask = tf.Variable([1] * seq_len_org + [0] * nMissings, dtype=tf.int32, trainable=False) # (MAX_TOKENS,) ## DO NOT USE tf.constant !!! unstable.
            # mask = mask[:, tf.newaxis] & mask[tf.newaxis, :]    # (MAX_TOKENS, MAX_TOKENS)

            return (baseId, sequence, base_bb, base_label, baseDateDetails, seq_len_org)


    def generator():
        count = 0
        for baseId, (report, ids) in fixture_id_to_ids.items():
            baseId = int(baseId)
            baseId, sequence, base_bb, base_label, baseDateDetails, seq_len_org = get_data_record(df_total, baseId, ids, tokenizer_team, normalization_parms, data_columns, train_mode=train_mode)
            print("count: {}, baseId: {}".format(count, baseId), end='\r')
            count += 1
            # if count > 200: break
            yield (baseId, sequence, base_bb, base_label, baseDateDetails, seq_len_org)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.int32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32),
        output_shapes=(tf.TensorShape(()), tf.TensorShape((max_seq_len, len(data_columns.BBAB_cols))), tf.TensorShape((1, len(data_columns.BB_cols),)), tf.TensorShape((len(data_columns._Label_cols),)), tf.TensorShape((1, 4)), tf.TensorShape(())),
        args=()
    )
    return ds

#=========================================================

# class PositionalEmbedding(tf.keras.layers.Layer):
class PositionalEmbedding(tf.keras.Model):

    def __init__(self, d_model, definitions):
        super().__init__()
        self.d_model = d_model
        self.definitions = definitions
        # if BALANCE_POS_CODE:
        #     self.total_years = 10   # not sure if it will cover all (baseDayss - sequenceDays)
        # else:
        self.total_years = definitions.SEQ_YEAR_SPAN  # datetime.datetime.now().year - config['baseDate'].year + 1 + 3   # 3 extra years. ------------
        self.total_days = 365 * self.total_years

        positional_resolution = d_model
        quotient = self.total_days / positional_resolution
        positions = tf.constant(range(self.total_days), dtype=tf.float32)    # (total_days,). range [0, total_days)
        fractional_pos = positions / quotient  # (total_days,). range (0, d_model)
        half_depth = d_model/2   #
        depths = tf.range(half_depth, dtype=tf.float32) / half_depth  # (half_depth,). range [0, 1), linear.
        BIG = d_model * 0.8
        depths = 1.0 / tf.pow(BIG, depths)        # (depth,). range [1, 1/BIG)
        angle_rads = fractional_pos[:, tf.newaxis] * depths  # (total_days, half_depth,). range [dayPos, dayPos/BIG) for each day
        pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)   # Seperated sin and cos. (batch, seq_len, d_model)
        self.total_positional_code = pos_encoding
        return

    def call(self, x, sequenceDays, baseDays, isEncoder):

        if self.definitions.IGNORE_HISTORY: pass # x: (batch, MAX_TOKENS or 1, hParams.d_model)
        else:
            if self.definitions.BALANCE_POS_CODE:
                # sequneceDays - (baseDays - totalDays),  baseDays - (baseDays - totalDays)
                positions = tf.cast(baseDays - sequenceDays, dtype=tf.int32) if isEncoder else tf.cast(baseDays - baseDays, dtype=tf.int32)
            else:
                positions = tf.cast(self.total_days - sequenceDays, dtype=tf.int32) if isEncoder else tf.cast(self.total_days - baseDays, dtype=tf.int32)
            positional_code = tf.gather(self.total_positional_code, positions, axis=0)
            if self.definitions.BALANCE_POS_CODE:
                positional_code /= tf.math.sqrt(tf.cast(x.shape[-1], tf.float32)) ################# rather than multiply to x. This makes a balance. Not in "Attention is all you need."
            x = x + positional_code
        return x
    

#================================================

class Preprocess(tf.keras.Model):
    def __init__(self, hParams, isEncoder, definitions, data_columns, country_specs, normalization_params):
        super().__init__()
        self.hParams = hParams
        self.isEncoder = isEncoder
        self.definitions = definitions
        self.data_columns = data_columns
        self.country_specs = country_specs
        self.odds_filter = []
        for id in range(len(country_specs.BOOKIES_FILTER)):
            base = id * definitions.NUM_QUERIES
            self.odds_filter += ([base, base+1, base+2] if country_specs.BOOKIES_FILTER[id] == 1 else [])

        # game
        self.division_emb = tf.keras.layers.Embedding(hParams.nDivisions, hParams.division_embs, dtype=tf.float32, mask_zero=False) # Learn Unknown
        self.team_emb = tf.keras.layers.Embedding(hParams.nTeams, hParams.team_embs, dtype=tf.float32, mask_zero=False) # Learn Unknown

        if self.isEncoder:
            if definitions.EMBED_AB_COLS:
                # AB_cols
                self.firstH_goal_emb = tf.keras.layers.Embedding(hParams.nGoals * hParams.nGoals, hParams.goal_embs, dtype=tf.float32, mask_zero=False)
                self.secondH_goal_emb = tf.keras.layers.Embedding(hParams.nGoals * hParams.nGoals, hParams.goal_embs, dtype=tf.float32, mask_zero=False)
                self.shoot_emb = tf.keras.layers.Embedding(hParams.nShoots * hParams.nShoots, hParams.shoot_embs, dtype=tf.float32, mask_zero=False)
                self.shootT_emb = tf.keras.layers.Embedding(hParams.nShootTs * hParams.nShootTs, hParams.shootT_embs, dtype=tf.float32, mask_zero=False)
                self.corner_emb = tf.keras.layers.Embedding(hParams.nCorners * hParams.nCorners, hParams.corner_embs, dtype=tf.float32, mask_zero=False)
                self.faul_emb = tf.keras.layers.Embedding(hParams.nFauls * hParams.nFauls, hParams.faul_embs, dtype=tf.float32, mask_zero=False)
                self.yellow_emb = tf.keras.layers.Embedding(hParams.nYellows * hParams.nYellows, hParams.yellow_embs, dtype=tf.float32, mask_zero=False)
            else:
                params = []
                for col in data_columns.AB_cols:
                    params.append(normalization_params[col])
                self.AB_mormalization_params = tf.Variable(params, dtype=tf.float32, trainable=False)  # (num AB_cols=14, 3 = <mean, std, maximum>)

        if not self.isEncoder:
            if definitions.DECODE_BASE_DATE:
                self.day_emb = tf.keras.layers.Embedding(31, 2, dtype=tf.float32, mask_zero=False)
                self.month_emb = tf.keras.layers.Embedding(12, 2, dtype=tf.float32, mask_zero=False)
                self.wday_emb = tf.keras.layers.Embedding(7, 2, dtype=tf.float32, mask_zero=False)

        if definitions.IGNORE_HISTORY: pass
        else: self.dimensional_permution = tf.keras.layers.Dense(hParams.d_model)

        self.idx_Days = data_columns.BB_cols.index('Date')

    def representDateDetails(self, dateDetails):
        # dateDetails: (batch, 1, 4)
        bYears, bMonths, bDays, bWDays = tf.split(dateDetails, [1, 1, 1, 1], axis=-1)   # All should be of (batch, seq_len = 1, 1)
        bYears = tf.cast(bYears, dtype=tf.float32)  # (batch, seq_len = 1, 1)
        bDays = self.day_emb(bDays)[:, :, -1]       # (batch, seq_len = 1, embs = 2)
        bMonths = self.month_emb(bMonths)[:, :, -1] # (batch, seq_len = 1, embs = 2)
        bWDays = self.wday_emb(bWDays)[:, :, -1]    # (batch, seq_len = 1, embs = 2)
        # w = tf.Variable(np.math.pi / 25, dtype=tf.float32, trainable=False)    # 25 years are covered by pi or a half circle.
        w = np.math.pi / 25
        bYearsCos = tf.math.cos(bYears * w)
        bYearsSin = tf.math.sin(bYears * w)
        bYears = tf.concat([bYearsCos, bYearsSin], axis=-1)   # (batch, seq_len = 1, 1+1 = 2)
        return bYears, bMonths, bDays, bWDays

    def combined_embeddings_of_double_columns(self, emb_layer, columns, nValues):
        # Assume emb_layer = Embedding(nValues * nValues, embs, mask_zero=False)
        cols = tf.cast(columns, dtype=tf.int32)
        cols = tf.clip_by_value(cols, 0, nValues-1)
        combi = cols[:, :, 0] * nValues + cols[:, :, 1]   # (batch, seq_len, 1). [0, ..., nValues * nValues - 1]
        combi = emb_layer(combi)
        return combi    # (batch, seq_len, 1)

    def call(self, x):
        (sequence, base_bb, baseDateDetails, seq_len_org) = x # sob = sequence or base_bb
        sequenceDays = sequence[:, :, self.idx_Days]  # (batch, seq_len)
        baseDays = base_bb[:, :, self.idx_Days]   # (batch, 1)

        # sequence follows BBAB, whereas base_bb follows 
        
        # BB_cols = id_cols + Div_cols + Date_cols + Team_cols + Odds_cols
        # AB_cols = Goal_cols + Result_cols + Shoot_cols + ShootT_cols + Corner_cols + Faul_cols + Yellow_cols + Red_cols

        if self.isEncoder:
            # ramainder: Shoot_cols + ShootT_cols + Corner_cols + Faul_cols + Yellow_cols + Red_cols  --- total 12 fields.
            id, div, days, teams, odds, half_goals, full_goals, shoot, shootT, corner, faul, yellow\
            = tf.split(sequence, [len(self.data_columns.id_cols), len(self.data_columns.Div_cols), len(self.data_columns.Date_cols), len(self.data_columns.Team_cols), 
                                  len(self.data_columns.Odds_cols), len(self.data_columns.Half_Goal_cols), len(self.data_columns.Full_Goal_cols), len(self.data_columns.Shoot_cols),
                                    len(self.data_columns.ShootT_cols), len(self.data_columns.Corner_cols), len(self.data_columns.Faul_cols), len(self.data_columns.Yellow_cols)], axis=-1)
            # id, div, days, teams, odds, half_goals, full_goals, shoot, shootT, corner, faul, yellow\
            # = tf.split(sequence, [len(self.data_columns.id_cols), len(self.data_columns.Div_cols), len(self.data_columns.Date_cols), len(self.data_columns.Team_cols), 
            #     len(self.country_specs.ODDS_INDICES), len(self.data_columns.Half_Goal_cols), len(self.data_columns.Full_Goal_cols), len(self.data_columns.Shoot_cols),
            #     len(self.data_columns.ShootT_cols), len(self.data_columns.Corner_cols), len(self.data_columns.Faul_cols), len(self.data_columns.Yellow_cols)], axis=-1)


            # All shape of (batch, sequence, own_cols), all tf.flaot32
        else:
            id, div, days, teams, odds, remainder \
                = tf.split(base_bb, [len(self.data_columns.id_cols), len(self.data_columns.Div_cols), len(self.data_columns.Date_cols), len(self.data_columns.Team_cols), len(self.data_columns.Odds_cols), -1], axis=-1) 
            # id, div, days, teams, odds, remainder \
            #     = tf.split(base_bb, [len(self.data_columns.id_cols), len(self.data_columns.Div_cols), len(self.data_columns.Date_cols), len(self.data_columns.Team_cols), len(self.country_specs.ODDS_INDICES), -1], axis=-1)  
            # remainder: [] 
            # All shape of (batch, 1, own_cols), guess., all tf.float32
        
        div = self.division_emb(tf.cast(div, dtype=tf.int32))   # (batch, MAX_TOKENS or 1, columns=1, division_embs)
        div = tf.reshape(div, [div.shape[0], div.shape[1], -1]) # (batch, MAX_TOKENS or 1, extended_columns=1*division_embs) --- 
        teams = self.team_emb(tf.cast(teams, dtype=tf.int32))   # (batch, MAX_TOKENS or 1, columns=2, team_embs)
        teams = tf.reshape(teams, [teams.shape[0], teams.shape[1], -1]) # (batch, MAX_TOKENS or 1, extended_columns=2*team_embs) --- 

        if self.isEncoder:
            if self.definitions.EMBED_AB_COLS:
                first_half_goals = self.combined_embeddings_of_double_columns(self.firstH_goal_emb, half_goals, self.hParams.nGoals)
                second_half_goals = self.combined_embeddings_of_double_columns(self.secondH_goal_emb, full_goals - half_goals,self. hParams.nGoals)
                shoot = self.combined_embeddings_of_double_columns(self.shoot_emb, shoot, self.hParams.nShoots)
                shootT = self.combined_embeddings_of_double_columns(self.shootT_emb, shootT, self.hParams.nShootTs)
                corner = self.combined_embeddings_of_double_columns(self.corner_emb, corner, self.hParams.nCorners)
                faul = self.combined_embeddings_of_double_columns(self.faul_emb, faul, self.hParams.nFauls)
                yellow = self.combined_embeddings_of_double_columns(self.yellow_emb, yellow, self.hParams.nYellows)
                if self.definitions.ODDS_IN_ENCODER:
                    # odds = odds[:, :][self.odds_filter]
                    concat = [div, teams, odds, first_half_goals, second_half_goals, shoot, shootT, corner, faul, yellow]
                else: 
                    concat = [div, teams, first_half_goals, second_half_goals, shoot, shootT, corner, faul, yellow]

            else:   # normalize now
                # AB_cols = Half_Goal_cols + Full_Goal_cols + Shoot_cols + ShootT_cols + Corner_cols + Faul_cols + Yellow_cols
                AB_values = [half_goals, full_goals, shoot, shootT, corner, faul, yellow]   # all (batch, 2 cols)
                AB_values = tf.concat(AB_values, axis=-1) # (batch, seq_len, num_AB_cols=14)
                # self.AB_mormalization_params  # (num AB_cols=14, 3 = <mean, std, maximum>)
                AB_values = (AB_values - self.AB_mormalization_params[:, 0]) / self.AB_mormalization_params[:, 1]
                if self.definitions.ODDS_IN_ENCODER: concat = [div, teams, odds, AB_values]
                else: concat = [div, teams, AB_values]
        else:
            if self.definitions.DECODE_BASE_DATE:
                bYears, bMonths, bDays, bWDays = self.representDateDetails(baseDateDetails)
                if self.definitions.ODDS_IN_DECODER:  
                    # odds = odds[:, :][self.odds_filter]
                    concat = [div, teams, odds, bYears, bMonths, bDays, bWDays]
                else: 
                    concat = [div, teams, bYears, bMonths, bDays, bWDays]
            else:
                if self.definitions.ODDS_IN_DECODER: concat = [div, teams, odds]
                else: concat = [div, teams]

        concat = tf.concat(concat, axis=-1)
        assert concat.shape[-1] <= self.hParams.d_model        

        if self.definitions.IGNORE_HISTORY: pass # concat: (batch, MAX_TOKENS or 1, hParams.d_model)
        else:
            concat = self.dimensional_permution(concat)  # (batch, MAX_TOKENS or 1, hParams.d_model)

        #---------------------- Construct mask. Can we find a faster tensorflowy algorithm?
        mask = []
        for id in range(seq_len_org.shape[0]):      # for len in seq_len_org: does not work with @tf.funciotn
            m = tf.concat([tf.ones(seq_len_org[id], dtype=tf.int32), tf.zeros(self.definitions.HISTORY_LEN - seq_len_org[id], dtype=tf.int32)], axis=-1)
            m = m[:, tf.newaxis] & m[tf.newaxis, :]
            mask.append(m)
        mask = tf.stack(mask, axis=0)   # (batch, MAX_TOEKNS, MAX_TOKENS)

        if self.isEncoder:  mask = mask     # (batch, MAX_TOKEN, MAX_TOKEN)
        else:   mask = mask[:, 0:concat.shape[1], :]    # concat: (batch, 1, MAX_TOKEN)

        return concat, mask, sequenceDays, baseDays
    

#=========================================== Transformer =========================================================

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
      # tf.keras.layers.Dropout(dropout_rate)
  ])

def scaled_dot_product_attention(q, k, v, mask=None):
  """
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v), seq_len_k == sel_len_v
    mask: Float 0/1 tensor with shape broadcastable to (..., seq_len_q, seq_len_k). 1 surpresses the score to zero.
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # (..., seq_len_q, seq_len_k)

  if mask is not None: scaled_attention_logits += (tf.cast(mask, dtype=tf.float32) * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len_q, d_model)
    k = self.wk(k)  # (batch_size, seq_len_k, d_model)
    v = self.wv(v)  # (batch_size, seq_len_v, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, d_head)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, d_head)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, d_head)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, d_head)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, d_head)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):
    # value = x, key = x, query = x
    self_att, self_att_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    self_att = self.dropout1(self_att, training=training)
    out1 = self.layernorm1(x + self_att)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out
  
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, context, training, 
           look_ahead_mask, padding_mask):
    # x: (batch, target_seq_len, d_model), context.shape: (batch_size, input_seq_len, d_model)

    # value = x, key = x, query = x
    self_att, self_att_weights = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len = 1, d_model)
    self_att = self.dropout1(self_att, training=training)
    self_att = self.layernorm1(self_att + x)

    # value = context, key = context, query = self_att
    cross_att, cross_att_weights = self.mha2(context, context, self_att, padding_mask)  # (batch_size, target_seq_len, d_model)
    cross_att = self.dropout2(cross_att, training=training)
    out2 = self.layernorm2(cross_att + self_att)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out, self_att_weights, cross_att_weights
  
class Encoder(tf.keras.layers.Layer):
    def __init__(self, hParams, dropout_rate=0.1):
      super().__init__()
      self.d_model = hParams.d_model
      self.num_layers = hParams.num_layers
      self.dropout = tf.keras.layers.Dropout(dropout_rate)
      self.enc_layers = [
          EncoderLayer(d_model=hParams.d_model, num_heads=hParams.num_heads, dff=hParams.d_model * 4, rate=dropout_rate)
          for _ in range(hParams.num_layers)]

    def call(self, x, training, encMask):
      x = self.dropout(x, training=training)
      for encoder_layer in self.enc_layers:
        x = encoder_layer(x, training, encMask)
      return x  # (batch_size, max_tokens, d_model)
 
class Decoder(tf.keras.layers.Layer):
    def __init__(self, hParams, definitions, dropout_rate=0.1):
      super(Decoder, self).__init__()
      self.definitions = definitions
      self.d_model = hParams.d_model
      self.num_layers = hParams.num_layers
      self.dropout = tf.keras.layers.Dropout(dropout_rate)
      if self.definitions.IGNORE_HISTORY: pass
      else:
        self.dec_layers = [
            DecoderLayer(d_model=hParams.d_model, num_heads=hParams.num_heads, dff=hParams.d_model * 4, rate=dropout_rate)
            for _ in range(hParams.num_layers)]

    def call(self, x, context, training, look_ahead_mask, padding_mask):
      if self.definitions.IGNORE_HISTORY: pass
      else:
        x = self.dropout(x, training=training)
        for decoder_layer in self.dec_layers:
          x, _, _  = decoder_layer(x, context, training, look_ahead_mask, padding_mask)
      return x
  
class Transformer(tf.keras.Model):
    def __init__(self, hParams, definitions, data_columns, country_specs, normalization_params, dropout_rate=0.1):
      super().__init__()
      self.definitions = definitions
      if self.definitions.IGNORE_HISTORY:
        self.all_one_context = tf.ones((self.definitions.BATCH_SIZE, self.definitions.MAX_TOKENS, hParams.d_model), dtype=tf.float32) # (batch, max_tokens, d_model)
      else:
        self.encPreprocess = Preprocess(hParams, isEncoder=True, definitions=definitions, data_columns=data_columns, country_specs=country_specs, normalization_params=normalization_params)
        self.decPreprocess = Preprocess(hParams, isEncoder=False, definitions=definitions, data_columns=data_columns, country_specs=country_specs, normalization_params=normalization_params)
        self.posEmbedding = PositionalEmbedding(hParams.d_model, definitions)
        self.encoder = Encoder(hParams, dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(hParams.d_model) #-------------- to modify
      self.decoder = Decoder(hParams, self.definitions, dropout_rate=dropout_rate)
      self.one = tf.constant(1, dtype=tf.int32)

    def call(self, input, training=False):
      # inputs = (sequence, base_bb, baseDateDetails, mask)
      # sequence: (batch, max_token, aabb), base: (batch, 1, bb), baseDateDetails: (batch, 1, 4), mask: (batch, max_token, max_token)
      if self.definitions.IGNORE_HISTORY: 
        context = self.all_one_context
      else:
        x, encMask, sequenceDays, baseDays = self.encPreprocess(input) # (batch, max_tokens, d_model), (batch, max_tokens, max_tokens), (batch, max_tokens), (batch, 1)
        x = self.posEmbedding(x, sequenceDays, baseDays, isEncoder=True) # (batch, max_tokens, d_model)
        encMask = self.one - encMask    # NEGATE !!! forgoten for two YEARS !!!
        encMask = encMask[:, tf.newaxis, :, :]  # newaxis: head
        context = self.encoder(x, training, encMask)  # (batch, max_tokens, d_model). Only sequence and mask are used.

      x, decMask, sequenceDays, baseDays = self.decPreprocess(input) # (batch, 1, d_model), (batch, 1, max_tokens), (batch, max_tokens), (batch, 1)
      x = self.posEmbedding(x, sequenceDays, baseDays, isEncoder=False) # (batch, 1, d_model)
      decMask = decMask[:, tf.newaxis, :, :]
      # look_ahead_mask is None, which means [[0]], as there is only one position in x, so is nothing to mask when doing mha(value=x, key=x, query=x).
      x = self.decoder(x, context, training, look_ahead_mask=None, padding_mask=decMask)  # (batch, 1, d_model).  Only base_bb, baseDateDetails, and mask are used.      

      logits = self.final_layer(x)  # (batch, 1, d_model)
      logits = tf.squeeze(logits, axis=-2)  # (batch, d_model)

      return logits

#==========================================================

class Dense_Add_Norm(tf.keras.layers.Layer):
    def __init__(self, dim, seed):
      super().__init__()
      self.dense =tf.keras.layers.Dense (dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=seed), bias_initializer=Zeros, activation='tanh')
      self.add = tf.keras.layers.Add()
      self.layernorm = tf.keras.layers.LayerNormalization()
    def call(self, x):
      dense = self.dense(x, training=False)
      x = self.add([x, dense])  # x.shape[-1] == dim
      x = self.layernorm(x)
      return x

Zeros = tf.keras.initializers.Zeros()

# Used for earlier versions that don't allow mixing bookies.
class Adaptor(tf.keras.Model):
  def __init__(self, nLayers, d_main, d_output, dropout_rate=0.1):
    super().__init__()
    # total (nLayers + nLayers) dims = 2 * nLayers dims
    dims = [d_main] * nLayers
    layers = [Dense_Add_Norm(dims[id], id) for id in range(len(dims))]
    self.seq = tf.keras.Sequential(layers)
    self.initial = tf.keras.layers.Dense (d_main, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=23), activation='tanh')
    self.final = tf.keras.layers.Dense (d_output, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=23), activation='tanh')
  def call(self, x, training=False):  # (batch, d_model)
    x = self.initial(x)
    x = self.seq(x)   # (batch, d_model)
    x = self.final(x) # (batch, nBookies * 3)
    return x


#===========================================================

class Model_1X2(tf.keras.Model):
    softmax = tf.keras.layers.Softmax(axis=-1)

    def __init__(self, hParams, definitions, data_columns, country_specs, normalization_params, dropout_rate=0.1):
        super().__init__()
        self.nQueries = definitions.NUM_QUERIES
        self.definitions = definitions
        self.country_specs = country_specs
        
        self.odds_filter = []
        for id in range(len(country_specs.BOOKIES_FILTER)):
            base = id * definitions.NUM_QUERIES
            self.odds_filter += ([base, base+1, base+2] if country_specs.BOOKIES_FILTER[id] == 1 else [])
        
        self.transformer = Transformer(hParams, definitions, data_columns, country_specs, normalization_params, dropout_rate=dropout_rate)
        #   self.bookies = ['B365', 'Betfair', 'Interwetten', 'William']
        self.bookies = ['HDA' + str(b) for b in range(country_specs.NUMBER_BOOKIES_DATA)]  #============= Defines the number of bookies, not which bookies.
        # self.bookies = ['HDA' + str(id) for id in range(country_specs.NUMBER_BOOKIES_ALGO) if country_specs.BOOKIES_FILTER[id] == 1]

        if definitions.SIMPLIFY_ADAPTOR:
            self.adaptor = tf.keras.layers.Dense(len(self.bookies) * self.nQueries)
        else:
            self.adaptor = Adaptor(definitions.ADAPTORS_LAYERS, hParams.d_model * definitions.ADAPTORS_WIDTH_FACTOR, self.nQueries * len(self.bookies))
     
        if definitions.LOSS_TYPE == 'entropy':
            self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1, reduction='sum_over_batch_size')
        return

    def call(self, x, trainig=False):
        x = self.transformer(x, training=trainig) # (batch, d_model)
        output = self.adaptor(x)   # (batch, nBookies * nQueries)
        output = tf.reshape(output, [output.shape[0], self.nQueries, -1])    # (batch, nQueries, nBookies)
        output = tf.transpose(output, perm=[2, 0, 1])     # (nBookies, batch, nQueries)
        if self.definitions.MODEL_ACTIVATION == 'softmax':
            output = tf.nn.softmax(output)  # (nBookies, batch, nQueries)   #
        elif self.definitions.MODEL_ACTIVATION == 'sigmoid':
            output = tf.math.sigmoid(output * 5)  # the previous activation is tanh, ranging (-1, 1). Multiplier 5 will result in range (near 0, near 1)
        elif self.definitions.MODEL_ACTIVATION == 'relu':
            output = tf.nn.relu(output)
        elif self.definitions.MODEL_ACTIVATION == 'open':
            pass    # output = output
        return output
    
    def h_true(self, ftGoals):  # Defines this QGroup. This is for 1X2 QGroup. Derived classes re-define this funciton.
        # ftGoals:  (batch, 2)
        ftGoals = tf.cast(ftGoals, dtype=tf.int32)  # (batch, 2)
        h = (tf.math.greater(ftGoals[..., 0], ftGoals[..., 1]), tf.math.equal(ftGoals[..., 0], ftGoals[..., 1]), tf.math.less(ftGoals[..., 0], ftGoals[..., 1]))
        h = tf.cast(tf.transpose(h), dtype=tf.float32)  # (batch, nQueries)
        return h

    def loss(self, y, output):   
        # y: (batch, len(Team_cols)+len(Odds_cols)) 
        # output: (nBookies, batch, nQueries)

        def well(x, mu, sigma):
            norm = tf.norm(x)
            return tf.nn.relu(norm - sigma) * 0.5   # so this component of loss is less steep than the other component.
            # return - tf.math.exp(- tf.math.pow((x-mu)/sigma, 2) / 2) / (sigma * tf.math.sqrt(np.pi*2))

        ftGoals, odds = tf.split(y, [2, -1], axis=-1) # (batch, 2), (batch, self.nQueries * len(self.bookies))
        odds = tf.split(odds, [self.nQueries] * len(self.bookies), axis=-1)  # [(batch, nQueries)] * nBookies  
        odds = tf.stack(odds, axis=0)  # (nBookies, batch, nQueries)     #=================== This is the place to define which bookies to have.
        happen_t = self.h_true(ftGoals) # (batch, nQueries)
        oh = tf.math.multiply(odds, happen_t)   # (nBookies, batch, nQueries)

        if self.definitions.LOSS_TYPE != 'entropy':
            (stake_p) = output  # (nBookies, batch, nQueries)
            # -----------------------------------------------------------------------------------------
            # Note: happen_p and stake_p are not converted to one-hot values, unlike they should.
            # Note: Do not normalize stake_p. It can learn whether to bet or not, as well as betting direction.
            #------------------------------------------------------------------------------------------
            profit_per_bookie_game = tf.reduce_sum(tf.math.multiply(oh - 1.0, stake_p), axis=-1)    # (nBooies, batch)
            mean_profit_by_game = tf.reduce_mean(profit_per_bookie_game, axis=0)    # (batch,)
            profit_backtest_sum = tf.reduce_sum(mean_profit_by_game, axis=None)  # () 
            loss_sum = - profit_backtest_sum  # U.action.42

            if self.definitions.MODEL_ACTIVATION == 'open':
                bell_loss = well(stake_p, 0, 2)
                loss += bell_loss

            if not self.definitions.VECTOR_BETTING:
                one_hot_stake_p = tf.squeeze(tf.one_hot(tf.nn.top_k(stake_p).indices, tf.shape(stake_p)[-1]), axis=-2)   # one_hot stake_p, (nBookies, batch, nQueries)
                profit_per_bookie_game = tf.reduce_sum(tf.math.multiply(oh - 1.0, one_hot_stake_p), axis=-1)    # (nBooies, batch)
                mean_profit_by_game = tf.reduce_mean(profit_per_bookie_game, axis=0)    # (batch,)
                profit_backtest_sum = tf.reduce_sum(mean_profit_by_game, axis=None)  # ()

        else:
            probability_p = tf.transpose(output, perm=[1, 2, 0])
            probability_p = tf.math.reduce_mean(probability_p, axis=-1) # (batch, nQueries)
            loss_sum = self.categorical_crossentropy(happen_t, probability_p)   # reduce sum over batch size. What over the axis -1.

            one_hot_stake_p = tf.squeeze(tf.one_hot(tf.nn.top_k(probability_p).indices, tf.shape(probability_p)[-1]), axis=1)   # one_hot stake_p
            profit_per_bookie_game = tf.reduce_sum(tf.math.multiply(oh - 1.0, one_hot_stake_p), axis=-1)    # (nBooies, batch)
            mean_profit_by_game = tf.reduce_mean(profit_per_bookie_game, axis=0)    # (batch,)
            profit_backtest_sum = tf.reduce_sum(mean_profit_by_game, axis=None)  # ()
    
        return loss_sum, profit_backtest_sum # (), ()  Bot loss_sum and profit_backtest_sum are a sum across batch. 
    
    
    def backtest_event_wise(self, y, output, key_a, key_b):
        # y: (batch, len(Team_cols)+len(Odds_cols)) 
        # output: (nBookies, batch, nQueries)
        ftGoals, odds = tf.split(y, [2, -1], axis=-1) # (batch, 2)!, (batch, self.nQueries * len(self.bookies))
        odds = tf.split(odds, [self.nQueries] * len(self.bookies), axis=-1)  # [(batch, nQueries)] * nBookies
        odds = tf.stack(odds, axis=0)  # (nBookies, batch, nQueries)!
        # odds and tfGoals were not normalized, so you don't need de-normalize them.
        happen_t = self.h_true(ftGoals) # (batch, nQueroes)
        oh = tf.math.multiply(odds, happen_t)   # (nBookies, batch, nQueries)
        (stake_p) = output

        # -----------------------------------------------------------------------------------------
        # Note: oh_p and stake_p are not converted to one-hot values, unlike they should.
        #------------------------------------------------------------------------------------------
        def analyze(stake_p):
            # sum_stake_p = tf.math.reduce_sum(stake_p, axis=-1)
            norm = tf.norm(stake_p, keepdims=True, axis=-1) + 1e-12
            profit_p = tf.math.reduce_sum(tf.math.multiply(odds * (stake_p / norm) - 1.0, stake_p), axis=-1)# (nBookies, batch)
            profit_backtest = tf.math.reduce_sum(tf.math.multiply(oh - 1.0, stake_p), axis=-1) # (nBookies, batch)
            condition = tf.math.logical_and(key_a <= profit_p, profit_p <= key_b)   # (nBookies, batch), dtype=tf.bool
            indices = tf.where(condition)   # (n, 2)    values: (bookieId, gameId)
            profit_backtest = profit_backtest[condition]  # (n,)    values: (profit)
            stake = stake_p[condition]  # (n, nQueries)
            # assert indices.shape[0] == profit_backtest.shape[0]
            return indices, stake, profit_backtest

        if self.definitions.VECTOR_BETTING:
            indices, stake, profit_backtest = analyze(stake_p)
        else:
            bestQuery = tf.argmax(stake_p, axis=-1)     #   (nBookies, batch, nQueries)
            stake_p = tf.one_hot(bestQuery, self.nQueries, dtype=tf.float32)   #   (nBookies, batch, nQueries)
            indices, stake, profit_backtest = analyze(stake_p)

        # assert indices.shape[0] == profit_backtest.shape[0]
        return indices, stake, profit_backtest  # Not normalized. i.e. not divided with the norm of stake_p
    
    def get_batch_backtest(self, y, output):
        # y: (batch, len(Team_cols)+len(Odds_cols)) 
        # output: (nBookies, batch, nQueries)
        ftGoals, odds = tf.split(y, [2, -1], axis=-1) # (batch, 2)!, (batch, self.nQueries * len(self.bookies))
        odds = tf.split(odds, [self.nQueries] * len(self.bookies), axis=-1)  # [(batch, nQueries)] * nBookies
        odds = tf.stack(odds, axis=0)  # (nBookies, batch, nQueries)!
        # odds and tfGoals were not normalized, so you don't need de-normalize them.
        happen_t = self.h_true(ftGoals) # (batch, nQueroes)
        oh = tf.math.multiply(odds, happen_t)   # (nBookies, batch, nQueries)
        (stake_p) = output
        sum = tf.math.reduce_sum(stake_p, keepdims=True, axis=-1) + 1e-12
        indicatorP = tf.math.reduce_sum(tf.math.multiply(odds * (stake_p / sum) - 1.0, stake_p), axis=-1)  # (nBookies, batch)
        backtestP = tf.math.reduce_sum(tf.math.multiply(oh - 1.0, stake_p), axis=-1)  # (nBookies, batch)
        indices = tf.where(indicatorP == indicatorP)    # (nBookies * batch, 2 [bookieId, rel_gameId])
        indicatorP = tf.reshape(indicatorP, [-1])   # (nBookies * batch, )
        backtestP = tf.reshape(backtestP, [-1])     # (nBookies * batch, )
        stake_p = tf.reshape(stake_p, [-1, self.nQueries])
        # (nBookies * batch,), (nBookies * batch,), (nBookies * batch, 2 [bookieId, rel_gameId]), (nBookies * batch, 3 [nQueries])
        return indicatorP, backtestP, indices, stake_p
    
#================================================= Miscellaneous ==========================================================

# @tf.function  #-------------------- Weird: no work.
def backtest_with_dataset(dataset, profit_keys):
    profits = [-1.0] * len(profit_keys)
    casts = [0] * len(profit_keys)
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(dataset):
        x = (sequence, base_bb, baseDateDetails, mask); y = value

        outputs = model_1x2(x, training=False)  #
        profit_list, cast_list = model_1x2.backtest(y, outputs, profit_keys)

        for p, c, id in zip(profit_list, cast_list, range(len(profit_keys))):
            if c > 0:
                profits[id] = (profits[id] * casts[id] + p * c) / (casts[id] + c)
                casts[id] = casts[id] + c
    # print('key', profit_back_mean, nBettingsTotal)
    return profits, casts

# @tf.function  #-------------------- Weird: no work.
def backtest_event_wise_with_dataset(ds_batches, model, key_a, key_b):
    indices = []; stakes = []; profits = []
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(ds_batches):
        x = (sequence, base_bb, baseDateDetails, mask); y = value

        outputs = model(x, training=False)  #
        new_indices, new_stakes, new_profits = model.backtest_event_wise(y, outputs, key_a, key_b)  # Tensor (n, 2 <bookieId, rel_gameId>), (n, nQueries), (n,)

        new_indices = [list(id) for id in new_indices.numpy()]  # [[0,2], ..., [bookie, rel_game_id]]. No tuple but list.
        baseId = list(baseId.numpy())   # absolute game ids
        new_indices = [[bookie, baseId[rel_game_id]] for [bookie, rel_game_id] in new_indices]
        indices = indices + new_indices     # [[0, 123], ..., [[bookie, gameId]]], len = n

        new_stakes = [list(stake) for stake in new_stakes.numpy()]  # [[sHWin, sDraw, sAWin], ...] No tuple but list
        stakes = stakes + new_stakes

        new_profits = list(new_profits.numpy())                 # [0.3, ...]
        profits = profits + new_profits     # [1.3, ..., profit], len = n

    interval_backtests = [(bookie, gameId, stake, profit) for (bookie, gameId), stake, profit in zip(indices, stakes, profits)]  # [ [bookieId ,gameId, (sHWin, sDraw, sAway), p], ...  ]
    return interval_backtests     # [(bookie, gameId, profit) for ...]

# @tf.function  #-------------------- Weird: no work.
def get_backtest_and_indicator_profits(dataset):
    backtestP = indicatorP = []
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(dataset):
        x = (sequence, base_bb, baseDateDetails, mask); y = value
        outputs = model_1x2(x, training=False)  #
        new_indicatorP, new_backtestP = model_1x2.get_backtest_and_indicator_profits(y, outputs)    # Tensor (nBookies * batch), (nBookies * batch)
        indicatorP = indicatorP + list(new_indicatorP.numpy())
        backtestP = backtestP + list(new_backtestP.numpy())
    return indicatorP, backtestP

#-------------------------------------------------------------------------------------------------------------------------------------------------
def get_key_a_key_b(indicatorP, backtestP, indices, stakes, threshold=0.0):
    backtestP = np.array(backtestP, dtype=np.float32)
    indicatorP = np.array(indicatorP, dtype=np.float32)
    indices = np.array(indices, dtype=np.int32)
    stakes = np.array(stakes, dtype=np.float32)
    idx = np.argsort(indicatorP)
    indicatorP = indicatorP[idx]    # increasing order
    backtestP = backtestP[idx]      # hope to be increading order
    indices = indices[idx]
    stakes = stakes[idx]

    ab = []
    for w in range(30, int(len(backtestP))):    # 30
        back = np.convolve(backtestP, np.ones(w), mode='valid') - threshold
        ab += [(back[i], i, i+w-1) for i in range(len(back))]   # [ i : i + w ] : w points. inclusive boundaries.
    ab = [(p, a, b) for (p, a, b) in ab if p > 0]   # a, b : inclusive
    ab.sort(reverse=False)  # False !   The larger the profit, the earlier it comes in ab.
    AB = ab.copy()
    for i in range(len(ab)):   # less valuable intervals are screened first.
        interval_i = ab[i];  P, A, B = interval_i    # sure A < B
        intersects = False
        for j in range(i+1, len(ab)):
            interval_j = ab[j]; p, a, b = interval_j    # sure a < b
            if not (b < A or B < a): intersects = True; break
        if intersects:  AB.remove(interval_i)   # interval_i is unique in AB
    AB.sort(reverse=True); AB = AB[:5]
    print("AB: ", AB)
    [interval_profit, idx_a, idx_b] = AB[0]     # idx_a, idx_b : inclusive
    keyPairs = [[indicatorP[a], indicatorP[b]] for (B, a, b) in AB]
    (key_a, key_b) = keyPairs[0]

    return indicatorP, backtestP, indices, stakes, interval_profit, key_a, key_b, idx_a, idx_b

# @tf.function  # gives a wrong result of tf.where(profit_p > key)
def inference_step(x, odds, interval_a, interval_b):
    stake_p = model_1x2(x, training=False)    # (nBookies, batch, nQueries)
    nQueries = stake_p.shape[-1]
    profit_p = tf.math.reduce_sum(tf.math.multiply(odds * stake_p - 1.0, stake_p), axis=-1)  # (nBookies, batch)

    bet_bool = interval_a <= profit_p and profit_p >= interval_b    # (nBookies, batch)
    bet_bool = tf.stack([bet_bool] * nQueries, axis=-1) # (nBookies, batch, nQueries)
    stake_vector = tf.math.multiply(stake_p, tf.cast(bet_bool, dtype=tf.float32))   # (nBookies, batch, nQueries)
    stake_vector = tf.reshape(stake_vector, [1, 0, 2])  # (batch, nBookies, nQueries)
    return stake_vector

# @tf.function  #-------------------- Weird: no work.
def inference_with_dataset(dataset, interval_a, interval_b): 
    vectors = []
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(dataset):
        x = (sequence, base_bb, baseDateDetails, mask)
        id, div, days, teams, odds, remainder \
            = tf.split(base_bb, [len(id_cols), len(Div_cols), len(Date_cols), len(Team_cols), len(Odds_cols), -1], axis=-1)
        stake_vector = inference_step(x, odds, interval_a, interval_b)  # (batch, nBookies, nQueries)
        vectors.append(stake_vector)
    
    stake_vectors = tf.concat(vectors, axis=0)   # (batch, nBookies, nQueries)
    return stake_vectors    # (batch, nBookies, nQueries)

def get_dataset_backtest(model, dataset):
    backtestP = indicatorP = indices = stakes = []

    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(dataset):
        x = (sequence, base_bb, baseDateDetails, mask); y = value
        outputs = model(x, training=False)
        # (nBookies * batch,), (nBookies * batch,), (nBookies * batch, 2 [bookieId, rel_gameId]), (nBookies * batch, 3 [nQueries])
        new_indicatorP, new_backtestP, new_indices, new_stakes = model.get_batch_backtest(y, outputs)
        indicatorP = indicatorP + list(new_indicatorP.numpy())  # + [1.2, ...], len = nBookies * batch
        backtestP = backtestP + list(new_backtestP.numpy())     # + [1.2, ...], len = nBookies * batch

        new_indices = [list(id) for id in new_indices.numpy()]  # [[0,2], ...], len = nBookies * batch. No tuple but list.
        baseId = list(baseId.numpy())   # absolute game ids
        new_indices = [[bookie, baseId[rel_game_id]] for [bookie, rel_game_id] in new_indices]
        indices = indices + new_indices     # + [[0, 1000123], ..., [[bookie, gameId]]], len = nBookies * batch

        new_stakes = [list(stake) for stake in new_stakes.numpy()]  # [[sHWin, sDraw, sAWin], ...] No tuple but list
        stakes = stakes + new_stakes    # + [[sHWin, sDraw, sAWin], ...], len = nBooks * batch

    indicatorP_permuted, backtestP_permuted, indices_permutated, stakes_permutated, interval_profit, interval_a, interval_b, idx_a, idx_b = \
        get_key_a_key_b(indicatorP, backtestP, indices, stakes, threshold=0.0)  # idx_a, idx_b : inclusive

    interval_backtests = [(bookie, gameId, list(stake), profit) for (bookie, gameId), stake, profit in \
        zip(indices_permutated[idx_a:idx_b+1], stakes_permutated[idx_a:idx_b+1], backtestP_permuted[idx_a:idx_b+1])]  # [ [bookieId ,gameId, (sHWin, sDraw, sAway), p], ...  ]

    return interval_backtests, indicatorP_permuted, backtestP_permuted, interval_a, interval_b, interval_profit, idx_a, idx_b


#================================================ History ===============================================

class history_class():
    def round_sig(self, x, sig=2):
            return x
            # return round(x, sig-int(math.floor(math.log10(abs(x))))-1)    # domain error for VERY small numbers.
    def __init__(self, filepath):
        self.filepath = filepath
        self.history = {'initial_profit': -float('inf'), 'loss': [], 'val_loss_0': [], 'val_profit_0': [], 'val_loss': [], 'val_profit': [], 'learning_rate': [], 'recall': [], 'precision': [], 'time_taken': [],  'gauge': [], 'backtest_reports': []}
    def set_initial_interval_profit(self, initail_inverval_profit):
        self.history['initial_profit'] = initail_inverval_profit
        self.save()
    def removeFile(self):
        files = glob.glob(self.filepath + "*")   # "*.*" may not work
        result = [os.remove(file) for file in files]
    def save(self):
        SaveJsonData(self.history, self.filepath)
    def reset(self):
        self.removeFile()
        # forgot to reset self.history? ---------------------- Check it.
        self.__init__(self.filepath)
        self.save()
    def load(self):
        history = LoadJsonData(self.filepath)
        if history is not None: self.history = history

    def append(self, train_loss, val_loss_0, val_profit_0, val_loss, val_profit, learning_rate, recall, precision, time_taken, gauge, backtest_report):
        self.history['loss'].append(self.round_sig(train_loss, 4))
        self.history['val_loss_0'].append(self.round_sig(val_loss_0, 4))
        self.history['val_profit_0'].append(self.round_sig(val_profit_0, 4))
        self.history['val_loss'].append(self.round_sig(val_loss, 4))
        self.history['val_profit'].append(self.round_sig(val_profit, 4))
        self.history['learning_rate'].append(learning_rate)
        self.history['recall'].append(self.round_sig(recall, 4))
        self.history['precision'].append(self.round_sig(precision, 4))
        self.history['time_taken'].append(time_taken)
        self.history['gauge'].append(gauge)
        self.history['backtest_reports'].append(backtest_report)
        self.save()

    def get_zipped_history(self):
        z = zip(self.history['loss'], self.history['val_loss_0'], self.history['val_profit_0'], self.history['val_loss'], self.history['val_profit'], self.history['learning_rate'], self.history['recall'], self.history['precision'], self.history['time_taken'])
        return list(z)
    # def append_backtests(self, epoch, key_a, key_b, interval_profit, backtest):
    #     self.history['backtests'].append((epoch, key_a, key_b, interval_profit, backtest))
    #     self.save()
    def get_backtest_report(self, epoch):
        return self.history['backtest_reports'][epoch]     # sure exists. epoch is selected.
    def len(self):
        assert len(self.history['loss']) == len(self.history['val_loss_0'])
        assert len(self.history['loss']) == len(self.history['val_profit_0'])
        assert len(self.history['loss']) == len(self.history['val_loss'])
        assert len(self.history['loss']) == len(self.history['val_profit'])
        assert len(self.history['loss']) == len(self.history['recall'])
        assert len(self.history['loss']) == len(self.history['precision'])
        assert len(self.history['loss']) == len(self.history['learning_rate'])
        assert len(self.history['loss']) == len(self.history['time_taken'])
        assert len(self.history['loss']) == len(self.history['gauge'])
        assert len(self.history['loss']) == len(self.history['backtest_reports'])
        return len(self.history['loss'])
    def get_min_val_loss(self):
        return float('inf') if self.len() <= 0 else min(self.history['val_loss'])
    def get_max_gauge(self, epoch):
        gauges = self.history['gauge']
        return -float('inf') if (len(gauges) <= 0 or epoch <= 0) else max(gauges[:epoch])
    def replace_gauge(self, epoch, gauge):
        self.history['gauge'][epoch] = gauge;   self.save()
    def show(self, ax, definitions, show_val_0=True):
        ax.set_title(definitions.TEST_ID + ": loss history")

        ax.plot(self.history['loss'], label='train_loss')
        if show_val_0: ax.plot(self.history['val_loss_0'], label='GET_VAL_LOSS_0')
        ax.plot(self.history['val_loss'], label='val_loss')
        ax.plot([-v for v in self.history['val_loss']], label='vector_profit')
        ax.plot(self.history['val_profit'], label='scalar_profit')
        
        ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_ylabel('loss or profit')
        ax.set_xlabel('epoch', loc='right')

#================================================== F1 scores ===================================================

class recall():
  def __init__(self, **kwargs):
    self.n = None
    self.recall = None
    self.reset()

  def update(self, label, pred):    # (batch,)
    label = tf.cast(label, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)
    hit_positives = tf.math.reduce_sum(label * pred, axis=None)
    labeled_positives = tf.math.reduce_sum(label, axis=None)
    recall = hit_positives / (labeled_positives + 1e-9) #tf.keras.backend.epsilon())
    self.n += 1
    self.recall = self.recall * (self.n-1)/self.n + recall / self.n

  def result(self):
    return self.recall
  
  def reset(self):
    self.n = 0
    self.recall = tf.Variable(0.0, dtype=tf.float32, trainable=False)

class precision():
  def __init__(self, **kwargs):
    self.n = None
    self.precision = None
    self.reset()

  def update(self, label, pred):
    label = tf.cast(label, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)
    hit_positives = tf.math.reduce_sum(label * pred, axis=None)
    predicted_positives = tf.math.reduce_sum(pred, axis=None)
    precision = hit_positives / (predicted_positives + 1e-9) #tf.keras.backend.epsilon())
    self.n += 1
    self.precision = self.precision * (self.n-1)/self.n + precision / self.n

  def result(self):
    return self.precision
  
  def reset(self):
    self.n = 0
    self.precision = tf.Variable(0.0, dtype=tf.float32, trainable=False)


# @tf.function
def find_recall_precision(model, ds_batches, recall_object, precision_object):
    recall_object.reset(); precision_object.reset()
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (y)) in enumerate(ds_batches):
        x = (sequence, base_bb, baseDateDetails, mask)
        pred = model(x, training=False)
        pred = tf.cast(pred > 0.5, dtype=tf.int32)
        # Wrong. transorm y to label.   #-------------------------------------------------------- Wrong
        recall_object.update(y, pred); precision_object.update(y, pred)
    return recall_object.result(), precision_object.result()



def accumulate(backtests, compound=False):
    initial = 1.0; Div = 3
    sum = initial; minS = sum; maxS = sum
    for (bookie, gameId, stake, profit) in backtests:
        sum_stake = 0.0
        for s in stake: sum_stake += s
        sum += (profit / sum_stake * ( sum if compound else initial*sum_stake/Div))
        if sum < minS: minS = sum
        if sum > maxS: maxS = sum
        if sum < 0.2: break
        # if sum > initial * 2: initial = sum   # this is a step-wise compound. as risky as simple compound.
    return sum, minS, maxS

#===================================================  


@tf.function    # Removing this decoration leads to GPU OOM!!!
def train_step(model, optimizer, x, y):
    
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)  # [ (batch, 1), (batch, nQueries) for _ in bookies]
        loss_sum, _ = model.loss(y, outputs)    # (), (batch,)

    grads = tape.gradient(loss_sum, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    del tape    # new
    return loss_sum  # (), (batch,)

@tf.function
def find_loss_for_step(model, x, y):
    outputs = model(x, training=False)  # [ (batch, 1), (batch, nQueries) for _ in bookies]
    loss_sum, backtest_profit_sum = model.loss(y, outputs)    # (), (batch,)    
    return loss_sum, backtest_profit_sum

def find_loss_for_dataset(model, ds_batches):
    prev_nSum = new_nSum = 0
    the_loss = tf.Variable(0.0, dtype=tf.float32, trainable=False) 
    the_backtest_profit = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    for step, ((baseId, sequence, base_bb, baseDateDetails, mask), (value)) in enumerate(ds_batches):
        x = (sequence, base_bb, baseDateDetails, mask); y = value
        loss_sum, backtest_profit_sum = find_loss_for_step(model, x, y)
        batch_size = baseId.shape[0]
        new_nSum += batch_size  # += baseId.shape[0], given sum of loss and profit. Ignore that the last batch might be smaller than others.
        the_loss = the_loss * prev_nSum / new_nSum + loss_sum / new_nSum
        the_backtest_profit = the_backtest_profit * prev_nSum / new_nSum + backtest_profit_sum / new_nSum
        prev_nSum = new_nSum
    return the_loss, the_backtest_profit    # agerage loss and backtest_profit per game

#=====================================================  