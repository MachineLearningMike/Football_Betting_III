import numpy as np
# import tensorflow as tf       # I suspect that tf, or graph, used in an external .py file behaves as if there were numberical errors, degrading the performance.
import pandas as pd
import datetime
import pickle
import json
import re
import os
import copy
import networkx as nx

from config import config

#=========================================================== basic =======================================================

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

#========================================================= get_train_and_new_from_football_data  =============================================================


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

def find_1X2_odds_group(columns):
        oGroups = []
        i = 0
        while i <= len(columns)-3:
                if columns[i].endswith('H') and columns[i+1].endswith('D') and columns[i+2].endswith('A') \
                and columns[i][:-1] == columns[i+1][:-1] and columns[i+1][:-1] == columns[i+2][:-1]:
                        oGroups.append(columns[i][:-1]); i += 3
                else: i += 1
        return oGroups

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


def get_train_and_new_from_football_data(gameHistoryFolderPath, countryThemeFolder, Non_Odds_cols, num_bookies, oddsGroupsToExclude, preferedOrder, train_mode, skip=False):

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

#======================================================== tokenizer =======================================================

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

#======================================================= fixture_id_to_ids_uk_v3  =============================================================


def fixture_id_to_ids_uk_v3(folder, id_to_ids_filename, targetLength, df_total, year_span, testcount=-1):

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

        #------------------------- main --------------------------------

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

#==================================================== get_normalization_params  =========================================================

def get_normalization_params(df, cols):
    def get_mean_and_std(col):
        array = df[col].dropna()
        array = np.array(array)
        return (array.mean(), array.std(), np.max(array))
    params = {}
    for col in cols:
        params[col] = get_mean_and_std(col)
    return params

#=========================================================  history_class  ==========================================================

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
    def show(self, ax, TEST_ID, show_val_0=True):
        ax.set_title(TEST_ID + ": loss history")

        ax.plot(self.history['loss'], label='train_loss')
        if show_val_0: ax.plot(self.history['val_loss_0'], label='GET_VAL_LOSS_0')
        ax.plot(self.history['val_loss'], label='val_loss')
        ax.plot([-v for v in self.history['val_loss']], label='vector_profit')
        ax.plot(self.history['val_profit'], label='scalar_profit')
        
        ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_ylabel('loss or profit')
        ax.set_xlabel('epoch', loc='right')


#========================================================    =========================================================

