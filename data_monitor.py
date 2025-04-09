
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

import data_helpers

class history_class():
    def round_sig(self, x, sig=2):
            return x
            # return round(x, sig-int(math.floor(math.log10(abs(x))))-1)    # domain error for VERY small numbers.
    def __init__(self, filepath):
        self.filepath = filepath
        self.history = {'loss': [], 'val_loss': []}
    def removeFile(self):
        files = glob.glob(self.filepath + "*")   # "*.*" may not work
        result = [os.remove(file) for file in files]
    def save(self):
        data_helpers.SaveJsonData(self.history, self.filepath)
    def reset(self):
        self.removeFile()
        self.save()
    def load(self):
        history = data_helpers.LoadJsonData(self.filepath)
        if history is not None:
            self.history = history
    def append(self, loss, val_loss):
        self.history['loss'].append(self.round_sig(float(loss), 4))
        self.history['val_loss'].append(self.round_sig(float(val_loss), 4))
        self.save()
    def len(self):
        assert len(self.history['loss']) == len(self.history['val_loss'])
        return len(self.history['loss'])
    def get_latest_item(self):
        return (self.history['loss'][-1], self.history['val_loss'][-1])
    def get_min_val_loss(self):
        return float('inf') if self.len() <= 0 else min(self.history['val_loss'])

    def show(self, ax):
        ax.set_title(TEST_ID + ": loss history")
        ax.plot(self.history['loss'])
        ax.plot(self.history['val_loss'])
        ax.grid(True)
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.legend(['train_loss', 'val_loss'], loc='upper left')

class test_class():
    def round_sig(self, x, sig=2):
            # return x
            return round(x, sig-int(math.floor(math.log10(abs(x))))-1)    # domain error for VERY small numbers.
    def __init__(self, profit_keys, valid_size, nBookies, filepath):
        self.profit_keys = profit_keys
        self.nTests = valid_size * nBookies
        self.filepath = filepath
        self.profits = {str(key): [] for key in self.profit_keys}
        self.casts = {str(key): [] for key in self.profit_keys}
    def removeFile(self):
        files = glob.glob(self.filepath + "*")   # "*.*" may not work
        result = [os.remove(file) for file in files]
    def save(self):
        data_helpers.SaveJsonData([self.profits, self.casts], self.filepath)
    def reset(self):
        self.removeFile()
        self.save()
    def load(self):
        test = data_helpers.LoadJsonData(self.filepath)
        if test is not None:
            [self.profits, self.casts] = test
    def getLen(self, dict):
        length = None
        try:
            for key, value in dict.items():
                if length is None:
                    length = len(value)
                else:
                    assert len(value) == length
            return length
        except:
            raise Exception("Un-uniform length in  distribution")      

    def len(self):
        assert len(self.profits) == len(self.profit_keys)
        assert len(self.casts) == len(self.profit_keys)
        length = self.getLen(self.profits)
        assert self.getLen(self.casts) == length
        return length

    def append(self, profits, casts):
        length = self.len()     # for all the asserts.
        assert len(profits) == len(self.profit_keys)
        assert len(casts) == len(self.profit_keys)
        for item_list, item in zip(self.profits.values(), profits):
            item_list.append(item)

        assert len(casts) == len(self.profit_keys)
        for item_list, item in zip(self.casts.values(), casts):
            item_list.append(item)

        self.save()

    def get_best_product(self, profits, casts):
        best_product = -float('inf') # MIN_PROFIT * 1e6
        for (p, n) in zip(profits, casts):
            if p * n > best_product:
                best_product = p * n
        return best_product
    
    def get_existing_best_product(self):
        all_profits = []
        for item_list in self.profits.values():
            all_profits += item_list
        all_casts = []
        for item_list in self.casts.values():
            all_casts += item_list
        return self.get_best_product(all_profits, all_casts)
    
    def find_profit_cast_series(self):
        nSeries = len(tuple(self.profits.values())[0])
        for v in self.profits.values():
            assert len(v) == nSeries
        for v in self.casts.values():
            assert len(v) == nSeries
        
        profit_series =  [[v[serial] for v in self.profits.values()] for serial in range(nSeries)] # [ [ profit for _ in profit_keys ] ] * nSeries
        cast_series =  [[v[serial] for v in self.casts.values()] for serial in range(nSeries)] # [ [ cast for _ in profit_keys ] ] * nSeries
        return profit_series, cast_series
       
    def find_total_profit_groups(self):       
        profit_series, cast_series = self.find_profit_cast_series()
        total_profit_groups = []
        for profits, casts in zip(profit_series, cast_series):
            profit_groups = self.find_profit_groups(profits, casts, sort=False)
            total_profit_groups.append(profit_groups)   # [ [ (product, profit, cast, id_string) for nGroups ] ] * nSeries
        return total_profit_groups
    
    def track_profit_groups(self, total_profit_groups):
        # total_profit_groups: [ [ (product, profit, cast, id_string) for nGroups ] ] * nSeries
        profit_groups_track = None
        nSeries = len(total_profit_groups)
        if nSeries > 0:
            nGroups = len(total_profit_groups[0])
            for profit_groups in total_profit_groups:
                assert len(profit_groups) == nGroups
            profit_groups_track = [[total_profit_groups[series][profit_group] for series in range(nSeries)] for profit_group in range(nGroups)]
        return profit_groups_track  # [ [ (product, profit, cast, id) for _ in range(nSeries)] for _ in range(nGroups) ]
        # profit_groups_track = { profit_groups_track[group][0][3] : [(profit, cast) for _, profit, cast, _ in profit_groups_track[group]] for group in range(nGroups) }
        # return profit_groups_track  # { id : [ (profit, cast) for _ in range(nSeries)] for _ in range(nGroups) }
    
    def find_profit_groups(self, profits, casts, sort=True):
        result = []
        for n1 in range(len(self.profit_keys)):
            for n2 in range(n1, len(self.profit_keys)): 
                # n2 >= n1. profit_keys[n2] >= profit_keys[n1], casts[n2] <= casts[n1]
                if n1 == n2:
                    result.append((profits[n1] * casts[n1], profits[n1], casts[n1], str(self.profit_keys[n1])+"-"))
                else:
                    cast3 = casts[n1] - casts[n2]
                    if cast3 > 0:
                        profit3 = (profits[n1] * casts[n1] - profits[n2] * casts[n2]) / cast3
                    else:
                        profit3 = MIN_PROFIT
                    result.append( (profit3 * cast3, profit3, cast3, str(self.profit_keys[n1])+"-"+str(self.profit_keys[n2])))
        if sort: result.sort(reverse=True)
        return result
    
    def find_profit_groups_elements(self, profits, casts, sort=True):
        result = []
        for n1 in range(len(self.profit_keys)-1):
            n2 = n1 + 1
            cast3 = casts[n1] - casts[n2]
            if cast3 > 0:
                profit3 = (profits[n1] * casts[n1] - profits[n2] * casts[n2]) / cast3
            else:
                profit3 = MIN_PROFIT
            result.append( (profit3 * cast3, profit3, cast3, str(self.profit_keys[n1])+"-"+str(self.profit_keys[n2])))
        if sort: result.sort(reverse=True)
        return result
    
    def print_profit_groups(self, groups, count):
        # groups: [ (product, profit, cast, interval) ] * n
        for (product, profit, cast, interval) in groups:
            print("[{:.5f}, {:.4f}, {}, {}]".format(product, profit, cast, interval), end=', ')
            count -= 1
            if count <= 0:
                print(); break
            
    # def show_profit_distribution(self):


    def show_profit_groups(self, minProduct=0.0):
        total_profit_groups = self.find_total_profit_groups()   # [ [ (product, profit, cast, group_id) for nGroups ] ] * nSeries
        if len(total_profit_groups) < 1: return

        profit_groups_track = self.track_profit_groups(total_profit_groups) # [ [ (product, profit, cast, group_id) for _ in range(nSeries)] for _ in range(nGroups) ]

        nGroups = len(total_profit_groups[0])
        for profit_groups in total_profit_groups:
            assert len(profit_groups) == nGroups
        profit_groups_track = { profit_groups_track[group][0][3] : [(profit, cast) for _, profit, cast, _ in profit_groups_track[group]] for group in range(nGroups) }
        # { group_id : [ (profit, cast) for _ in range(nSeries)] for _ in range(nGroups) }
        
        minCasts = self.nTests; maxCasts = 0
        minProfit = 50.0; maxProfit = MIN_PROFIT
        for key, value in profit_groups_track.items():
            casts = [cast for _, cast in value]
            profits = [profit for profit, _ in value]
            # if profits[-1] * casts[-1] > minProduct:
            if key.endswith('-'):
                if minCasts > min(casts): minCasts = min(casts)
                if maxCasts < max(casts): maxCasts = max(casts)
                if minProfit > min(profits): minProfit = min(profits)
                if maxProfit < max(profits): maxProfit = max(profits)

        step = 5; x = np.arange(minCasts, maxCasts + step, step).reshape(-1, 1)
        step = 0.0005; y = np.arange(minProfit, maxProfit + step, step).reshape(-1, 1)
        X, Y = np.meshgrid(x, y)    # (n, m)
        XY = np.stack((X, Y), axis=-1)  # (n, m, 2)
        Z = XY[:, :, 0] * XY[:, :, 1]   # (n, m)

        sLevels = (0) #, 1, 2, 3, 4, 5) if GUI.loss == 'mean_squared_error' else (0,)
        sColors = ['r'] # GUI.colors[: len(sLevels)]
        nContours = 80
        plt.figure(figsize=(12,8))
        CS0 = plt.contourf(X, Y, Z, nContours, cmap=plt.cm.bone, origin='lower')
        CS = plt.contour(X, Y, Z, CS0.levels, colors=('k'), origin='lower', linewidths=.2)
        plt.contour(X, Y, Z, sLevels, colors=sColors, origin='lower', linewidths=.5)    
        plt.clabel(CS, fmt='%1.1f', colors='c', fontsize=8, inline=True)

        for key, value in profit_groups_track.items():
            casts = [cast for _, cast in value]
            profits = [profit for profit, _ in value]
            # if profits[-1] * casts[-1] > minProduct:
            if key.endswith('-'):
                plt.plot(casts, profits, label=key, marker='o', lw=0.5)
                plt.plot(casts[-1], profits[-1], marker='o', color='k')
                plt.plot(casts[-1], profits[-1], marker='x', color='w')
        plt.legend(loc='best')
        plt.grid(True)

        plt.show()

    def show(self, ax):
        colors = ['black', 'firebrick', 'darkgreen', 'c', 'blue', 'blueviolet', 'magenta', 'maroon', "yellowgreen", 'cadetblue', 'purple', 'c', 'blue']

        gmin = MIN_PROFIT - 1.0; gmax = MIN_PROFIT
        all_profits = []
        for item_list in self.profits.values():
            all_profits += item_list
        if len(all_profits) > 0:
            gmin = min(all_profits); gmax = max(all_profits)

        _min = 0.0; _max = self.nTests        
        # _min = 0.0; _max = 1.0
        # all_nBettings = []
        # for item_list in self.nBettings.values():
        #     all_nBettings += item_list
        # if len(all_nBettings) > 0:
        #     _min = min(all_nBettings); _max = max(all_nBettings)
        
        legends = []
        for item_list, color, key in zip(self.profits.values(), colors[:len(self.profit_keys)], self.profit_keys):
            # print(item_list, color, key)
            ax.plot(item_list, color=color, linewidth=0.7)
            legends.append("> " + str(key))
        # print(legends)

        for item_list, color in zip(self.casts.values(), colors[:len(self.profit_keys)]):
            item_list = [ (item-_min)/(_max-_min+1e-9) * (gmax-gmin) + gmin for item in item_list]
            ax.plot(item_list, color=color, linestyle='--', linewidth=0.7)

        ax.legend(legends, loc='upper left')
        ax.grid(True)
        ax.set_title(TEST_ID + ": avg_profit and scaled nBettings per profit threshold key. max: {}".format(gmax))
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')