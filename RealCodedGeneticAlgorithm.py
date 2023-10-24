import numpy as np
import os, shutil
import pandas as pd

class RealCodecGA_JGG_AREX:
    '''
    
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）

    Note:
        集団サイズ population は問題に応じて 5n ~ 50n を目安に設定
        
        ［参考］
            https://st1990.hatenablog.com/entry/2019/04/21/212326
            https://github.com/statsu1990/Real-coded-genetic-algorithm

    '''
    def __init__(self, gene_num, evaluation_func, population, seed=None, result_path='result'):
        # 乱数シード
        np.random.seed(seed=seed)

        # 結果削除
        self._initialize_result(result_path)

        # メンバの初期化
        self._initialize_attributes(gene_num, evaluation_func, population)

        # 集団の初期化
        self._initialize_genes()

        # 最初の評価
        self._initial_evaluation()

    def _initialize_result(self, result_path):
        self.result_path = result_path
        if os.path.isdir(self.result_path): shutil.rmtree(self.result_path)
        os.makedirs(self.result_path)
        os.makedirs(f'{self.result_path}/population')
        os.makedirs(f'{self.result_path}/elite_img')

    def _initialize_attributes(self, gene_num, evaluation_func, population):
        # 遺伝子サイズ
        self.gene_num = gene_num

        # 評価関数
        self.evaluation_func = evaluation_func

        #遺伝子の初期値範囲
        self.initial_min = -1.0
        self.initial_max = 1.0

        # 集団サイズ
        assert population is not None
        self.population = population

        # 親個体の数
        self.parent_num = gene_num+1

        # 子個体の数
        self.child_num = 4*self.gene_num

        # 拡張率
        self.expantion_rate = 1.0

        # 拡張率の学習率
        self.learning_rate = 1.0/(5.0*gene_num)

        self.genes = None               # 集団
        self.evals = None               # 集団の評価値
        self.best_gene = None           # エリート個体
        self.best_evaluation = None     # エリート個体の評価値
        self.last_gene = None           # 最終時点での遺伝子
        self.generation = 0             # 世代数

    def _initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)

    def _initial_evaluation(self):
        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx].copy()

    def generation_step(self):
        self.generation += 1

        # 個体数
        pop = len(self.genes)

        # 交叉する個体の選択
        # 交叉する個体のインデックス
        crossed_idx = self._random_select(pop, self.parent_num)

        # 交叉
        child_genes, rand_mtrx = self._arex_crossover(self.genes[crossed_idx], self.evals[crossed_idx], self.child_num, self.expantion_rate)

        # 残す個体の選択
        survive_genes, survive_evals, survive_idx = self._ranking_survival(child_genes, self.evaluation_func, survive_num=self.parent_num)
        # 親個体と変更
        self.genes[crossed_idx] = survive_genes.copy()
        self.evals[crossed_idx] = survive_evals

        # 拡張率の更新
        self.expantion_rate = self._update_arex_expantion_rate(self.expantion_rate, self.learning_rate, rand_mtrx[survive_idx], crss_num=self.parent_num)
        #print(self.expantion_rate)

        min_idx = np.argmin(survive_evals)
        # 生き残ったうちの最高評価
        best_survive_evals = survive_evals[min_idx]
        # 最高評価の更新
        if self.best_evaluation > survive_evals[min_idx]:
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx].copy()
        
        return best_survive_evals

    def save_result(self, save_population=False):
        if save_population:
            pd.DataFrame(self.genes).to_csv(
                f'{self.result_path}/population/{self.generation}.csv', index=None
            )
        with open(f'{self.result_path}/obj_func.csv', 'a') as f: f.write(f'{self.best_evaluation}\n')
        with open(f'{self.result_path}/elite.csv', 'a') as f: f.write(f'{",".join(str(i) for i in self.best_gene)}\n')


    @staticmethod
    def _random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def _arex_crossover(genes, evals, child_num, expantion_rate):
        # 交叉される遺伝子数
        crss_num = len(genes)

        # 評価が高い順に並べたインデックス
        sorted_idx = np.argsort(evals)

        # 荷重重心
        w = 2.0 * (crss_num + 1.0 - np.arange(1, crss_num+1)) / (crss_num * (crss_num + 1.0))
        wG = np.dot(w[np.newaxis,:], genes[sorted_idx])

        # 重心
        G = np.average(genes, axis=0)

        # 乱数
        rnd_mtrx = np.random.normal(loc=0.0, scale=np.sqrt(1/(crss_num-1)), size=(child_num, crss_num))
        
        # 子個体
        child_genes = wG + expantion_rate * np.dot(rnd_mtrx, genes - G)

        return child_genes, rnd_mtrx

    @staticmethod
    def _update_arex_expantion_rate(expantion_rate, learning_rate, survive_rand_mtrx, crss_num=None):
        survive_num = len(survive_rand_mtrx)
        #
        crss_num_ = crss_num if crss_num is not None else survive_num
        #
        ave_r = np.average(survive_rand_mtrx, axis=0)
        L_cdp = expantion_rate**2 * (survive_num - 1) * (np.sum(np.square(ave_r)) - (np.sum(ave_r))**2 / survive_num)
        L_ave = expantion_rate**2 * (1/(crss_num_ - 1)) * (survive_num - 1)**2 / survive_num
        #
        new_expantion_rate = np.maximum(1.0, expantion_rate * np.sqrt((1.0 - learning_rate) + learning_rate * L_cdp / L_ave))

        return new_expantion_rate

    @staticmethod
    def _ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        
        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]
        return survive_genes, survive_eval, survive_idx
