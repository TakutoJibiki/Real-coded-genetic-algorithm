import numpy as np
import os, shutil
import pandas as pd

class RealCodecGA_JGG_AREX:
    '''
    実数値遺伝的アルゴリズム（生存選択モデルJGG、交叉SPX）

    初期化
    交叉する個体の選択
    交叉
    残す個体の選択
    '''
    def __init__(self, gene_num, 
                 evaluation_func, 
                 initial_min, initial_max, 
                 population=None, 
                 crossover_num=None, child_num=None, 
                 initial_expantion_rate=None, learning_rate=None, 
                 seed=None):
        '''
        recommended value of parameter are following.
         crossover_num = gene_num + 1
         child_num = 4 * gene_num (~4 * crossover_num)
         initial_expantion_rate = 1.0
         learning_rate = 1/5/gene_num
        '''
        # 結果削除
        if os.path.isdir('result'): shutil.rmtree('result')

        #乱数シード
        np.random.seed(seed=seed)

        # 遺伝子情報
        self.gene_num = gene_num

        # 個体の評価関数
        # evaluation function defined as following.
        #  func(genes):
        #    return evaluations
        self.evaluation_func = evaluation_func

        #遺伝子の初期値範囲
        self.initial_min = initial_min
        self.initial_max = initial_max

        # 全個体数
        # 交叉する個体数
        if (population is not None) and (crossover_num is not None):
            self.population = population
            self.crossed_num = crossover_num
        elif (population is None) and (crossover_num is not None):
            self.crossed_num = crossover_num
            self.population = self.crossed_num
        elif (population is not None) and (crossover_num is None):
            self.population = population
            self.crossed_num = np.minimum(gene_num+1, population)
        elif (population is None) and (crossover_num is None):
            self.crossed_num = gene_num+1
            self.population = self.crossed_num

        # 交叉して生成する子個体数
        self.child_num = child_num if child_num is not None else 4*self.crossed_num

        # 拡張率
        self.expantion_rate = initial_expantion_rate if initial_expantion_rate is not None else 1.0
        # 拡張率の学習率
        self.learning_rate = learning_rate if learning_rate is not None else 1.0/5.0 / gene_num

        #
        self.genes = None # 候補の遺伝子
        self.evals = None # 候補の遺伝子の評価値
        self.best_gene = None # 現時点で評価値が一番高い遺伝子
        self.best_evaluation = None # 現時点で一番高い評価値
        self.last_gene = None # 最終時点での遺伝子
        self.generation = 0 # 世代数
        
        #
        self.__initialize_genes()

        return

    def __initialize_genes(self):
        # 遺伝子の初期値設定
        # genes = [[gene_0], [gene_1], ... ,[gene_population]]
        self.genes = self.initial_min + (self.initial_max - self.initial_min) * np.random.rand(self.population, self.gene_num)

        # 遺伝子の評価値
        self.evals = self.evaluation_func(self.genes)
        
        # min
        min_idx = np.argmin(self.evals)
        self.best_evaluation = self.evals[min_idx]
        self.best_gene = self.genes[min_idx].copy()

        return

    def generation_step(self):
        self.generation += 1

        # 個体数
        pop = len(self.genes)

        # 交叉する個体の選択
        # 交叉する個体のインデックス
        crossed_idx = self.__random_select(pop, self.crossed_num)

        # 交叉
        child_genes, rand_mtrx = self.__arex_crossover(self.genes[crossed_idx], self.evals[crossed_idx], self.child_num, self.expantion_rate)

        # 残す個体の選択
        survive_genes, survive_evals, survive_idx = self.__ranking_survival(child_genes, self.evaluation_func, survive_num=self.crossed_num)
        # 親個体と変更
        self.genes[crossed_idx] = survive_genes.copy()
        self.evals[crossed_idx] = survive_evals

        # 拡張率の更新
        self.expantion_rate = self.__update_arex_expantion_rate(self.expantion_rate, self.learning_rate, rand_mtrx[survive_idx], crss_num=self.crossed_num)
        #print(self.expantion_rate)

        min_idx = np.argmin(survive_evals)
        # 最終遺伝子の更新
        self.last_gene = survive_genes[min_idx].copy()
        # 生き残ったうちの最高評価
        best_survive_evals = survive_evals[min_idx]
        # 最高評価の更新
        if self.best_evaluation > survive_evals[min_idx]:
            self.best_evaluation = survive_evals[min_idx]
            self.best_gene = survive_genes[min_idx].copy()
        
        return best_survive_evals

    def calc_diversity(self):
        dvs = np.average(np.std(self.genes, axis=0))
        return dvs
    
    def save_result(self):
        if not os.path.isdir('result'): os.makedirs('result')
        if not os.path.isdir('result/population'): os.makedirs('result/population')
        if not os.path.isdir('result/elite_img'): os.makedirs('result/elite_img')
        pd.DataFrame(self.genes).to_csv(
            f'result/population/{self.generation}.csv', index=None
        )
        with open('result/obj_func.csv', 'a') as f: f.write(f'{self.best_evaluation}\n')
        with open('result/elite.csv', 'a') as f: f.write(f'{",".join(str(i) for i in self.best_gene)}\n')


    @staticmethod
    def __random_select(population, select_num):
        selected_idx = np.random.choice(np.arange(population), select_num, replace=False)
        return selected_idx

    @staticmethod
    def __arex_crossover(genes, evals, child_num, expantion_rate):
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
    def __update_arex_expantion_rate(expantion_rate, learning_rate, survive_rand_mtrx, crss_num=None):
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
    def __ranking_survival(genes, evaluation_func, survive_num):
        evals = evaluation_func(genes)
        survive_idx = np.argpartition(evals, survive_num)[:survive_num]
        #
        survive_genes = genes[survive_idx].copy()
        survive_eval = evals[survive_idx]
        return survive_genes, survive_eval, survive_idx


if __name__ == '__main__':
    #test()
    import verification
    
    #verification.test_Minimize_L2_1()
    verification.test_LinearLeastSquares_withRCGA()
    
