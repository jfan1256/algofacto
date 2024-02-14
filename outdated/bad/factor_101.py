import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import talib


class Factor101:
    # Constructor Method
    def __init__(self, data):
        self.data = data
        self.o = self.data.Open.unstack('ticker')
        self.h = self.data.High.unstack('ticker')
        self.l = self.data.Low.unstack('ticker')
        self.c = self.data.Close.unstack('ticker')
        self.v = self.data.Volume.unstack('ticker')
        self.vwap = self.o.add(self.h).add(self.l).add(self.c).div(4)
        self.adv20 = self.v.rolling(20).mean()
        self.r = pd.DataFrame()

    # Functions
    @staticmethod
    def createOneDayReturn(df):
        by_ticker = df.groupby(level='ticker')
        df[f'RET_01'] = by_ticker.Close.pct_change(1)

    @staticmethod
    def rank(df):
        return df.rank(axis=1, pct=True)

    @staticmethod
    def scale(df):
        return df.div(df.abs().sum(axis=1), axis=0)

    @staticmethod
    def log(df):
        return np.log1p(df)

    @staticmethod
    def sign(df):
        return np.sign(df)

    @staticmethod
    def power(df, exp):
        return df.pow(exp)

    @staticmethod
    def ts_lag(df: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        return df.shift(t)

    @staticmethod
    def ts_delta(df, period=1):
        return df.diff(period)

    @staticmethod
    def ts_sum(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        return df.rolling(window).sum()

    @staticmethod
    def ts_mean(df, window):
        return df.rolling(window).mean()

    @staticmethod
    def ts_weighted_mean(df, period=10):
        return (df.apply(lambda x: talib.WMA(x, timeperiod=period)))

    @staticmethod
    def ts_std(df, window):
        return (df.rolling(window).std())

    @staticmethod
    def ts_rank(df, window):
        return (df.rolling(window).apply(lambda x: x.rank().iloc[-1]))

    @staticmethod
    def ts_product(df, window):
        return (df.rolling(window).apply(np.prod))

    @staticmethod
    def ts_min(df, window):
        return df.rolling(window).min()

    @staticmethod
    def ts_max(df, window):
        return df.rolling(window).max()

    @staticmethod
    def ts_argmax(df, window):
        return df.rolling(window).apply(np.argmax).add(1)

    @staticmethod
    def ts_argmin(df, window):
        return (df.rolling(window).apply(np.argmin).add(1))

    @staticmethod
    def ts_corr(x, y, window):
        return x.rolling(window).corr(y)

    @staticmethod
    def ts_cov(x, y, window):
        return x.rolling(window).cov(y)

    def copyToNegateReadOnly(self):
        self.o = self.o.copy()
        self.h = self.h.copy()
        self.l = self.h.copy()
        self.c = self.c.copy()
        self.v = self.v.copy()
        self.vwap = self.vwap.copy()
        self.adv20 = self.adv20.copy()
        self.r = self.r.copy()

    # -----------------------------------------------------101 Factors------------------------------------------------------------
    def alphaCreateOneDayReturn(self):
        self.copyToNegateReadOnly()
        self.createOneDayReturn(self.data)
        self.r = self.data.RET_01.unstack('ticker')

    def alpha001(self):
        self.copyToNegateReadOnly()
        self.c[self.r < 0] = self.ts_std(self.r, 20)
        self.data['001'] = (self.rank(self.ts_argmax(self.power(self.c, 2), 5)).mul(-.5).stack().swaplevel())

    def alpha002(self):
        self.copyToNegateReadOnly()
        s1 = self.rank(self.ts_delta(self.log(self.v), 2))
        s2 = self.rank((self.c / self.o) - 1)
        alpha = -self.ts_corr(s1, s2, 6)
        self.data['002'] = alpha.stack('ticker').swaplevel().replace([-np.inf, np.inf], np.nan)

    def alpha003(self):
        self.copyToNegateReadOnly()
        self.data['003'] = (-self.ts_corr(self.rank(self.o), self.rank(self.v), 10).stack('ticker').swaplevel().replace(
            [-np.inf, np.inf], np.nan))

    def alpha004(self):
        self.copyToNegateReadOnly()
        self.data['004'] = (-self.ts_rank(self.rank(self.l), 9).stack('ticker').swaplevel())

    def alpha005(self):
        self.copyToNegateReadOnly()
        self.data['005'] = (self.rank(self.o.sub(self.ts_mean(self.vwap, 10))).mul(
            self.rank(self.c.sub(self.vwap)).mul(-1).abs()).stack('ticker').swaplevel())

    def alpha006(self):
        self.copyToNegateReadOnly()
        self.data['006'] = (-self.ts_corr(self.o, self.v, 10).stack('ticker').swaplevel())

    def alpha007(self):
        self.copyToNegateReadOnly()
        delta7 = self.ts_delta(self.c, 7)
        self.data['007'] = (-self.ts_rank(abs(delta7), 60).mul(self.sign(delta7)).where(self.adv20 < self.v, -1).stack(
            'ticker').swaplevel())

    def alpha008(self):
        self.copyToNegateReadOnly()
        self.data['008'] = (-(self.rank(((self.ts_sum(self.o, 5) * self.ts_sum(self.r, 5)) - self.ts_lag(
            (self.ts_sum(self.o, 5) * self.ts_sum(self.r, 5)), 10)))).stack('ticker').swaplevel())

    def alpha009(self):
        self.copyToNegateReadOnly()
        close_diff = self.ts_delta(self.c, 1)
        alpha = close_diff.where(self.ts_min(close_diff, 5) > 0,
                                 close_diff.where(self.ts_max(close_diff, 5) < 0, -close_diff))
        self.data['009'] = (alpha.stack('ticker').swaplevel())

    def alpha010(self):
        self.copyToNegateReadOnly()
        close_diff = self.ts_delta(self.c, 1)
        alpha = close_diff.where(self.ts_min(close_diff, 4) > 0,
                                 close_diff.where(self.ts_min(close_diff, 4) > 0, -close_diff))
        self.data['010'] = (alpha.stack('ticker').swaplevel())

    def alpha011(self):
        self.copyToNegateReadOnly()
        self.data['011'] = (
            self.rank(self.ts_max(self.vwap.sub(self.c), 3)).add(self.rank(self.ts_min(self.vwap.sub(self.c), 3))).mul(
                self.rank(self.ts_delta(self.v, 3))).stack('ticker').swaplevel())

    def alpha012(self):
        self.copyToNegateReadOnly()
        self.data['012'] = (
            self.sign(self.ts_delta(self.v, 1)).mul(-self.ts_delta(self.c, 1)).stack('ticker').swaplevel())

    def alpha013(self):
        self.copyToNegateReadOnly()
        self.data['013'] = (
            -self.rank(self.ts_cov(self.rank(self.c), self.rank(self.v), 5)).stack('ticker').swaplevel())

    def alpha014(self):
        self.copyToNegateReadOnly()
        alpha = -self.rank(self.ts_delta(self.r, 3)).mul(
            self.ts_corr(self.o, self.v, 10).replace([-np.inf, np.inf], np.nan))
        self.data['014'] = (alpha.stack('ticker').swaplevel())

    def alpha015(self):
        self.copyToNegateReadOnly()
        alpha = (-self.ts_sum(
            self.rank(self.ts_corr(self.rank(self.h), self.rank(self.v), 3).replace([-np.inf, np.inf], np.nan)), 3))
        self.data['015'] = (alpha.stack('ticker').swaplevel())

    def alpha016(self):
        self.copyToNegateReadOnly()
        self.data['016'] = (
            -self.rank(self.ts_cov(self.rank(self.h), self.rank(self.v), 5)).stack('ticker').swaplevel())

    def alpha017(self):
        self.copyToNegateReadOnly()
        adv20 = self.ts_mean(self.v, 20)
        self.data['017'] = (
            -self.rank(self.ts_rank(self.c, 10)).mul(self.rank(self.ts_delta(self.ts_delta(self.c, 1), 1))).mul(
                self.rank(self.ts_rank(self.v.div(adv20), 5))).stack('ticker').swaplevel())

    def alpha018(self):
        self.copyToNegateReadOnly()
        self.data['018'] = (-self.rank(self.ts_std(self.c.sub(self.o).abs(), 5).add(self.c.sub(self.o)).add(
            self.ts_corr(self.c, self.o, 10).replace([-np.inf, np.inf], np.nan))).stack('ticker').swaplevel())

    def alpha019(self):
        self.copyToNegateReadOnly()
        self.data['019'] = (-self.sign(self.ts_delta(self.c, 7) + self.ts_delta(self.c, 7)).mul(
            1 + self.rank(1 + self.ts_sum(self.r, 250))).stack('ticker').swaplevel())

    def alpha020(self):
        self.copyToNegateReadOnly()
        self.data['020'] = (
            self.rank(self.o - self.ts_lag(self.h, 1)).mul(self.rank(self.o - self.ts_lag(self.c, 1))).mul(
                self.rank(self.o - self.ts_lag(self.l, 1))).mul(-1).stack('ticker').swaplevel())

    def alpha021(self):
        self.copyToNegateReadOnly()
        sma2 = self.ts_mean(self.c, 2)
        sma8 = self.ts_mean(self.c, 8)
        std8 = self.ts_std(self.c, 8)
        cond_1 = sma8.add(std8) < sma2
        cond_2 = sma8.add(std8) > sma2
        cond_3 = self.v.div(self.ts_mean(self.v, 20)) < 1
        val = np.ones_like(self.c)
        alpha = pd.DataFrame(np.select(condlist=[cond_1, cond_2, cond_3], choicelist=[-1, 1, -1], default=1),
                             index=self.c.index, columns=self.c.columns)
        self.data['021'] = (alpha.stack('ticker').swaplevel())

    def alpha022(self):
        self.copyToNegateReadOnly()
        self.data['022'] = (self.ts_delta(self.ts_corr(self.h, self.v, 5).replace([-np.inf, np.inf], np.nan), 5).mul(
            self.rank(self.ts_std(self.c, 20))).mul(-1).stack('ticker').swaplevel())

    def alpha023(self):
        self.copyToNegateReadOnly()
        self.data['023'] = (
            self.ts_delta(self.h, 2).mul(-1).where(self.ts_mean(self.h, 20) < self.h, 0).stack('ticker').swaplevel())

    def alpha024(self):
        self.copyToNegateReadOnly()
        cond = self.ts_delta(self.ts_mean(self.c, 100), 100) / self.ts_lag(self.c, 100) <= 0.05
        self.data['024'] = (self.c.sub(self.ts_min(self.c, 100)).mul(-1).where(cond, -self.ts_delta(self.c, 3)).stack(
            'ticker').swaplevel())

    def alpha025(self):
        self.copyToNegateReadOnly()
        self.data['025'] = (
            self.rank(-self.r.mul(self.adv20).mul(self.vwap).mul(self.h.sub(self.c))).stack('ticker').swaplevel())

    def alpha026(self):
        self.copyToNegateReadOnly()
        self.data['026'] = (self.ts_max(
            self.ts_corr(self.ts_rank(self.v, 5), self.ts_rank(self.h, 5), 5).replace([-np.inf, np.inf], np.nan),
            3).mul(-1).stack('ticker').swaplevel())

    def alpha027(self):
        self.copyToNegateReadOnly()
        cond = self.rank(self.ts_mean(self.ts_corr(self.rank(self.v), self.rank(self.vwap), 6), 2))
        alpha = cond.notnull().astype(float)
        self.data['027'] = (alpha.where(cond <= 0.5, -alpha).stack('ticker').swaplevel())

    def alpha028(self):
        self.copyToNegateReadOnly()
        self.data['028'] = (self.scale(self.ts_corr(self.adv20, self.l, 5).replace([-np.inf, np.inf], 0).add(
            self.h.add(self.l).div(2).sub(self.c))).stack('ticker').swaplevel())

    def alpha029(self):
        self.copyToNegateReadOnly()
        self.data['029'] = (self.ts_min(self.rank(self.rank(
            self.scale(self.log(self.ts_sum(self.rank(self.rank(-self.rank(self.ts_delta((self.c - 1), 5)))), 2))))),
            5).add(self.ts_rank(self.ts_lag((-1 * self.r), 6), 5)).stack(
            'ticker').swaplevel())

    def alpha030(self):
        self.copyToNegateReadOnly()
        close_diff = self.ts_delta(self.c, 1)
        self.data['030'] = (self.rank(self.sign(close_diff).add(self.sign(self.ts_lag(close_diff, 1))).add(
            self.sign(self.ts_lag(close_diff, 2)))).mul(-1).add(1).mul(self.ts_sum(self.v, 5)).div(
            self.ts_sum(self.v, 20)).stack('ticker').swaplevel())

    def alpha031(self):
        self.copyToNegateReadOnly()
        self.data['031'] = (self.rank(
            self.rank(
                self.rank(self.ts_weighted_mean(self.rank(self.rank(self.ts_delta(self.c, 10))).mul(-1), 10)))).add(
            self.rank(self.ts_delta(self.c, 3).mul(-1))).add(
            self.sign(self.scale(self.ts_corr(self.adv20, self.l, 12).replace([-np.inf, np.inf], np.nan)))).stack(
            'ticker').swaplevel())

    def alpha032(self):
        self.copyToNegateReadOnly()
        self.data['032'] = (self.scale(self.ts_mean(self.c, 7).sub(self.c)).add(
            20 * self.scale(self.ts_corr(self.vwap, self.ts_lag(self.c, 5), 230))).stack('ticker').swaplevel())

    def alpha033(self):
        self.copyToNegateReadOnly()
        self.data['033'] = (self.rank(self.o.div(self.c).mul(-1).add(1).mul(-1)).stack('ticker').swaplevel())

    def alpha034(self):
        self.copyToNegateReadOnly()
        self.data['034'] = (self.rank(
            self.rank(self.ts_std(self.r, 2).div(self.ts_std(self.r, 5)).replace([-np.inf, np.inf], np.nan)).mul(
                -1).sub(self.rank(self.ts_delta(self.c, 1))).add(2)).stack('ticker').swaplevel())

    def alpha035(self):
        self.copyToNegateReadOnly()
        self.data['035'] = (self.ts_rank(self.v, 32).mul(1 - self.ts_rank(self.c.add(self.h).sub(self.l), 16)).mul(
            1 - self.ts_rank(self.r, 32)).stack('ticker').swaplevel())

    def alpha036(self):
        self.copyToNegateReadOnly()
        self.data['036'] = (self.rank(self.ts_corr(self.c.sub(self.o), self.ts_lag(self.v, 1), 15)).mul(2.21).add(
            self.rank(self.o.sub(self.c)).mul(.7)).add(
            self.rank(self.ts_rank(self.ts_lag(-self.r, 6), 5)).mul(0.73)).add(
            self.rank(abs(self.ts_corr(self.vwap, self.adv20, 6)))).add(
            self.rank(self.ts_mean(self.c, 200).sub(self.o).mul(self.c.sub(self.o))).mul(0.6)).stack(
            'ticker').swaplevel())

    def alpha037(self):
        self.copyToNegateReadOnly()
        self.data['037'] = (self.rank(self.ts_corr(self.ts_lag(self.o.sub(self.c), 1), self.c, 200)).add(
            self.rank(self.o.sub(self.c))).stack('ticker').swaplevel())

    def alpha038(self):
        self.copyToNegateReadOnly()
        self.data['038'] = (self.rank(self.ts_rank(self.o, 10)).mul(
            self.rank(self.c.div(self.o).replace([-np.inf, np.inf], np.nan))).mul(-1).stack('ticker').swaplevel())

    def alpha039(self):
        self.copyToNegateReadOnly()
        self.data['039'] = (self.rank(self.ts_delta(self.c, 7).mul(
            self.rank(self.ts_weighted_mean(self.v.div(self.adv20), 9)).mul(-1).add(1))).mul(-1).mul(
            self.rank(self.ts_mean(self.r, 250).add(1))).stack('ticker').swaplevel())

    def alpha040(self):
        self.copyToNegateReadOnly()
        self.data['040'] = (self.rank(self.ts_std(self.h, 10)).mul(self.ts_corr(self.h, self.v, 10)).mul(-1).stack(
            'ticker').swaplevel())

    def alpha041(self):
        self.copyToNegateReadOnly()
        self.data['041'] = (self.power(self.h.mul(self.l), 0.5).sub(self.vwap).stack('ticker').swaplevel())

    def alpha042(self):
        self.copyToNegateReadOnly()
        self.data['042'] = (
            self.rank(self.vwap.sub(self.c)).div(self.rank(self.vwap.add(self.c))).stack('ticker').swaplevel())

    def alpha043(self):
        self.copyToNegateReadOnly()
        self.data['043'] = (
            self.ts_rank(self.v.div(self.adv20), 20).mul(self.ts_rank(self.ts_delta(self.c, 7).mul(-1), 8)).stack(
                'ticker').swaplevel())

    def alpha044(self):
        self.copyToNegateReadOnly()
        self.data['044'] = (self.ts_corr(self.h, self.rank(self.v), 5).replace([-np.inf, np.inf], np.nan).mul(-1).stack(
            'ticker').swaplevel())

    def alpha045(self):
        self.copyToNegateReadOnly()
        self.data['045'] = (self.rank(self.ts_mean(self.ts_lag(self.c, 5), 20)).mul(
            self.ts_corr(self.c, self.v, 2).replace([-np.inf, np.inf], np.nan)).mul(
            self.rank(self.ts_corr(self.ts_sum(self.c, 5), self.ts_sum(self.c, 20), 2))).mul(-1).stack(
            'ticker').swaplevel())

    def alpha046(self):
        self.copyToNegateReadOnly()
        cond = self.ts_lag(self.ts_delta(self.c, 10), 10).div(10).sub(self.ts_delta(self.c, 10).div(10))
        alpha = pd.DataFrame(-np.ones_like(cond), index=self.c.index, columns=self.c.columns)
        alpha[cond.isnull()] = np.nan
        self.data['046'] = (
            cond.where(cond > 0.25, -alpha.where(cond < 0, -self.ts_delta(self.c, 1))).stack('ticker').swaplevel())

    def alpha047(self):
        self.copyToNegateReadOnly()
        self.data['047'] = (self.rank(self.c.pow(-1)).mul(self.v).div(self.adv20).mul(
            self.h.mul(self.rank(self.h.sub(self.c)).div(self.ts_mean(self.h, 5))).sub(
                self.rank(self.ts_delta(self.vwap, 5)))).stack('ticker').swaplevel())

    def alpha049(self):
        self.copyToNegateReadOnly()
        cond = (self.ts_delta(self.ts_lag(self.c, 10), 10).div(10).sub(
            self.ts_delta(self.c, 10).div(10)) >= -0.1 * self.c)
        self.data['049'] = (-self.ts_delta(self.c, 1).where(cond, 1).stack('ticker').swaplevel())

    def alpha050(self):
        self.copyToNegateReadOnly()
        self.data['050'] = (
            self.ts_max(self.rank(self.ts_corr(self.rank(self.v), self.rank(self.vwap), 5)), 5).mul(-1).stack(
                'ticker').swaplevel())

    def alpha051(self):
        self.copyToNegateReadOnly()
        cond = (self.ts_delta(self.ts_lag(self.c, 10), 10).div(10).sub(
            self.ts_delta(self.c, 10).div(10)) >= -0.05 * self.c)
        self.data['051'] = (-self.ts_delta(self.c, 1).where(cond, 1).stack('ticker').swaplevel())

    def alpha052(self):
        self.copyToNegateReadOnly()
        self.data['052'] = (self.ts_delta(self.ts_min(self.l, 5), 5).mul(
            self.rank(self.ts_sum(self.r, 240).sub(self.ts_sum(self.r, 20)).div(220))).mul(
            self.ts_rank(self.v, 5)).stack('ticker').swaplevel())

    def alpha053(self):
        self.copyToNegateReadOnly()
        inner = (self.c.sub(self.l)).add(1e-6)
        self.data['053'] = (
            self.ts_delta(self.h.sub(self.c).mul(-1).add(1).div(self.c.sub(self.l).add(1e-6)), 9).mul(-1).stack(
                'ticker').swaplevel())

    def alpha054(self):
        self.copyToNegateReadOnly()
        self.data['054'] = (self.l.sub(self.c).mul(self.o.pow(5)).mul(-1).div(
            self.l.sub(self.h).replace(0, -0.0001).mul(self.c ** 5)).stack('ticker').swaplevel())

    def alpha055(self):
        self.copyToNegateReadOnly()
        self.data['055'] = (self.ts_corr(self.rank(self.c.sub(self.ts_min(self.l, 12)).div(
            self.ts_max(self.h, 12).sub(self.ts_min(self.l, 12)).replace(0, 1e-6))), self.rank(self.v), 6).replace(
            [-np.inf, np.inf], np.nan).mul(-1).stack('ticker').swaplevel())

    def alpha057(self):
        self.copyToNegateReadOnly()
        self.data['057'] = (
            self.c.sub(self.vwap.add(1e-5)).div(self.ts_weighted_mean(self.rank(self.ts_argmax(self.c, 30)))).mul(
                -1).stack('ticker').swaplevel())

    def alpha060(self):
        self.copyToNegateReadOnly()
        self.data['060'] = (self.scale(
            self.rank(self.c.mul(2).sub(self.l).sub(self.h).div(self.h.sub(self.l).replace(0, 1e-5)).mul(self.v))).mul(
            2).sub(self.scale(self.rank(self.ts_argmax(self.c, 10)))).mul(-1).stack('ticker').swaplevel())

    def alpha061(self):
        self.copyToNegateReadOnly()
        self.data['061'] = (self.rank(self.vwap.sub(self.ts_min(self.vwap, 16))).lt(
            self.rank(self.ts_corr(self.vwap, self.ts_mean(self.v, 180), 18))).astype(int).stack('ticker').swaplevel())

    def alpha062(self):
        self.copyToNegateReadOnly()
        self.data['062'] = (self.rank(self.ts_corr(self.vwap, self.ts_sum(self.adv20, 22), 9)).lt(
            self.rank(self.rank(self.o).mul(2)).lt(self.rank(self.h.add(self.l).div(2)).add(self.rank(self.h)))).mul(
            -1).stack('ticker').swaplevel())

    def alpha064(self):
        self.copyToNegateReadOnly()
        w = 0.178404
        self.data['064'] = (self.rank(self.ts_corr(self.ts_sum(self.o.mul(w).add(self.l.mul(1 - w)), 12),
                                                   self.ts_sum(self.ts_mean(self.v, 120), 12), 16)).lt(
            self.rank(self.ts_delta(self.h.add(self.l).div(2).mul(w).add(self.vwap.mul(1 - w)), 3))).mul(-1).stack(
            'ticker').swaplevel())

    def alpha065(self):
        self.copyToNegateReadOnly()
        w = 0.00817205
        self.data['065'] = (self.rank(
            self.ts_corr(self.o.mul(w).add(self.vwap.mul(1 - w)), self.ts_mean(self.ts_mean(self.v, 60), 9), 6)).lt(
            self.rank(self.o.sub(self.ts_min(self.o, 13)))).mul(-1).stack('ticker').swaplevel())

    def alpha066(self):
        self.copyToNegateReadOnly()
        w = 0.96633
        self.data['066'] = (self.rank(self.ts_weighted_mean(self.ts_delta(self.vwap, 4), 7)).add(self.ts_rank(
            self.ts_weighted_mean(self.l.mul(w).add(self.l.mul(1 - w)).sub(self.vwap).div(
                self.o.sub(self.h.add(self.l).div(2)).add(1e-3)), 11), 7)).mul(-1).stack('ticker').swaplevel())

    def alpha068(self):
        self.copyToNegateReadOnly()
        w = 0.518371
        self.data['068'] = (
            self.ts_rank(self.ts_corr(self.rank(self.h), self.rank(self.ts_mean(self.v, 15)), 9), 14).lt(
                self.rank(self.ts_delta(self.c.mul(w).add(self.l.mul(1 - w)), 1))).mul(-1).stack('ticker').swaplevel())

    def alpha071(self):
        self.copyToNegateReadOnly()
        s1 = (self.ts_rank(self.ts_weighted_mean(
            self.ts_corr(self.ts_rank(self.c, 3), self.ts_rank(self.ts_mean(self.v, 180), 12), 18), 4), 16))
        s2 = (self.ts_rank(self.ts_weighted_mean(self.rank(self.l.add(self.o).sub(self.vwap.mul(2))).pow(2), 16), 4))
        self.data['071'] = (s1.where(s1 > s2, s2).stack('ticker').swaplevel())

    def alpha072(self):
        self.copyToNegateReadOnly()
        self.data['072'] = (self.rank(
            self.ts_weighted_mean(self.ts_corr(self.h.add(self.l).div(2), self.ts_mean(self.v, 40), 9), 10)).div(
            self.rank(
                self.ts_weighted_mean(self.ts_corr(self.ts_rank(self.vwap, 3), self.ts_rank(self.v, 18), 6), 2))).stack(
            'ticker').swaplevel())

    def alpha073(self):
        self.copyToNegateReadOnly()
        w = 0.147155
        s1 = self.rank(self.ts_weighted_mean(self.ts_delta(self.vwap, 5), 3))
        s2 = (self.ts_rank(self.ts_weighted_mean(
            self.ts_delta(self.o.mul(w).add(self.l.mul(1 - w)), 2).div(self.o.mul(w).add(self.l.mul(1 - w)).mul(-1)),
            3), 16))
        self.data['073'] = (s1.where(s1 > s2, s2).mul(-1).stack('ticker').swaplevel())

    def alpha074(self):
        self.copyToNegateReadOnly()
        w = 0.0261661
        self.data['074'] = (self.rank(self.ts_corr(self.c, self.ts_mean(self.ts_mean(self.v, 30), 37), 15)).lt(
            self.rank(self.ts_corr(self.rank(self.h.mul(w).add(self.vwap.mul(1 - w))), self.rank(self.v), 11))).mul(
            -1).stack('ticker').swaplevel())

    def alpha075(self):
        self.copyToNegateReadOnly()
        self.data['075'] = (self.rank(self.ts_corr(self.vwap, self.v, 4)).lt(
            self.rank(self.ts_corr(self.rank(self.l), self.rank(self.ts_mean(self.v, 50)), 12))).astype(int).stack(
            'ticker').swaplevel())

    def alpha077(self):
        self.copyToNegateReadOnly()
        s1 = self.rank(self.ts_weighted_mean(self.h.add(self.l).div(2).sub(self.vwap), 20))
        s2 = self.rank(self.ts_weighted_mean(self.ts_corr(self.h.add(self.l).div(2), self.ts_mean(self.v, 40), 3), 5))
        self.data['077'] = (s1.where(s1 < s2, s2).stack('ticker').swaplevel())

    def alpha078(self):
        self.copyToNegateReadOnly()
        w = 0.352233
        self.data['078'] = (self.rank(self.ts_corr(self.ts_sum((self.l.mul(w).add(self.vwap.mul(1 - w))), 19),
                                                   self.ts_sum(self.ts_mean(self.v, 40), 19), 6)).pow(
            self.rank(self.ts_corr(self.rank(self.vwap), self.rank(self.v), 5))).stack('ticker').swaplevel())

    def alpha081(self):
        self.copyToNegateReadOnly()
        self.data['081'] = (self.rank(self.log(self.ts_product(
            self.rank(self.rank(self.ts_corr(self.vwap, self.ts_sum(self.ts_mean(self.v, 10), 50), 8)).pow(4)),
            15))).lt(self.rank(self.ts_corr(self.rank(self.vwap), self.rank(self.v), 5))).mul(-1).stack(
            'ticker').swaplevel())

    def alpha083(self):
        self.copyToNegateReadOnly()
        s = self.h.sub(self.l).div(self.ts_mean(self.c, 5))
        self.data['083'] = (self.rank(self.rank(self.ts_lag(s, 2)).mul(self.rank(self.rank(self.v))).div(s).div(
            self.vwap.sub(self.c).add(1e-3))).stack('ticker').swaplevel().replace((np.inf, -np.inf), np.nan))

    def alpha084(self):
        self.copyToNegateReadOnly()
        self.data['084'] = (self.rank(
            self.power(self.ts_rank(self.vwap.sub(self.ts_max(self.vwap, 15)), 20), self.ts_delta(self.c, 6))).stack(
            'ticker').swaplevel())

    def alpha085(self):
        self.copyToNegateReadOnly()
        w = 0.876703
        self.data['085'] = (
            self.rank(self.ts_corr(self.h.mul(w).add(self.c.mul(1 - w)), self.ts_mean(self.v, 30), 10)).pow(
                self.rank(self.ts_corr(self.ts_rank(self.h.add(self.l).div(2), 4), self.ts_rank(self.v, 10), 7))).stack(
                'ticker').swaplevel())

    def alpha086(self):
        self.copyToNegateReadOnly()
        self.data['086'] = (self.ts_rank(self.ts_corr(self.c, self.ts_mean(self.ts_mean(self.v, 20), 15), 6), 20).lt(
            self.rank(self.c.sub(self.vwap))).mul(-1).stack('ticker').swaplevel())

    def alpha088(self):
        self.copyToNegateReadOnly()
        s1 = (self.rank(self.ts_weighted_mean(
            self.rank(self.o).add(self.rank(self.l)).sub(self.rank(self.h)).add(self.rank(self.c)), 8)))
        s2 = self.ts_rank(
            self.ts_weighted_mean(self.ts_corr(self.ts_rank(self.c, 8), self.ts_rank(self.ts_mean(self.v, 60), 20), 8),
                                  6), 2)
        self.data['088'] = (s1.where(s1 < s2, s2).stack('ticker').swaplevel())

    def alpha094(self):
        self.copyToNegateReadOnly()
        self.data['094'] = (self.rank(self.vwap.sub(self.ts_min(self.vwap, 11))).pow(
            self.ts_rank(self.ts_corr(self.ts_rank(self.vwap, 20), self.ts_rank(self.ts_mean(self.v, 60), 4), 18),
                         2)).mul(-1).stack('ticker').swaplevel())

    def alpha095(self):
        self.copyToNegateReadOnly()
        self.data['095'] = (self.rank(self.o.sub(self.ts_min(self.o, 12))).lt(self.ts_rank(self.rank(
            self.ts_corr(self.ts_mean(self.h.add(self.l).div(2), 19), self.ts_sum(self.ts_mean(self.v, 40), 19),
                         13).pow(5)), 12)).astype(int).stack('ticker').swaplevel())

    def alpha099(self):
        self.copyToNegateReadOnly()
        self.data['099'] = ((self.rank(
            self.ts_corr(self.ts_sum((self.h.add(self.l).div(2)), 19), self.ts_sum(self.ts_mean(self.v, 60), 19),
                         8)).lt(self.rank(self.ts_corr(self.l, self.v, 6))).mul(-1)).stack('ticker').swaplevel())

    def alpha101(self):
        self.copyToNegateReadOnly()
        self.data['101'] = (self.c.sub(self.o).div(self.h.sub(self.l).add(1e-3)).stack('ticker').swaplevel())

