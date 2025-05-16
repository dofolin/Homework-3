"""
Markowitz_2.py — 雙策略融合版 for Problem 4
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY", "XLB", "XLC", "XLE", "XLF", "XLI",
    "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"
]

# 取得歷史資料
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust=False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

class MyPortfolio:
    def __init__(self, price, exclude="SPY", mode="one"):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.mode = mode

    def calculate_weights(self):
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=self.price.index, columns=self.price.columns)
        prev_w = pd.Series(0, index=self.price.columns)

        if self.mode == "spy":
            momentum_window = 180
            top_n = 5
        else:  # mode == "one"
            momentum_window = 90
            top_frac = 0.3

        for i in range(momentum_window, len(self.price)):
            date = self.price.index[i]
            if i > momentum_window and date.month == self.price.index[i - 1].month:
                self.portfolio_weights.loc[date] = prev_w
                continue

            if self.mode == "spy":
                momentum = self.price[assets].iloc[i - momentum_window:i].apply(lambda x: x[-1] / x[0] - 1)
                selected = momentum[momentum > 0].sort_values(ascending=False).head(top_n).index
                w = pd.Series(0, index=self.price.columns)
                if len(selected) > 0:
                    w[selected] = 1.0 / len(selected)
                    w = np.clip(w, 0, 0.35)
                    w /= w.sum()
            else:
                window = self.returns.iloc[i - momentum_window : i-1]
                cum_ret = (1 + window[assets]).prod() - 1
                cum_ret = cum_ret / (1 + window[assets].iloc[-1])
                score = cum_ret.rank(ascending=False)
                k = max(1, int(len(assets) * top_frac))
                win = score[score <= k].index
                sigma = window[assets].std().replace(0, 1e-9)[win]
                sigma = sigma / sigma.sum()
                w = pd.Series(0, index=self.price.columns)
                w[win] = 1 / k * sigma
                w = np.clip(w, 0, 0.3)
                w /= w.sum()

            self.portfolio_weights.loc[date] = w
            prev_w = w

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns = self.returns.copy()
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets] * self.portfolio_weights[assets]
        ).sum(axis=1)

    def get_results(self):
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()
        return self.portfolio_weights, self.portfolio_returns


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY", mode="one").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY", mode="spy").get_results()

    def plot_performance(self, price, strategy):
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")
        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe = df_bl.mean() / df_bl.std() * np.sqrt(252)
        if show:
            print("Sharpe Ratio (MP vs SPY):")
            print(sharpe)
        return sharpe

    def check_portfolio_position(self, weights):
        return (weights.sum(axis=1) <= 1.01).all()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            print("Leverage violation")
            return 0
        if self.report_metrics(df, self.mp)["MP"] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.Bmp[0]):
            print("Leverage violation")
            return 0
        result = self.report_metrics(Bdf, self.Bmp)
        if result["MP"] > result["SPY"]:
            print("Problem 4.2 Success - Get 15 points")
            return 15
        print("Problem 4.2 Fail")
        return 0

    def check_all_answer(self):
        return self.check_sharp_ratio_greater_than_one() + self.check_sharp_ratio_greater_than_spy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="append")
    parser.add_argument("--report", action="append")
    parser.add_argument("--performance", action="append")
    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if "one" in args.score:
            judge.check_sharp_ratio_greater_than_one()
        if "spy" in args.score:
            judge.check_sharp_ratio_greater_than_spy()
        if "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.report:
        judge.report_metrics(df, judge.mp, show=True)

    if args.performance:
        judge.plot_performance(df, judge.mp)