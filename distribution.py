import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DistributionPlotter:
    def __init__(self, data, cols=None, hue_col=None, hue_order=None, max_cat_thr=20):
        self.data = data.copy()
        self.hue_col = hue_col
        self.max_cat_thr = max_cat_thr
        self.cols = cols or data.columns
        self.categorical_cols = self.get_categorical_columns()
        self.numeric_cols = self.get_numeric_columns()
        self.hue_order = hue_order
        self.fig, self.ax = None, None

    def get_numeric_columns(self):
        return [
            col
            for col in set(self.data.columns).intersection(self.cols)
            if col not in self.categorical_cols
            and pd.api.types.is_numeric_dtype(self.data[col])
            and col != self.hue_col
        ]

    def get_categorical_columns(self):
        return [
            col
            for col in set(self.data.columns).intersection(self.cols)
            if len(self.data[col].unique()) <= self.max_cat_thr and col != self.hue_col
        ]

    def plot_distribution(self, column, drop_zero=False):
        if column in self.numeric_cols:
            if drop_zero:
                data = self.data[self.data[column] != 0]
            else:
                data = self.data

            self.fig, self.ax = plt.subplots(1, 3, figsize=(18, 6))

            # Plot histogram
            sns.histplot(
                data=data,
                x=column,
                hue=self.hue_col,
                hue_order=self.hue_order,
                multiple="stack",
                element="step",
                stat="count",
                alpha=0.8,
                ax=self.ax[0],
            )
            self.ax[0].set_title(f"Histogram of {column}")
            self.ax[0].grid(True, axis="y")

            # Plot boxenplot
            sns.boxenplot(
                data=data,
                x=self.hue_col,
                y=column,
                showfliers=False,
                ax=self.ax[1],
                order=self.hue_order,
            )
            sns.stripplot(
                data=(
                    data.sample(200, random_state=42)
                    if data[column].count() > 200
                    else data
                ),
                x=self.hue_col,
                y=column,
                color="black",
                alpha=0.5,
                hue_order=self.hue_order,
                ax=self.ax[1],
            )
            self.ax[1].set_title(f"Boxenplot of {column}")
            self.ax[1].grid(True, axis="y")

            # Plot special distribution (0 and NaN)
            if self.hue_col:
                data = (
                    self.data.groupby(self.hue_col)[column]
                    .agg(
                        **{
                            "0": lambda x: (x == 0).sum() / x.size,
                            "NaN": lambda x: x.isna().sum() / x.size,
                        }
                    )
                    .reset_index()
                )
                data["0"].replace(
                    0, -0.01 * data[["0", "NaN"]].max().max(), inplace=True
                )
                data["NaN"].replace(
                    0, -0.01 * data[["0", "NaN"]].max().max(), inplace=True
                )
                data = pd.melt(
                    data, id_vars=self.hue_col, var_name="Column", value_name="Value"
                )
                sns.barplot(
                    data=data,
                    x="Column",
                    y="Value",
                    hue=self.hue_col,
                    hue_order=self.hue_order,
                )
            else:
                data = self.data[column].agg(
                    **{
                        "0": lambda x: (x == 0).sum() / x.size,
                        "NaN": lambda x: x.isna().sum() / x.size,
                    }
                )
                data.replace(0, -0.01 * data.max(), inplace=True)
                sns.barplot(
                    data=data,
                    edgecolor="black",
                    ax=self.ax[2],
                )
            self.ax[2].axhline(0, color="black", ls="--")
            self.ax[2].set_title(f"0 and NaN proportion of {column}")
            self.ax[2].grid(True, axis="y")

            plt.tight_layout()

        elif column in self.categorical_cols:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            data = self.data.copy()
            data[column] = data[column].astype(str)
            data[column].fillna("<NaN>", inplace=True)
            data[column].replace("", "<EMPTY>", inplace=True)
            sns.countplot(
                data=data,
                x=column,
                hue=self.hue_col,
                stat="count",
                edgecolor="black",
                ax=self.ax,
            )
            self.ax.set_title(f"Count of {column} classes")
            self.ax.tick_params("x", rotation=90)

    def show_plot(self):
        plt.show()

    def plot_all(self, drop_zero=False):
        for numeric_column in self.get_numeric_columns():
            self.plot_distribution(numeric_column, drop_zero)

        for categorical_column in self.get_categorical_columns():
            self.plot_distribution(categorical_column, drop_zero)


def get_df_info(data: pd.DataFrame, thr: float = 0.8, *args, **kwargs):
    data = data.drop(columns=data.columns[data.dtypes == "object"])
    data_info = pd.DataFrame(index=data.columns)

    data_info["dtype"] = data.dtypes

    data_info["nunique"] = data.nunique(dropna=False)

    nans = data.isna()
    data_info["nan"] = nans.mean()[nans.sum() > 0].round(3).apply(lambda x: f"n: {x}")

    zeroes = data.eq(0)
    data_info["zero"] = (
        zeroes.mean()[zeroes.sum() > 0].round(3).apply(lambda x: f"z: {x}")
    )

    empty_strs = data.eq("")
    data_info["empty string"] = (
        empty_strs.mean()[empty_strs.sum() > 0].round(3).apply(lambda x: f"e: {x}")
    )

    data_info["example(-s)"] = data.apply(
        lambda x: pd.Series(x.dropna().unique()).sample(
            x.nunique() > 1 and 2 or 1, ignore_index=True, random_state=42
        )
    ).T.apply(tuple, axis=1)

    modes = data.apply(
        lambda x: round(x.value_counts(normalize=True).rename_axis(""), 3)
        .reset_index()
        .loc[0]
    ).T
    data_info["mode, mode proportion"] = modes.apply(tuple, axis=1)

    trash_sum = nans + zeroes + empty_strs
    data_info["trash_score"] = pd.DataFrame(
        [
            modes.loc[modes["proportion"] > thr, "proportion"],
            trash_sum.mean()[trash_sum.sum() > 0].round(3),
        ]
    ).max()

    data_info = pd.DataFrame(data_info).sort_values("trash_score", ascending=False)

    return data_info
