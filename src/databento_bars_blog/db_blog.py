import marimo

__generated_with = "0.6.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import polars as pl
    import pandas as pd
    from polars.type_aliases import FrameType
    from typing import Literal
    from datetime import timedelta
    from polars_finance.bars import (
        tick_bars,
        volume_bars,
        dollar_bars,
        standard_bars,
    )
    import hvplot.polars
    import holoviews as hv
    from great_tables import GT
    from datetime import date

    hv.extension("bokeh")

    pl.enable_string_cache()
    return (
        FrameType,
        GT,
        Literal,
        date,
        dollar_bars,
        hv,
        hvplot,
        mo,
        pd,
        pl,
        standard_bars,
        tick_bars,
        timedelta,
        volume_bars,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Downsampling Pricing Data into Useful Bars with Python

        ## Introduction

        There are a lot of ways to downsample granular market data into useful price bars. The most common of these ways is to sample by time, calculating an OHLCV bar for each second, minute, hour, or day. In some respects, this is a great way to aggregate bars. From a human perspective, it is easy to understand and measure changes in value over set periods of time. However, there are other useful ways to downsample and aggregate this data to provide more information dense price bars to trading algorithms. All of the strategies I am going to review today can be found in the book [**Advances in Financial Machine Learning** by *Marcos Lopez De Prado*](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089).

        We are going to walk through how to go from data that comes from the `trades` schema from DataBento's Nasdaq ITCH dataset, to calculate the following types of price bars and understand the differences between each of them:  

        - Standard Time Bars  
        - Tick Bars  

        ## Dependencies

        One of my favorite parts of DataBento is their attention to performance. In that same line of thinking, we will be going through these exercises using [Polars](https://docs.pola.rs/) as opposed to the very popular [Pandas](https://pandas.pydata.org/) library. This is due to a few reasons:  

        - **Memory:** When we are aggregating huge datasets, being memory efficient is important. And Polars is much better at being memory efficient with data than Pandas.  
        - **Speed:** We will see a huge speed boost using Polars as opposed to Pandas.  
        - **Type Safety:** Polars adheres to a stricter typing system minimizing the potential for errors.  

        I will defer to other blog posts if you want to read more about the comparison between the two libraries. For my purposes, I will simply state that I believe Polars to be a better option for dealing with the data provided by DataBento and we will move forward.

        ## Getting Data

        We will go ahead and pull some data from DataBento using the python API. We will grab data for 4 securities over a month time period from the trades schema and load the data into polars using the code snippet below:

        **Note:** In the notebook below I read data from parquet files that I saved from Databento in order to save myself from pinging the Databento service over and over.
        """
    )
    return


@app.cell
def __(mo):
    # Example of how to get data from DataBento
    mo.ui.code_editor(
        """
        import polars as pl
        import databento as db
        import datetime as dt

        start_time = dt.datetime(2024, 2, 12, 9, 30, 0, 0, pytz.timezone("US/Eastern"))
        end_time = dt.datetime(2024, 3, 12, 16, 0, 0, 0, pytz.timezone("US/Eastern"))

        client = db.Historical()  # Either pass your API key here or set your ENV variable.
        dbn = client.timeseries.get_range(
            "XNAS.ITCH",
            start_time,
            end_time,
            symbols=["AAPL", "MSFT", "AMZN", "TQQQ"],
            schema="trades"
        )  # This will cost ~5 dollars.

        # Make the dataframe lazy for future ops
        df = pl.from_pandas(dbn.to_df().reset_index()).lazy()  
        """,
        language="python",
    )
    return


@app.cell
def __(pl):
    df = pl.scan_parquet("./data/itch.parquet")
    return df,


@app.cell
def __(mo):
    mo.md(
        """
        ## Data Preparation
        We will do a few small things to make the dataset more manageable for our purposes. First, we are going to trim the data down from all columns, to just the data we need. We will then change the types of some our columns so that we are not working with strings. Then for readability, we will convert our UTC datetimes to US/Eastern time to line up with the NYSE market hours.  

        Here is a quick look at what our data looks like before making any changes:
        """
    )
    return


@app.cell
def __(df):
    df.collect().head()
    return


@app.cell
def __(df, pl):
    clean_df = (
        df.select(
            pl.col("symbol").cast(pl.Enum(["AAPL", "AMZN", "MSFT", "TQQQ"])),
            pl.col("ts_event").dt.convert_time_zone("US/Eastern"),
            "price",
            "size",
        )
        .sort("symbol", "ts_event")
        .collect()
    )
    clean_df.head()
    return clean_df,


@app.cell
def __(clean_df, standard_bars):
    standard_bars(clean_df, bar_size="15m")
    return


@app.cell
def __(clean_df, tick_bars):
    tick_bars(clean_df, bar_size=200)
    return


@app.cell
def __(clean_df, pl, standard_bars, tick_bars):
    _sb = standard_bars(clean_df)
    _tb = tick_bars(clean_df, bar_size=250)
    # _vb = volume_bars(df, bar_size=10_000).collect()
    # _db = dollar_bars(df, bar_size=2_000_000).collect()


    def get_counts_by_time(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col(time_col).dt.date().alias("date"),
                pl.col(time_col).dt.hour().alias("hour"),
            )
            .group_by("date", "hour")
            .agg(pl.len().alias("bar_count"))
            .group_by("hour")
            .agg(pl.col("bar_count").mean())
            .sort("hour")
        )


    get_counts_by_time(_sb.filter(pl.col("symbol") == "AAPL"), "ts_event").rename(
        {"bar_count": "time"}
    ).join(
        get_counts_by_time(_tb.filter(pl.col("symbol") == "AAPL"), "begin_ts_event").rename(
            {"bar_count": "tick"}
        ),
        on="hour",
    ).hvplot.bar(
        x="hour",
        y=["time", "tick"],
        stacked=False,
        title="Avgerage Number of Bars by Hour - AAPL",
        height=500,
        responsive=True,
        xlabel="Hour",
        ylabel="Number of Bars",
        legend="top_left",
    ).opts(
        xrotation=90
    )
    return get_counts_by_time,


@app.cell
def __(mo, pl):
    # Python impl volume bars
    def volume_bars_py(df: pl.DataFrame, bar_size) -> pl.DataFrame:

        def _to_ohlcv_df(bar_rows: list[tuple]):
            return pl.DataFrame(
                bar_rows, schema=["symbol", "ts_event", "price", "size"]
            ).select(
                pl.col("symbol").first(),
                pl.col("ts_event").first().alias("start_dt"),
                pl.col("ts_event").last().alias("end_dt"),
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias(
                    "vwap"
                ),
                pl.col("size").sum().alias("volume"),
                pl.len().alias("n_transactions"),
            )

        ohlcv_rows = []
        current_bar = []
        for row in mo.status.progress_bar(df.sort("ts_event").rows(named=True)):
            remaining_size = bar_size - sum([r[-1] for r in current_bar])
            while row["size"] > remaining_size:
                current_bar.append(
                    (row["symbol"], row["ts_event"], row["price"], remaining_size)
                )
                ohlcv_rows.append(_to_ohlcv_df(current_bar))
                current_bar = []
                row["size"] = row["size"] - remaining_size
                remaining_size = bar_size

            if row["size"] > 0:
                current_bar.append(
                    (row["symbol"], row["ts_event"], row["price"], row["size"])
                )
        return pl.concat(ohlcv_rows)
    return volume_bars_py,


@app.cell
def __(clean_df, date, pl):
    clean_df.filter(pl.col("symbol") == "AAPL").filter(
        pl.col("ts_event").dt.date() == date(2024, 2, 12)
    )
    return


@app.cell
def __(clean_df, date, pl, volume_bars_py):
    volume_bars_py(
        clean_df.filter(pl.col("symbol") == "AAPL").filter(
            pl.col("ts_event").dt.date() == date(2024, 2, 12)
        ),
        1000,
    )
    return


@app.cell
def __(clean_df, date, pl, volume_bars):
    volume_bars(
        clean_df.filter(pl.col("symbol") == "AAPL").filter(
            pl.col("ts_event").dt.date() == date(2024, 2, 12)
        ),
        bar_size=1000,
    )
    return


@app.cell
def __(pl):
    def autocorr(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1)
                .over("symbol")
                .alias("return")
            )
            .with_columns(pl.col("return").shift(1).over("symbol").alias("lagged_return"))
            .group_by("symbol")
            .agg(pl.corr("return", "lagged_return"))
        )
    return autocorr,


@app.cell
def __(pl):
    # Bring your own data here
    aapl = (
        pl.scan_parquet("./data/xnas_2023_2024/*.parquet")
        .filter(pl.col("symbol") == "AAPL")
        .collect()
    )
    return aapl,


@app.cell
def __(GT, pl):
    GT(
        pl.DataFrame({"time": 2.97, "tick": 1.79, "volume": 10.48, "dollar": 43.23})
    ).tab_header("Compute Time AAPL - 1yr (seconds)")
    return


@app.cell
def __(aapl, standard_bars):
    aapl_std = standard_bars(aapl, bar_size="15m").rename({"ts_event": "start_dt"})
    return aapl_std,


@app.cell
def __(aapl, tick_bars):
    aapl_tick = tick_bars(aapl, bar_size=2200).rename({"begin_ts_event": "start_dt"})
    return aapl_tick,


@app.cell
def __(aapl, volume_bars):
    aapl_vol = volume_bars(aapl, bar_size=330_000)
    return aapl_vol,


@app.cell
def __(aapl, dollar_bars):
    aapl_dollar = dollar_bars(aapl, bar_size=6e7)
    return aapl_dollar,


@app.cell
def __(pl):
    def get_daily_bars(df, n):
        return df.group_by(pl.col("start_dt").dt.date()).agg(pl.len().alias(f"n_bars_{n}"))
    return get_daily_bars,


@app.cell
def __(aapl_dollar, aapl_std, aapl_tick, aapl_vol, get_daily_bars):
    from functools import reduce

    bar_counts = reduce(
        lambda x, y: x.join(y, on="start_dt"),
        [
            get_daily_bars(d, n)
            for d, n in [
                (aapl_std, "time"),
                (aapl_tick, "tick"),
                (aapl_vol, "volume"),
                (aapl_dollar, "dollar"),
            ]
        ],
    )
    bar_counts.plot.line(
        x="start_dt",
        y=["n_bars_tick", "n_bars_volume", "n_bars_dollar"],
        height=400,
        responsive=True,
        title="Number of Bars Per Day",
    )
    return bar_counts, reduce


@app.cell
def __(aapl_std, aapl_tick, aapl_vol, pl):
    from bokeh.models.formatters import NumeralTickFormatter

    aapl_std.group_by(pl.col("start_dt").dt.truncate("1h")).agg(pl.len()).with_columns(
        pl.col("len") / pl.col("len").sum().over(pl.col("start_dt").dt.date())
    ).group_by(pl.col("start_dt").dt.hour().alias("Hour")).agg(
        pl.mean("len").alias("Time")
    ).sort(
        "Hour"
    ).join(
        aapl_tick.group_by(pl.col("end_ts_event").dt.truncate("1h"))
        .agg(pl.len())
        .with_columns(
            pl.col("len") / pl.col("len").sum().over(pl.col("end_ts_event").dt.date())
        )
        .group_by(pl.col("end_ts_event").dt.hour().alias("Hour"))
        .agg(pl.mean("len").alias("Tick"))
        .sort("Hour"),
        on="Hour",
    ).join(
        aapl_vol.with_columns(pl.col("end_dt").dt.convert_time_zone("US/Eastern"))
        .group_by(pl.col("end_dt").dt.truncate("1h"))
        .agg(pl.len())
        .with_columns(pl.col("len") / pl.col("len").sum().over(pl.col("end_dt").dt.date()))
        .group_by(pl.col("end_dt").dt.hour().alias("Hour"))
        .agg(pl.mean("len").alias("Volume"))
        .sort("Hour"),
        on="Hour",
    ).plot.bar(
        x="Hour",
        y=["Tick", "Volume"],
        height=400,
        responsive=True,
        ylabel="Percent of Bars",
        xlabel="Hour",
        title="Percent of Bars by Hour",
        yformatter=NumeralTickFormatter(format="%0"),
    ).opts(xrotation=45)
    return NumeralTickFormatter,


if __name__ == "__main__":
    app.run()
