"""
CONVENTIONS:
    - A DataFrame of *transactions* is one with the following columns:
        * date: Numpy datetime
        * amount: float
        * description (optional): string
        * category (optional): Pandas 'category'
        * comment (optional): string
"""
import random
from collections import OrderedDict
from copy import deepcopy

import pandas as pd
import numpy as np
import colorlover as cl
from highcharts import Highchart


COLUMNS = {"date", "amount", "description", "category", "comment"}
REQUIRED_COLUMNS = {"date", "amount"}


def create_transactions(
    date1, date2, freq="12H", income_categories=None, expense_categories=None
):
    """
    Create a DataFrame of sample transactions between the given dates
    (date strings that Pandas can interpret, such as YYYYMMDD) and at
    the given Pandas frequency.
    Include all the columns in the set ``COLUMNS``.
    Each positive transaction will be assigned a income category from
    the given list ``income_categories``, and each negative transaction
    will be assigned a expense category from the given list
    ``expense_categories``.
    If these lists are not given, then whimsical default ones will be
    created.
    """
    # Create date range
    rng = pd.date_range(date1, date2, freq=freq, name="date")
    n = len(rng)

    # Create random amounts
    low = -70
    high = 100
    f = pd.DataFrame(
        np.random.randint(low, high, size=(n, 1)), columns=["amount"], index=rng
    )
    f = f.reset_index()

    # Create random descriptions and comments
    f["description"] = [hex(random.getrandbits(20)) for i in range(n)]
    f["comment"] = [hex(random.getrandbits(40)) for i in range(n)]

    # Categorize amounts
    if income_categories is None:
        income_categories = ["programming", "programming", "investing", "reiki"]
    if expense_categories is None:
        expense_categories = [
            "food",
            "shelter",
            "shelter",
            "transport",
            "healthcare",
            "soil testing",
        ]

    def categorize(x):
        if x > 0:
            return random.choice(income_categories)
        else:
            return random.choice(expense_categories)

    f["category"] = f["amount"].map(categorize)
    f["category"] = f["category"].astype("category")

    return f


def find_columns(raw_transactions):
    """
    Given a DataFrame, lowercase the column names and search for the
    columns in ``COLUMNS``.
    Build a dictionary of the form
    name in ``COLUMNS`` -> name in given DataFrame
    and return the result, which might not contain all the ``COLUMNS``
    keys.
    """
    f = raw_transactions.copy()
    col_dict = {}
    for key in COLUMNS:
        for c in f.columns:
            if key == c.lower():
                col_dict[key] = c
    return col_dict


def read_transactions(path, date_format=None, **kwargs):
    """
    Read a CSV file of transactions located at the given path (string
    or Path object), parse the date and category, and return the
    resulting DataFrame.

    The CSV should contain at least the following columns

    - ``'date'``: string
    - ``'amount'``: float; amount of transaction; positive or negative,
      indicating a income or expense, respectively
    - ``'description'`` (optional): string; description of transaction,
      e.g. 'dandelion and burdock tea'
    - ``'category'`` (optional): string; categorization of description,
      e.g. 'healthcare'
    - ``'comment'`` (optional): string; comment on transaction, e.g.
      'a gram of prevention is worth 62.5 grams of cure'

    If the date format string ``date_format`` is given,  e.g
    ``'%Y-%m-%d'``, then parse dates using that format; otherwise use
    let Pandas guess the date format.
    """
    f = pd.read_csv(path, **kwargs)
    col_dict = find_columns(f)
    if not set(col_dict.keys()) >= REQUIRED_COLUMNS:
        raise ValueError(
            "Could not find columns resembling {!s} in file".format(REQUIRED_COLUMNS)
        )

    # Reformat column names
    rename1 = {val: key for key, val in col_dict.items()}
    rename2 = {c: c.strip().lower().replace(" ", "_") for c in f.columns}
    f = f.rename(columns=rename1).rename(columns=rename2)

    # Parse some
    f["date"] = pd.to_datetime(f["date"], format=date_format)
    if "category" in f.columns:
        f["category"] = f["category"].str.lower()
        f["category"] = f["category"].astype("category")

    return f.sort_values(["date", "amount"])


def insert_repeating(
    transactions,
    amount,
    freq,
    description=None,
    category=None,
    comment=None,
    start_date=None,
    end_date=None,
):
    """
    Given a DataFrame of transactions, add to it a repeating transaction
    at the given frequency for the given amount with the given optional
    description, category, and comment.
    Restrict the repeating transaction to the given start and end dates
    (date objects), inclusive, if given; otherwise repeat from the first
    transaction date to the last.
    Drop duplicate rows and return the resulting DataFrame.
    """
    f = transactions.copy()
    if start_date is None:
        start_date = f["date"].min()
    if end_date is None:
        end_date = f["date"].max()

    g = pd.DataFrame([])
    dates = pd.date_range(start_date, end_date, freq=freq)
    g["date"] = dates
    g["amount"] = amount

    if description is not None:
        g["description"] = description
    if category is not None:
        g["category"] = category
        g["category"] = g["category"].astype("category")
    if comment is not None:
        g["comment"] = comment

    h = pd.concat([f, g]).drop_duplicates().sort_values(["date", "amount"])
    if "category" in h.columns:
        h["category"] = h["category"].astype("category")

    return h


def summarize(
    transactions,
    freq=None,
    by_category=False,
    decimals=2,
    start_date=None,
    end_date=None,
):
    """
    Given a DataFrame of transactions, slice it from the given start
    date to and including the given end date date (strings that Pandas
    can interpret, such as YYYYMMDD) if specified, and return a
    DataFrame with the columns:

    - ``'date'``: start date of period
    - ``'category'`` (if `by_category`): transaction category
    - ``'amount'``: amount for the period and category
    - ``'balance'``: cumulative sum of income - expense
    - ``'savings_pc_for_period'``:
      100*(income sum - expense sum)/(income sum)
    - ``'spending_pc_for_period'``: 100*expense/(income sum)
    - ``'spending_pc_for_period_and_category'`` (if `by_category`):
      100*expense/(income sum)
    - ``'income_pc_for_period_and_category'`` (if `by_category`):
      100*income/(income sum)
    - ``'expense_pc_for_period_and_category'`` (if `by_category`):
      100*expense/(expense sum)
    - ``'daily_avg'`` (if `freq is None`): income - expense
      divided by number of days between start and end date
    - ``'weekly_avg'`` (if `freq is None`): income - expense
      divided by number of weeks between start and end date
    - ``'monthly_avg'`` (if `freq is None`): income - expense
      divided by number of months (possibly fractional) between
      start and end date
    - ``'yearly_avg'`` (if `freq is None`): income - expense
      divided by number of years (possibly fractional) between
      start and end date

    The period is given by the Pandas frequency string ``freq``.
    If that frequency is ``None``, then there is only one period,
    namely one that runs from the first to the last date in
    ``transactions`` (ordered chronologically);
    the ``'date'`` value is then the first date.

    If ``by_category``, then group by the ``'category'`` column of
    ``transactions`` in addition to the period.

    Round all values to the given number of decimals.
    """
    f = transactions.copy()
    if by_category and "category" not in f.columns:
        raise ValueError("category column missing from DataFrame")

    # Set start and end dates
    if start_date is None:
        start_date = f["date"].min()
    else:
        start_date = pd.to_datetime(start_date)
    if end_date is None:
        end_date = f["date"].max()
    else:
        end_date = pd.to_datetime(end_date)

    # Filter to start and end dates
    f = f.loc[lambda x: (x.date >= start_date) & (x.date <= end_date)].copy()

    # Removed unused categories
    if "category" in f.columns:
        f.category = f.category.cat.remove_unused_categories()

    # Create income and expense columns
    f["income"] = f["amount"].map(lambda x: x if x > 0 else 0)
    f["expense"] = f["amount"].map(lambda x: -x if x < 0 else 0)
    del f["amount"]

    if freq is None:
        income = f["income"].sum()
        expense = f["expense"].sum()

        if income:
            savings_pc = 100 * (income - expense) / income
            spending_pc = 100 * expense / income
        else:
            savings_pc = np.nan
            spending_pc = np.nan

        if by_category:
            g = f.groupby("category").sum().reset_index()
            g["balance"] = income - expense
            g["savings_pc_for_period"] = savings_pc
            g["spending_pc_for_period"] = spending_pc
            g["spending_pc_for_period_and_category"] = 100 * g["expense"] / income
            g["income_pc_for_period_and_category"] = 100 * g["income"] / income
            g["expense_pc_for_period_and_category"] = 100 * g["expense"] / expense
        else:
            d = OrderedDict()
            d["income"] = income
            d["expense"] = expense
            d["balance"] = income - expense
            d["savings_pc_for_period"] = savings_pc
            d["spending_pc_for_period"] = spending_pc
            g = pd.DataFrame(d, index=[0])

        # Append first transaction date
        g["date"] = start_date

        # Also append dailyng, weekly, etc. averages
        delta = end_date - start_date
        num_days = delta.days + 1
        num_weeks = num_days / 7
        num_months = num_days / (365 / 12)
        num_years = num_days / 365
        g["daily_avg"] = (g["income"] - g["expense"]) / num_days
        g["weekly_avg"] = (g["income"] - g["expense"]) / num_weeks
        g["monthly_avg"] = (g["income"] - g["expense"]) / num_months
        g["yearly_avg"] = (g["income"] - g["expense"]) / num_years

    else:
        tg = pd.Grouper(freq=freq, label="left", closed="left")
        if by_category:
            cols = [tg, "category"]
            g = f.set_index("date").groupby(cols).sum().reset_index()
            balance = 0
            balances = []
            savings_pcs = []
            spending_pcs = []
            spending_pcs_c = []
            income_pcs_c = []
            expense_pcs_c = []
            for __, group in g.set_index("date").groupby(tg):
                n = group.shape[0]
                balance += (group["income"] - group["expense"]).sum()
                balances.extend([balance for i in range(n)])
                savings_pc = (
                    100
                    * (group["income"] - group["expense"]).sum()
                    / group["income"].sum()
                )
                savings_pcs.extend([savings_pc for i in range(n)])
                spending_pc = 100 * group["expense"].sum() / group["income"].sum()
                spending_pcs.extend([spending_pc for i in range(n)])
                spending_pc_c = 100 * group["expense"] / group["income"].sum()
                spending_pcs_c.extend(spending_pc_c.values)
                income_pc_c = 100 * group["income"] / group["income"].sum()
                income_pcs_c.extend(income_pc_c.values)
                expense_pc_c = 100 * group["expense"] / group["expense"].sum()
                expense_pcs_c.extend(expense_pc_c.values)
            g["balance"] = balances
            g["savings_pc_for_period"] = savings_pcs
            g["spending_pc_for_period"] = spending_pcs
            g["spending_pc_for_period_and_category"] = spending_pcs_c
            g["income_pc_for_period_and_category"] = income_pcs_c
            g["expense_pc_for_period_and_category"] = expense_pcs_c
        else:
            g = f.set_index("date").groupby(tg).sum().reset_index()
            g["balance"] = (g["income"] - g["expense"]).cumsum()
            g["savings_pc_for_period"] = (
                100 * (g["income"] - g["expense"]) / g["income"]
            )
            g["spending_pc_for_period"] = 100 * g["expense"] / g["income"]

    # Prepare final columns
    if by_category:
        cols = g.columns.tolist()
        cols.remove("date")
        cols.remove("category")
        cols.insert(0, "date")
        cols.insert(1, "category")
        g = g[cols].copy()

    # Replace infinities with nans
    g = g.replace(np.inf, np.nan).sort_values(
        ["date", "spending_pc_for_period", "savings_pc_for_period"],
        ascending=[True, True, False],
    )

    # Round
    if decimals is not None:
        g = g.round(decimals)

    return g


def get_colors(column_name, n):
    """
    Return a list of ``n`` (positive integer) nice RGB color strings
    to use for color coding the given column (string; one of
    ``['income', 'expense', 'period_budget', 'balance']``.

    NOTES:

    - Returns at most 6 distinct colors. Repeats color beyond that.
    - Helper function for :func:`plot`.
    """

    # Clip n to range or sequential-type colors
    low = 3
    high = 6
    k = np.clip(n, low, high)
    kk = str(k)

    # Build colors in clipped range
    if column_name == "income":
        colors = cl.scales[kk]["seq"]["GnBu"][::-1]
    elif column_name == "expense":
        colors = cl.scales[kk]["seq"]["OrRd"][::-1]
    else:
        colors = ["#555" for __ in range(k)]

    # Extend colors to unclipped range as required
    if n <= 0:
        colors = []
    elif 0 < n < low:
        colors = colors[:n]
    elif n > high:
        # Repeat colors
        q, r = divmod(n, k)
        colors = colors * q + colors[:r]

    return colors


def plot(summary, currency=None, width=None, height=None):
    """
    Given a transaction summary of the form output by :func:`summarize`,
    plot it using Python HighCharts.
    Include the given currency units (string; e.g. 'NZD') in the y-axis
    label.
    Override the default chart width and height, if desired.
    """
    f = summary.copy()
    chart = Highchart()

    # Initialize chart options.
    # HighCharts kludge: use categorical x-axis to display dates properly.
    dates = f["date"].map(lambda x: x.strftime("%Y-%m-%d")).unique()
    dates = sorted(dates.tolist())

    if currency is not None:
        y_text = "Money ({!s})".format(currency)
    else:
        currency = ""
        y_text = "Money"

    chart_opts = {
        "lang": {"thousandsSep": ","},
        "chart": {"zoomType": "xy"},
        "title": {"text": "Account Summary"},
        "xAxis": {"type": "category", "categories": dates},
        "yAxis": {"title": {"text": y_text}, "reversedStacks": False},
        "tooltip": {"headerFormat": "<b>{point.key}</b><table>", "useHTML": True},
        "plotOptions": {
            "column": {"pointPadding": 0, "borderWidth": 1, "borderColor": "#333333"}
        },
        "credits": {"enabled": False},
    }

    if width is not None:
        chart_opts["chart"]["width"] = width

    if height is not None:
        chart_opts["chart"]["height"] = height

    if "category" in f.columns:
        # Update chart options
        chart_opts["plotOptions"]["series"] = {"stacking": "normal"}
        chart_opts["tooltip"]["pointFormat"] = (
            """
          <tr>
          <td style="padding-right:1em">{series.name}
          ({point.percentage:.0f}%)</td>
          <td style="text-align:right">{point.y:,.0f} """
            + currency
            + """
          </td>
          </tr>
          """
        )
        chart_opts["tooltip"]["footerFormat"] = (
            """
          <tr>
          <td style="padding-right:1em">Stack total</td>
          <td style="text-align:right">{point.total:,.0f} """
            + currency
            + """
          </td>
          </tr></table>
          """
        )
        chart_opts["tooltip"]["shared"] = False

        # Create data series.
        # Split income and expense into two stacks, each split by category.
        for column in ["income", "expense"]:
            # Sort categories by greatest value to least
            g = (
                f.groupby("category")
                .sum()
                .reset_index()
                .sort_values(column, ascending=False)
            )
            categories = g.loc[g[column] > 0, "category"].unique()
            n = len(categories)
            colors = get_colors(column, n)
            cond1 = f[column] > 0
            for category, color in zip(categories, colors):
                cond2 = (cond1 | f[column].isnull()) & (f["category"] == category)
                g = f[cond2].copy()
                name = "{!s} {!s}".format(column.capitalize(), category)
                series_opts = {
                    "name": name,
                    "color": color,
                    "series_type": "column",
                    "stack": column,
                    "borderColor": "white",
                }
                chart.add_data_set(g[column].values.tolist(), **series_opts)

        # Aggregate balance
        def my_agg(group):
            d = {}
            d["balance"] = group["balance"].iat[0]
            return pd.Series(d)

        g = f.groupby("date")["balance"].first().reset_index()
        series_opts = {
            "name": "Balance",
            "color": get_colors("balance", 1)[0],
            "series_type": "line",
            "borderColor": "white",
        }
        chart.add_data_set(g["balance"].values.tolist(), **series_opts)

    else:
        # Update chart options
        chart_opts["tooltip"]["pointFormat"] = (
            """
          <tr>
          <td style="padding-right:1em">{series.name}</td>
          <td style="text-align:right">{point.y:,.0f} """
            + currency
            + """
          </td>
          </tr>
          """
        )
        chart_opts["tooltip"]["footerFormat"] = "</table>"
        chart_opts["tooltip"]["shared"] = True

        # Create data series
        for column in ["income", "expense", "balance"]:
            series_opts = {
                "color": get_colors(column, 1)[0],
                "name": column.split("_")[-1].capitalize(),
                "series_type": "line" if column == "balance" else "column",
            }
            chart.add_data_set(f[column].values.tolist(), **series_opts)

    chart.set_dict_options(chart_opts)

    return chart
