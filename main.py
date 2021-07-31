# Please run every function one by one
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("WXAgg")
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColorma

import seaborn as sns
from math import pi

stores_df = pd.read_csv("StoreMarketing.csv")

stores_df = pd.concat([stores_df, pd.read_csv("StoreOverheads.csv")["Overheads (£)"]], axis=1)
stores_df = pd.concat([stores_df, pd.read_csv("StoreSize.csv")["Size (msq)"]], axis=1)
stores_df = pd.concat([stores_df, pd.read_csv("StoreStaff.csv")["Staff"]], axis=1)

daily_customers_df = pd.read_csv('DailyCustomers.csv')

sum_of_daily_customer = daily_customers_df.loc[:, daily_customers_df.columns != 'Date'].sum(axis=0, skipna=True)
stores_name = sum_of_daily_customer.index.tolist()

def radar_chart():

    labels = stores_name
    daily_customer_maen = daily_customers_df.mean().values.flatten().tolist()
    daily_customer_maen += daily_customer_maen[:1]
    angle = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angle += angle[:1]

    figure, axis = plot.subplots(nrows=1, ncols=1, figsize=(12, 10),
                                 subplot_kw=dict(polar=True))

    plot.xticks(angle[:-1], labels, color='black', size=10)

    plot.yticks([-230, 0, 230, 460, 690, 920, 1150], ['-230', '0', '230', '460', '690', '920', '1150'],
                color='red', size=10)
    plot.ylim(-230, 1150)
    axis.set_rlabel_position(30)

    axis.plot(angle, daily_customer_maen, linewidth=1, linestyle='solid')
    axis.fill(angle, daily_customer_maen, 'skyblue', alpha=0.4)

    plot.savefig('radar.svg')
    figure.tight_layout()
    plot.show()


def grouped_bar_chart():
    # plot.style.use('ggplot')
    marketing_expense = stores_df["Marketing (£)"].tolist()
    marketing_expense = list(np.array(np.asarray(marketing_expense) / 1000, int))

    overhead_cost = stores_df["Overheads (£)"].tolist()
    overhead_cost = list(np.array(np.array(overhead_cost) / 1000, int))

    label_range = np.arange(len(stores_name))
    width = 0.35

    figure, axis = plot.subplots(figsize=(16, 8))
    bar1 = axis.bar(label_range - width / 2, marketing_expense, width, label='Marketing Expense')
    bar2 = axis.bar(label_range + width / 2, overhead_cost, width, label='Overheads Expense')

    axis.set_xlabel('Stores', size=12)
    axis.set_ylabel('Expense in thousands(1000) Â£', size=12)
    axis.set_title('For Year 2019 Annual Expense of Marketing vs Overheads')
    axis.set_xticks(label_range)
    plot.xticks(size=8, color='black', weight='bold')
    axis.set_xticklabels([str(i) for i in stores_name], rotation=90)
    axis.legend()

    def bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            axis.annotate('{}'.format(height),
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points", ha='center', va='bottom')

    bar_labels(bar1)
    bar_labels(bar2)

    plot.savefig('grouped_barchart.svg')
    plot.show()


def co_rel_matrix():

    p = sns.pairplot(stores_df, kind='reg', height=1.5, aspect=2, plot_kws=dict(marker="."), diag_kws=dict(fill=False))
    # p.fig.set_size_inches(12,6)
    p.fig.suptitle("Correlation Matrix")
    plot.savefig('correlation_matrix.svg')
    plot.show()


def scatter_plot():

    colors = np.array(daily_customers_df.mean().values.flatten().tolist())+200
    sizes = (stores_df["Marketing (£)"].add(stores_df["Overheads (£)"])/100).astype(int)
    plot.figure(figsize=(12, 6))
    axis = plot.scatter(stores_df["Staff"], stores_df["Size (msq)"],
                 sizes=sizes, c=colors, alpha=0.9, cmap='Blues', vmin=0, vmax=1000)
    plot.title("Staff VS Store's Size Having Expenses of Marketing and Overheads in 2019")
    plot.xlabel('Number of Staff')
    plot.ylabel('Size (msq)')

    s1 = plot.scatter([], [], s=50, marker='o', color='#80bfff')
    s2 = plot.scatter([], [], s=100, marker='o', color='#80bfff')
    s3 = plot.scatter([], [], s=200, marker='o', color='#80bfff')
    s4 = plot.scatter([], [], s=400, marker='o', color='#80bfff')
    s5 = plot.scatter([], [], s=800, marker='o', color='#80bfff')

    plot.legend((s1, s2, s3, s4, s5),
                ('75','150', '300', '600', '1200'),
                scatterpoints=1,
                title="Marketing + Overheads expense in 100",
                ncol=6,
                borderpad=1,
                fontsize=11)
    cbar = plot.colorbar(axis)
    cbar.set_label('Average of Visitors in 2019', rotation=270)
    plot.savefig('scatter_plot.svg')
    plot.show()


def bar_chart():

    plot.style.use('ggplot')

    x_position = [x for x, _ in enumerate(stores_name)]

    f, ax = plot.subplots(figsize=(14, 6))
    plot.bar(x_position, sum_of_daily_customer.tolist(), color='green', align='center', width=0.5)
    plot.ylabel("Number of Customers")
    plot.xlabel("Stores")
    plot.title("Total Number of Customers from each Store")

    plot.xticks(x_position, [str(x) for x in stores_name], rotation=90)
    plot.savefig('bar_chart.svg')
    plot.show()


def box_vs_violin():
    Marketing_Overheads = [stores_df["Marketing (£)"].tolist(), stores_df["Overheads (£)"].tolist()]
    figure, axis = plot.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot.suptitle("Overheads and Marketing's Expense")
    axis[0].set_ylabel('Expense in (Â£)')

    axis[0].violinplot(Marketing_Overheads, showmeans=False, showmedians=True)
    axis[1].boxplot(Marketing_Overheads)
    axis[0].set_title('Violin plot')
    axis[1].set_title('Box plot')


    for item in axis:
        item.yaxis.grid(True)
        item.set_xticks([y + 1 for y in range(len(Marketing_Overheads))])

    plot.setp(axis, xticks=[y + 1 for y in range(len(Marketing_Overheads))],
             xticklabels=['Marketing', 'Overheads'])
    plot.savefig('box_vs_violin.svg')
    plot.show()


def stacked_bar_chart():
    plot.style.use('ggplot')
    daily_customers_df['Date']= pd.to_datetime(daily_customers_df['Date'])
    average_customer_month_wise_df = daily_customers_df.groupby([daily_customers_df['Date'].dt.year.rename('year'), daily_customers_df['Date'].dt.month.rename('month')]).agg({'sum'})
    daily_customer_2d_array = average_customer_month_wise_df.to_numpy()

    month_names = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    x_position = [x for x, _ in enumerate(stores_name)]

    figure, axis = plot.subplots(1, sharey=True, sharex=True, figsize=(14, 6))
    figure.subplots_adjust(bottom=0.2)

    plot.xticks(x_position, [str(x) for x in stores_name], rotation=90)


    p = []

    def subplot(matrix, ax, title):
        renderer_bar = []
        bottoms = np.cumsum(np.vstack((np.zeros(matrix.shape[1]), matrix)), axis=0)[:-1]
        for ind, item in enumerate(matrix):
            row = ax.bar(np.arange(matrix.shape[1]), item, width=0.5, bottom=bottoms[ind])
            renderer_bar.append(row)
        ax.set_title(title)
        return renderer_bar

    p.extend(subplot(daily_customer_2d_array, axis, "Customers's Visit in each Month of 2019"))

    axis.set_ylabel('Number of Customers')
    axis.set_xlabel("Stores")


    figure.legend(p, (month_names), bbox_to_anchor=(0.5, 0), loc='lower center', ncol=12)
    plot.savefig('stacked_bar_chart.svg')
    plot.show()


def histogram():

    Marketing = np.array(stores_df["Marketing (£)"].tolist())
    Overheads = np.array(stores_df["Overheads (£)"].tolist())

    figure, axis = plot.subplots(1, 2, sharey=True, tight_layout=True, figsize=(10, 5))
    plot.suptitle("Overheads and Marketing's Expense")

    axis[0].hist(Marketing, bins=20)
    axis[0].set_title("Marketing's Expense in (Â£)")

    axis[1].hist(Overheads, bins=20)
    axis[1].set_title("Overheads's Expense in (Â£)")

    plot.savefig('histogram.svg')
    plot.show()

if __name__ == '__main__':
    radar_chart()
    grouped_bar_chart()
    co_rel_matrix()
    scatter_plot()
    bar_chart()
    stacked_bar_chart()
    box_vs_violin()
    histogram()
    print("End")
