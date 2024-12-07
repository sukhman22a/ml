# import pandas as pd

# df= pd.read_csv("data/Book1.csv")
# # ['Brand', 'Contact', 'Order Date', 'Price per Unit', 'Units Sold','Sales']
# x= df.to_string
# y = df.columns
# price_per_unit = df["Price per Unit"].str.replace("$",'').astype(float)

# units_sold = df["Units Sold"].str.replace(",","").astype(int)
# revenue = ((units_sold)*price_per_unit)
# total_revenue = "$" + "{:,}".format(revenue.sum())

# # print(revenue)
# df["Revenue"] = "$" + revenue.apply(lambda x: "{:,}".format(x))
# df["Order Date"] = pd.to_datetime(df["Order Date"] , dayfirst=True)
# df["Weekday"] = pd.DatetimeIndex(df["Order Date"]).weekday
# weekday_sale = df.groupby('Weekday')
# df.loc['Total'] = pd.Series(total_revenue, index=['Revenue'])
# # print(df)

# top_sales = df.sort_values("Units Sold",ascending=False)
# top_5_sales = top_sales.head(5)
# # top_5_sales = df.sort_values("Units Sold",ascending=False).head(5)
# df_2 = pd.read_csv("data/Book2.csv")
# df_df2 = df.merge(df_2, how = 'outer', on='Contact')
# print(df_df2)
