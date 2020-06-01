import os
from VisualizeData import *


start_year = 2000

def main():
    company_list = []
    plot_objects = {}
    for company_file in ["MSFT.csv","IBM.csv"]:
        company = company_file.split('.')[0]
        company_list.append(company)
        plot_objects[company] = PlotData(company=company)

    for company in company_list:
        print ("Company Name ", company)
        do_work(company,plot_objects[company])
    print("Done!")


def do_work(company,plot_data):
    plot_data.plot_complete_history()
    plot_data.plot_prices_data(start_year=start_year, end_year=2017)
    plot_data.plot_normalized_prices(first_year=start_year, last_year=2017)
    plot_data.plot_gp_predictions(train_start=start_year, train_end=2017, pred_year=2018)
    plot_data.plot_prices_data(start_year=start_year, end_year=2019)
    plot_data.plot_gp_predictions(train_start=start_year, train_end=2019, pred_year=2019, pred_quarters=[3, 4])
    print(company + ' summary done!')

if __name__ == "__main__":
    main()
